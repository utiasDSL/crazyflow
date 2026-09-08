# Object-Oriented API

The `Sim` class is the main entry point. It provides a Python-level control loop that is easy to script and debug.

!!! note
    The OO API is not compatible with JAX transformations. If you need to run simulation inside `jax.jit`, `jax.grad`, or `jax.lax.scan`, use the [Functional API](functional-api.md) instead.

## Creating a Sim

All configuration is fixed at construction time.

```python
from crazyflow.sim import Sim, Dynamics
from crazyflow.sim.integration import Integrator
from crazyflow.control import Control

sim = Sim(
    n_worlds=1,
    n_drones=1,
    drone="cf21B_500",
    dynamics=Dynamics.first_principles,
    control=Control.state,
    integrator=Integrator.rk4,
    freq=500,  # dynamics update rate, Hz
    state_freq=100,  # state controller rate, Hz
    attitude_freq=500,  # attitude controller rate, Hz
    device="cpu",
)
sim.reset()
```

Key constructor arguments:

| Argument | Description |
|---|---|
| `n_worlds` | Number of independent parallel environments |
| `n_drones` | Drones per world |
| `drone` | Drone configuration (see `crazyflow.available_drones`) |
| `dynamics` | Dynamics |
| `control` | Control mode |
| `integrator` | Numerical integrator |
| `freq` | Dynamics frequency, Hz |
| `device` | `"cpu"` or `"gpu"` |

See [`Sim`][crazyflow.sim.Sim] in the API reference for the defaults and the full argument list.

## Control methods

All control methods take an array of shape `(n_worlds, n_drones, command_dim)` and stage it for the next `step` call.

### State control

The highest-level interface. A 13-element command sets desired position, velocity, acceleration, yaw, and angular rates. An internal Mellinger controller converts this to attitude commands.

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(n_worlds=1, n_drones=1, control=Control.state)
sim.reset()

# [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate]
cmd = np.zeros((1, 1, 13), dtype=np.float32)
cmd[0, 0, 2] = 0.5  # hover at 0.5 m

sim.state_control(cmd)
sim.step(sim.freq // sim.control_freq)
```

### Attitude control

Commands roll, pitch, yaw setpoints (rad) and a collective thrust (N). This level bypasses the position/velocity loop and is suitable for attitude tracking or RL agents that output attitude targets.

```python
import numpy as np
from crazyflow.sim import Sim, Dynamics
from crazyflow.control import Control

sim = Sim(n_worlds=1, n_drones=1, control=Control.attitude, dynamics=Dynamics.so_rpy)
sim.reset()

# [roll, pitch, yaw, collective_thrust_N]
cmd = np.zeros((1, 1, 4), dtype=np.float32)
cmd[0, 0, 3] = float(sim.data.params.mass[0]) * 9.81  # hover thrust

sim.attitude_control(cmd)
sim.step(sim.freq // sim.control_freq)
```

### Force-torque control

Direct force and torque input, useful for testing dynamics or custom controllers. Requires `Dynamics.first_principles`.

```python
import numpy as np
from crazyflow.sim import Sim, Dynamics
from crazyflow.control import Control

sim = Sim(n_worlds=1, n_drones=1, control=Control.force_torque, dynamics=Dynamics.first_principles)
sim.reset()

# [collective_force_N, torque_x_Nm, torque_y_Nm, torque_z_Nm]
cmd = np.zeros((1, 1, 4), dtype=np.float32)
cmd[0, 0, 0] = float(sim.data.params.mass[0]) * 9.81  # hover force

sim.force_torque_control(cmd)
sim.step(1)
```

### Rotor velocity control

The lowest level: directly command each motor's RPM. Requires `Dynamics.first_principles`.

```python
import numpy as np
from crazyflow.sim import Sim, Dynamics
from crazyflow.control import Control

sim = Sim(n_worlds=1, n_drones=1, control=Control.rotor_vel, dynamics=Dynamics.first_principles)
sim.reset()

# [rpm_motor_0, rpm_motor_1, rpm_motor_2, rpm_motor_3]
cmd = np.full((1, 1, 4), 15_000.0, dtype=np.float32)

sim.rotor_vel_control(cmd)
sim.step(1)
```

## Stepping and resetting

`sim.step(n_steps)` advances the simulation by `n_steps` dynamics ticks. On each tick, the full step pipeline runs, including the control stack. Controllers fire at their configured rate (e.g. the state controller at `state_freq`, the attitude controller at `attitude_freq`), not on every dynamics tick. Between controller ticks, the previously staged command is held.

Passing more steps to a single `step(n_steps)` call is more efficient than multiple `step(1)` calls: XLA compiles the full loop into a single kernel. If you have staged a control command and do not need to set a new one, you can advance the simulation by any number of steps and the controllers will continue firing at the correct rate.

!!! note
    Changing `n_steps` between calls triggers recompilation. Keep it consistent inside a training or evaluation loop.

`sim.reset()` reinitialises all worlds to their default state. Pass a boolean mask of shape `(n_worlds,)` to reset only selected worlds: `True` resets that world, `False` leaves it unchanged. This is useful in RL training loops where episodes end at different times.

A full reset restores everything except the rng key. A mask selects along the world axis, so it only restores per-world arrays, and leaves parameters shared by all worlds untouched. See [The world axis](world-axis.md) for more details.

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(n_worlds=4, n_drones=1, control=Control.state)
sim.reset()  # reset all worlds

# Stage a command and advance 50 dynamics steps (controllers fire at their rate)
cmd = np.zeros((4, 1, 13), dtype=np.float32)
cmd[..., 2] = 0.5
sim.state_control(cmd)
sim.step(50)

# Reset only worlds 0 and 2, leaving 1 and 3 running
import jax.numpy as jnp

mask = jnp.array([True, False, True, False])
sim.reset(mask=mask)
```

## Reading state

Access any state field through `sim.data.states`:

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(n_worlds=2, n_drones=3, control=Control.state)
sim.reset()

cmd = np.zeros((2, 3, 13), dtype=np.float32)
for _ in range(10):
    sim.state_control(cmd)
    sim.step(sim.freq // sim.control_freq)

# All drones in all worlds
pos = sim.data.states.pos  # (2, 3, 3)
quat = sim.data.states.quat  # (2, 3, 4)
vel = sim.data.states.vel  # (2, 3, 3)

# Drone 1 in world 0
pos_w0_d1 = sim.data.states.pos[0, 1]  # (3,)
```

## Rendering

`sim.render()` opens an interactive MuJoCo viewer or returns an image array for offscreen rendering.

<!-- notest: requires rendering -->
```{ .python notest }
sim.render()                          # interactive window, world 0
sim.render(mode="rgb_array")          # returns (H, W, 3) uint8
sim.render(mode="depth_array")        # returns (H, W) float32
sim.render(world=1, camera="front")   # different world or named camera
sim.close()                           # close the viewer
```

## Domain randomization

Define physical-parameter randomization as a reset pipeline stage. Each stage receives the restored `SimData`, the default `SimData`, and an optional world mask, and returns the randomized data:

```python
import jax
from jax import Array

from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.pipeline import append_fn
from crazyflow.utils import leaf_replace


def randomize_mass(data: SimData, default_data: SimData, mask: Array | None = None) -> SimData:
    key, mass_key = jax.random.split(data.core.rng_key)
    shape = (data.core.n_worlds, data.core.n_drones, 1)  # One mass per drone
    mass = default_data.params.mass + jax.random.normal(mass_key, shape) * 2e-3
    params = leaf_replace(data.params, mask, mass=mass)
    return data.replace(params=params, core=data.core.replace(rng_key=key))


sim = Sim(n_worlds=4, n_drones=1)
append_fn(sim.reset_pipeline, randomize_mass)
sim.build_reset_fn()
sim.reset()  # randomizes every world
```

Passing a boolean mask to `sim.reset(mask=mask)` randomizes only the worlds being reset. Every parameter of every dynamics model can be randomized per world this way. See [Randomization and disturbances](pipelines.md#randomization-and-disturbances) for the details and the recompilation this might cause, and the [domain randomization example](../examples/index.md#domain-randomization) for a complete simulation.

## Next steps

- [Functional API](functional-api.md) — run simulation inside `jax.jit` and `jax.grad`
- [Pipelines](pipelines.md) — insert custom stages for disturbances and logging
