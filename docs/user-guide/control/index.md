# Control Modes

Crazyflow provides multiple control modes, from high-level position setpoints down to direct motor commands. Each mode is selected at construction time.

## Control hierarchy

Commands flow down a hierarchy. A state command is converted to an attitude command by the Mellinger position controller, the attitude command is converted to force/torque by the geometric controller, and force/torque is converted to rotor velocities by the mixer. Body rate control feeds the geometric controller with a rate setpoint instead of an attitude, so it enters the hierarchy at the same level as attitude control. The rate setpoint in the state command is forwarded to the rate controller.

```
State (16D)
  └─ Mellinger position controller
       └─ Attitude (4D: roll, pitch, yaw, thrust) + body rates (3D: ωx, ωy, ωz)  |  Body rates (4D: ωx, ωy, ωz, thrust)
            └─ Geometric controller
                 └─ Force/torque (4D: Fc, Tx, Ty, Tz)
                      └─ Mixer
                           └─ Rotor velocities (4D: ω₁…ω₄)
```

When you select `Control.state`, the full chain runs on every control tick. When you select `Control.attitude` or `Control.body_rate`, only the lower two stages run.

## State control

```python
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(control=Control.state, state_freq=100, attitude_freq=500)
sim.reset()
```

Command shape: `(n_worlds, n_drones, 16)`

| Index | Variable | Units |
|---|---|---|
| 0–2 | Target position \(x, y, z\) | m |
| 3–5 | Target velocity \(\dot{x}, \dot{y}, \dot{z}\) | m/s |
| 6–8 | Target acceleration \(\ddot{x}, \ddot{y}, \ddot{z}\) | m/s² |
| 9–12 | Attitude quaternion \(q_x, q_y, q_z, q_w\) | |
| 13–15 | Body rates \(\omega_x, \omega_y, \omega_z\) | rad/s |

As in the firmware's full state setpoint, only the yaw of the attitude quaternion is used. The body rates are the angular velocity in the body frame. The so_rpy family ignores them.

Set unused elements to zero. The attitude quaternion must be valid. A common hover command sets only the z position:

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control
from scipy.spatial.transform import Rotation as R

sim = Sim(control=Control.state)
sim.reset()

cmd = np.zeros((1, 1, 16), dtype=np.float32)
cmd[..., 9:13] = R.from_euler("z", 0.0).as_quat()
cmd[0, 0, 2] = 1.0  # hover at 1 m

sim.state_control(cmd)
sim.step(sim.freq // sim.control_freq)
```

## Attitude control

```python
from crazyflow.sim import Sim, Dynamics
from crazyflow.control import Control

sim = Sim(control=Control.attitude, dynamics=Dynamics.so_rpy, attitude_freq=500)
sim.reset()
```

Command shape: `(n_worlds, n_drones, 4)`

| Index | Variable | Units |
|---|---|---|
| 0 | Roll setpoint | rad |
| 1 | Pitch setpoint | rad |
| 2 | Yaw setpoint | rad |
| 3 | Collective thrust | N |

For a hover command, set thrust to `mass × g`:

```python
import numpy as np
from crazyflow.sim import Sim, Dynamics
from crazyflow.control import Control

sim = Sim(control=Control.attitude, dynamics=Dynamics.so_rpy)
sim.reset()

mass = float(sim.data.params.mass[0])
cmd = np.zeros((1, 1, 4), dtype=np.float32)
cmd[0, 0, 3] = mass * 9.81

sim.attitude_control(cmd)
sim.step(sim.freq // sim.control_freq)
```

## Body rate control

Commands body-frame angular rates and a collective thrust. The Mellinger controller tracks the rates with the same gains as in attitude control. As in the firmware, its attitude terms level the drone at the current yaw. Set the `kR` and `ki_m` parameters of the body rate controller to zero to track body rates without the levelling terms, see the [body rate example](../../examples/control.md#body-rate-control). Requires `Dynamics.first_principles`.

```python
from crazyflow.sim import Sim, Dynamics
from crazyflow.control import Control

sim = Sim(control=Control.body_rate, dynamics=Dynamics.first_principles, body_rate_freq=500)
sim.reset()
```

Command shape: `(n_worlds, n_drones, 4)`

| Index | Variable | Units |
|---|---|---|
| 0 | Body rate \(\omega_x\) | rad/s |
| 1 | Body rate \(\omega_y\) | rad/s |
| 2 | Body rate \(\omega_z\) | rad/s |
| 3 | Collective thrust | N |

Zero rates and hover thrust hold the current attitude:

```python
import numpy as np
from crazyflow.sim import Sim, Dynamics
from crazyflow.control import Control

sim = Sim(control=Control.body_rate, dynamics=Dynamics.first_principles)
sim.reset()

mass = float(sim.data.params.mass[0])
cmd = np.zeros((1, 1, 4), dtype=np.float32)
cmd[0, 0, 3] = mass * 9.81

sim.body_rate_control(cmd)
sim.step(sim.freq // sim.control_freq)
```

## Force-torque control

Direct force and torque input. Requires `Dynamics.first_principles`.

Command shape: `(n_worlds, n_drones, 4)`

| Index | Variable | Units |
|---|---|---|
| 0 | Collective force \(F_c\) | N |
| 1 | Body-frame torque \(\tau_x\) | Nm |
| 2 | Body-frame torque \(\tau_y\) | Nm |
| 3 | Body-frame torque \(\tau_z\) | Nm |

```python
import numpy as np
from crazyflow.sim import Sim, Dynamics
from crazyflow.control import Control

sim = Sim(control=Control.force_torque, dynamics=Dynamics.first_principles)
sim.reset()

mass = float(sim.data.params.mass[0])
cmd = np.zeros((1, 1, 4), dtype=np.float32)
cmd[0, 0, 0] = mass * 9.81

sim.force_torque_control(cmd)
sim.step(1)
```

## Rotor velocity control

Direct motor commands. Requires `Dynamics.first_principles`.

Command shape: `(n_worlds, n_drones, 4)`

| Index | Motor | Units |
|---|---|---|
| 0–3 | Motors 0–3 angular velocity | RPM |

The hover RPM for `cf2x_L250` is approximately 15 000 RPM, but the exact value depends on drone mass.

```python
import numpy as np
from crazyflow.sim import Sim, Dynamics
from crazyflow.control import Control

sim = Sim(control=Control.rotor_vel, dynamics=Dynamics.first_principles)
sim.reset()

cmd = np.full((1, 1, 4), 15_000.0, dtype=np.float32)

sim.rotor_vel_control(cmd)
sim.step(1)
```

## Control frequency

Each control mode has its own update rate. The dynamics tick (`freq`) is always the fastest.

| Mode | Rate argument | Default |
|---|---|---|
| `state` | `state_freq` | 100 Hz |
| `attitude` | `attitude_freq` | 500 Hz |
| `body_rate` | `body_rate_freq` | 500 Hz |
| `force_torque` | `force_torque_freq` | 500 Hz |
| `rotor_vel` | — | every dynamics step |

The simulator applies a new command only when the control tick fires. Between ticks, the previous command is held. The number of dynamics steps per control tick is `freq // control_freq`.

## Using the controllers standalone

The control modes above are how the simulator drives the onboard controllers. Those controllers also live in `crazyflow.control` as a self-contained library of pure functions, usable on their own for control design, learning-based policies, or as a reference implementation, independent of `Sim`. The following guides cover that standalone API:

- [Controllers](controllers.md): the controller interface and the Mellinger pipeline
- [Mellinger controller](mellinger.md): the three stages and the body rate variant, their inputs and outputs
- [Parametrization](parametrize.md): binding a controller to a drone configuration
- [Integral errors](integral-errors.md): carrying controller state across calls
- [Batching](batching.md): evaluating many drones at once
- [JIT compilation](jit.md): compiling controllers with `jax.jit`

## Next steps

- [Functional API](../functional-api.md): running control inside JIT with `F.controllable`
- [Dynamics](../dynamics/index.md): compatibility between dynamics and control modes
