# Simulator Overview

## Worlds and drones

Crazyflow organises simulation state into a two-dimensional batch: **worlds × drones**.

- **`n_worlds`** — number of independent simulation environments. Each world has its own dynamics state and evolves independently. Use this to run domain randomisation, parallel RL rollouts, or MPPI sampling.
- **`n_drones`** — number of drones per world. All drones in a world share the same dynamics tick but have independent states.

Every state array has shape `(n_worlds, n_drones, feature_dim)`. To read the position of drone 0 in world 2:

```python
from crazyflow.sim import Sim

sim = Sim(n_worlds=4, n_drones=2)
sim.reset()

pos = sim.data.states.pos[2, 0]  # world 2, drone 0 → shape (3,)
```

## SimData

All simulation state is stored in `sim.data`, a `SimData` pytree. The main sub-trees are:

| Field | Type | Description |
|---|---|---|
| `states` | `SimState` | Current kinematic state of every drone |
| `states_deriv` | `SimStateDeriv` | Time derivatives computed by the dynamics |
| `controls` | `SimControls` | Staged commands and controller state |
| `params` | `SimParams` | Physical parameters (mass, inertia, motor constants, …) |
| `core` | `SimCore` | Metadata: step count, frequency, RNG key, device |

### SimState fields

| Field | Shape | Units |
|---|---|---|
| `pos` | `(N, M, 3)` | Position in world frame, meters |
| `quat` | `(N, M, 4)` | Orientation quaternion, scalar-last `xyzw` |
| `vel` | `(N, M, 3)` | Linear velocity, m/s |
| `ang_vel` | `(N, M, 3)` | Angular velocity in body frame, rad/s |
| `force` | `(N, M, 3)` | External force applied to the drone body, N |
| `torque` | `(N, M, 3)` | External torque applied to the drone body, Nm |
| `rotor_vel` | `(N, M, 4)` | Motor angular velocities, RPM |

where `N = n_worlds` and `M = n_drones`.

## Immutability and `data.replace`

`SimData` is a JAX pytree. All fields are immutable — operations return a new `SimData` rather than modifying in place. This is what makes the simulation compatible with `jax.jit`, `jax.grad`, and `jax.vmap`.

To modify a field, use `.replace()`:

```python
from crazyflow.sim import Sim

sim = Sim(n_worlds=1, n_drones=1)
sim.reset()

import jax.numpy as jnp

new_pos = sim.data.states.pos.at[..., 2].add(1.0)
sim.data = sim.data.replace(states=sim.data.states.replace(pos=new_pos))
```

## Simulation frequency and the control stack

`freq` sets the dynamics update rate in Hz. The control stack is executed as part of each dynamics step, but controllers fire at their own sub-frequency rather than every tick. For example, with `freq=500` and `state_freq=100`, the state (Mellinger) controller runs every 5 dynamics steps, and the attitude controller runs at `attitude_freq`.

This means you can advance multiple dynamics steps in a single `sim.step(n_steps)` call and the control stack will execute at the correct rate automatically, with no manual sub-stepping required. This is also what makes fusing many steps into a single compiled call efficient.

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(freq=500, control=Control.state)
sim.reset()
cmd = np.zeros((1, 1, 13), dtype=np.float32)
sim.state_control(cmd)
sim.step(sim.freq // sim.control_freq)  # 500 // 100 = 5 dynamics steps, controller fires once
```

## The step and reset pipelines

Each call to `sim.step()` runs `sim.step_pipeline`, an ordered collection of named, pure JAX functions that transform `SimData`. By default it contains the control conversion functions, the numerical integrator, a clip of the rotor state to the physical motor limits, a step counter, and a floor clip. Similarly, `sim.reset_pipeline` is applied during `sim.reset()` and is empty by default.

Both pipelines can be extended with custom functions for disturbances, domain randomization, or logging without modifying the core simulator.

See [Pipelines](pipelines.md) for full details.

## Next steps

- [Object-Oriented API](oo-api.md) — all `Sim` methods in detail
- [Functional API](functional-api.md) — working with `SimData` directly inside JAX transformations
- [Pipelines](pipelines.md) — extending the step and reset pipelines
