# Quick Start

This page walks through a complete minimal workflow: create a simulator, send a position command, step it forward, and read back the drone's state.

## Create a simulator

`Sim` is the top-level object. All configuration is provided at construction time: dynamics, control mode, simulation frequency, number of parallel worlds, and number of drones per world.

```python
from crazyflow.sim import Sim

sim = Sim(n_worlds=1, n_drones=1, freq=500)
sim.reset()
```

`reset()` initialises all worlds to the default state: the drone is at the origin, upright, with zero velocity.

## State and command

The default control mode is `Control.state`. A state command is a 13-element vector that sets the desired position, velocity, acceleration, yaw, and body angular rates.

| Index | Variable | Units |
|---|---|---|
| 0–2 | Position \(x, y, z\) | m |
| 3–5 | Velocity \(\dot{x}, \dot{y}, \dot{z}\) | m/s |
| 6–8 | Acceleration \(\ddot{x}, \ddot{y}, \ddot{z}\) | m/s² |
| 9 | Yaw | rad |
| 10 | Roll rate | rad/s |
| 11 | Pitch rate | rad/s |
| 12 | Yaw rate | rad/s |

The command array has shape `(n_worlds, n_drones, 13)`.

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(n_worlds=1, n_drones=1, freq=500, control=Control.state)
sim.reset()

cmd = np.zeros((1, 1, 13), dtype=np.float32)
cmd[0, 0, 2] = 0.5  # target height: 0.5 m
```

## Step the simulation

`state_control` stages the command. `step` advances the simulation by the given number of dynamics steps. Calling `sim.step(sim.freq // sim.control_freq)` advances exactly one control cycle.

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(n_worlds=1, n_drones=1, freq=500, control=Control.state)
sim.reset()

cmd = np.zeros((1, 1, 13), dtype=np.float32)
cmd[0, 0, 2] = 0.5

for _ in range(10):
    sim.state_control(cmd)
    sim.step(sim.freq // sim.control_freq)
```

## Read back state

All simulation state lives in `sim.data.states`. Arrays are indexed as `[world, drone, :]`.

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(n_worlds=1, n_drones=1, freq=500, control=Control.state)
sim.reset()

cmd = np.zeros((1, 1, 13), dtype=np.float32)
cmd[0, 0, 2] = 0.5

for _ in range(10):
    sim.state_control(cmd)
    sim.step(sim.freq // sim.control_freq)

pos = sim.data.states.pos[0, 0]  # (3,)  — position in meters
quat = sim.data.states.quat[0, 0]  # (4,)  — quaternion xyzw
vel = sim.data.states.vel[0, 0]  # (3,)  — linear velocity m/s
ang_vel = sim.data.states.ang_vel[0, 0]  # (3,)  — angular velocity rad/s
```

## Simulate multiple worlds

Increase `n_worlds` to run independent simulations in a single batched call. All state arrays gain a leading world dimension.

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(n_worlds=4, n_drones=1, freq=500, control=Control.state)
sim.reset()

cmd = np.zeros((4, 1, 13), dtype=np.float32)
cmd[:, 0, 2] = np.array([0.2, 0.4, 0.6, 0.8])  # different target heights per world

for _ in range(10):
    sim.state_control(cmd)
    sim.step(sim.freq // sim.control_freq)

pos = sim.data.states.pos[:, 0, :]  # (4, 3) — position of drone 0 in each world
```

## Simulate multiple drones

Increase `n_drones` to place multiple drones inside a single world. Each drone has its own independent state; all receive commands from the same `(n_worlds, n_drones, 13)` array.

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(n_worlds=1, n_drones=4, freq=500, control=Control.state)
sim.reset()

cmd = np.zeros((1, 4, 13), dtype=np.float32)
cmd[0, :, 2] = np.array([0.2, 0.4, 0.6, 0.8])  # different height per drone

for _ in range(10):
    sim.state_control(cmd)
    sim.step(sim.freq // sim.control_freq)

pos = sim.data.states.pos[0, :, :]  # (4, 3) — all 4 drones in world 0
```

## Next steps

- [Object-Oriented API](../user-guide/oo-api.md) — all control modes, rendering, and reset
- [Functional API](../user-guide/functional-api.md) — purely functional interface for use inside JAX transformations
- [Dynamics](../user-guide/dynamics/index.md) — choosing between first-principles and fitted dynamics
- [Examples](../examples/index.md) — runnable scripts
