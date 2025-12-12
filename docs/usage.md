# Quickstart

Minimal example (create sim, reset, control step, render)

```python
from crazyflow.sim import Sim
import numpy as np

# Minimal sim
sim = Sim()
sim.reset()

# simple state command (n_worlds=1, n_drones=1, 13 state cmd)
cmd = np.zeros((1, 1, 13), dtype=np.float32)
cmd[0, 0, 2] = 0.5  # 0.5m hovering setpoint
sim.state_control(cmd)
sim.step(1)   # step one control cycle
sim.render()  # opens a MuJoCo window (or render to images)
sim.close()
```

A slightly more explicit example where you adjust sim settings (control, integrator, physics):

```python
from crazyflow.sim import Sim
from crazyflow.control import Control
from crazyflow.sim.integration import Integrator
from crazyflow.sim.physics import Physics
import numpy as np

sim = Sim(
    n_drones=4,
    n_worlds=2,
    control=Control.state,
    integrator=Integrator.rk4,
    physics=Physics.first_principles,
    drone_model="cf2x_T350",
)
sim.reset()

duration = 5.0
fps = 60

# send a simple state command (shape: n_worlds, n_drones, 13)
cmd = np.zeros((4, 2, 13), dtype=np.float32)
cmd[..., 2] = sim.data.states.pos[..., :3] + 0.5  # 0.5m hovering setpoint
for i in range(int(duration * sim.control_freq)):
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
sim.close()
```

For the full API and configuration options (Sim args, control helpers, render options), see the API reference: [API reference](api/index.md)