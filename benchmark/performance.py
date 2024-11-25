import jax
import numpy as np
from pyinstrument import Profiler
from pyinstrument.renderers.html import HTMLRenderer

from crazyflow.sim.core import Sim


def profile_step(sim: Sim, n_steps: int, device: str):
    device = jax.devices(device)[0]
    ndim = 13 if sim.control == "state" else 4
    control_fn = sim.state_control if sim.control == "state" else sim.attitude_control
    cmd = np.zeros((sim.n_worlds, sim.n_drones, ndim))
    # Ensure JIT compiled dynamics and control
    sim.reset()
    control_fn(cmd)
    sim.step()
    control_fn(cmd)
    sim.step()
    sim.reset()
    jax.block_until_ready(sim.states.pos)

    profiler = Profiler()
    profiler.start()

    for _ in range(n_steps):
        control_fn(cmd)
        # sim.reset()
        sim.step()
        jax.block_until_ready(sim.states.pos)
    profiler.stop()
    renderer = HTMLRenderer()
    renderer.open_in_browser(profiler.last_session)


def main():
    device = "cpu"
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        physics="analytical",
        control="state",
        controller="emulatefirmware",
        device=device,
    )
    profile_step(sim, 1000, device)
    # old | new
    # sys_id + attitude:
    # 0.61 reset, 0.61 step  |  0.61 reset, 0.61 step
    # sys_id + state:
    # 14.53 step, 0.53 reset |  0.75 reset, 0.88 step

    # Analytical + attitude:
    # 0.75 reset, 9.38 step  |  0.75 reset, 0.89 step
    # Analytical + state:
    # 0.75 reset, 15.1 step  |  0.75 reset, 0.5 step


if __name__ == "__main__":
    main()
