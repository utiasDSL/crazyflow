import jax
import numpy as np
from pyinstrument import Profiler
from pyinstrument.renderers.html import HTMLRenderer

from crazyflow.sim.core import Sim


def profile_step(sim: Sim, n_steps: int, device: str):
    device = jax.devices(device)[0]
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))
    # Ensure JIT compiled dynamics and control
    sim.reset()
    sim.attitude_control(cmd)
    sim.step()
    sim.reset()
    jax.block_until_ready(sim._mjx_data)

    profiler = Profiler()
    profiler.start()

    for _ in range(n_steps):
        sim.attitude_control(cmd)
        sim.step()
        jax.block_until_ready(sim._mjx_data)
    profiler.stop()
    renderer = HTMLRenderer()
    renderer.open_in_browser(profiler.last_session)


def main():
    device = "cpu"
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        physics="sys_id",
        control="attitude",
        controller="emulatefirmware",
        device=device,
    )
    profile_step(sim, 1000, device)


if __name__ == "__main__":
    main()
