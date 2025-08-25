from collections import deque

import numpy as np

from crazyflow.sim import Physics, Sim
from crazyflow.sim.visualize import draw_line


def main():
    """Spawn 25 drones in one world and render each with a trace behind it."""
    n_worlds, n_drones = 1, 25
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=Physics.so_rpy, device="cpu")
    fps = 60
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))
    cmd[..., 3] = sim.data.params.mass[0, 0, 0] * 9.81 * 0.5
    rgbas = np.random.default_rng(0).uniform(0, 1, (n_drones, 4))
    rgbas[..., 3] = 1.0

    pos = deque(maxlen=16)

    for i in range(int(5 * sim.control_freq)):
        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if i % 20 == 0:
            pos.append(sim.data.states.pos[0, :])
        if ((i * fps) % sim.control_freq) < fps:
            lines = np.array(pos)
            for i in range(n_drones):
                draw_line(sim, lines[:, i, :], rgbas[i, :], start_size=0.3, end_size=3.0)
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
