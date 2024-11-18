import numpy as np

from crazyflow.control.controller import Control, Controller
from crazyflow.sim.core import Sim
from crazyflow.sim.physics import Physics


def main():
    sim = Sim(
        n_worlds=1,
        n_drones=4,
        physics=Physics.analytical,
        control=Control.state,
        controller=Controller.emulatefirmware,
        freq=500,
        control_freq=500,
        device="cpu",
    )
    sim.reset()
    duration = 5.0
    fps = 60

    N = int(np.ceil(np.sqrt(sim.n_drones)))
    points = np.linspace(-0.5 * (N - 1), 0.5 * (N - 1), N)
    x, y = np.meshgrid(points, points)
    grid = np.stack((x.flatten(), y.flatten()), axis=-1)
    grid = grid[: sim.n_drones]
    sim.states["pos"] = sim.states["pos"].at[..., :2].set(grid)

    for i in range(int(duration * sim.freq)):
        if sim.controllable:
            omega = i / sim.freq
            circle = np.array([np.cos(omega) - 1, np.sin(omega)])
            cmd = np.zeros((1, sim.n_drones, 13))
            cmd[:, :, :2] = grid + circle
            cmd[:, :, 2] = 0.2 * (i / sim.freq)
            sim.state_control(cmd)
        sim.step()
        if i * fps % sim.freq < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
