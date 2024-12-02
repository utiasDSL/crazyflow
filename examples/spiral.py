import numpy as np

from crazyflow.control.controller import Control
from crazyflow.sim.core import Sim


def main():
    sim = Sim(n_worlds=1, n_drones=4, control=Control.state, device="cpu")
    sim.reset()
    duration = 5.0
    fps = 60

    start_xy = sim.states.pos[0, :, :2]
    for i in range(int(duration * sim.freq)):
        if sim.controllable:
            omega = i / sim.freq
            circle = np.array([np.cos(omega) - 1, np.sin(omega)])
            cmd = np.zeros((1, sim.n_drones, 13))
            cmd[:, :, :2] = start_xy + circle
            cmd[:, :, 2] = 0.2 * (i / sim.freq)
            sim.state_control(cmd)
        sim.step()
        if i * fps % sim.freq < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
