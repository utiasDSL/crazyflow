import numpy as np

from crazyflow.control.controller import Control
from crazyflow.sim.core import Sim


def main():
    sim = Sim(control=Control.thrust)
    sim.reset()
    duration = 5.0
    fps = 60

    cmd = np.ones((sim.n_worlds, sim.n_drones, 4)) * 0.0615  # 4 individual motor thrusts
    for i in range(int(duration * sim.freq)):
        sim.thrust_control(cmd)
        sim.step()
        if ((i * fps) % sim.freq) < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
