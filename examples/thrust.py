import numpy as np

from crazyflow.constants import GRAVITY, MASS
from crazyflow.control import Control
from crazyflow.sim import Sim


def main():
    sim = Sim(control=Control.thrust)
    sim.reset()
    duration = 5.0
    fps = 60

    # 4 individual motor thrusts -> Weight devided by 4, plus a small margin to accelerate slightly
    cmd = np.ones((sim.n_worlds, sim.n_drones, 4)) * (MASS + 1e-4) * GRAVITY / 4
    for i in range(int(duration * sim.control_freq)):
        sim.thrust_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
