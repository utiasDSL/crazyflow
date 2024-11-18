import numpy as np

from crazyflow.control.controller import Control, Controller
from crazyflow.sim.core import Sim
from crazyflow.sim.physics import Physics


def main():
    sim = Sim(
        n_worlds=1,
        n_drones=1,
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

    for i in range(int(duration * sim.freq)):
        if sim.controllable:
            # Emulate firmware state cmd is [x, y, z, qx, qy, qz, qw, vx, vy, vz]
            cmd = np.array([[[0.0, 0.0, 0.2, 0, 0, 0, 1, 0, 0, 0]]])
            sim.state_control(cmd)
        sim.step()
        if i * fps % sim.freq < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
