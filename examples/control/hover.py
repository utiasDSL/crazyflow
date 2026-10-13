"""Fly a single drone to a fixed position and hold it with state control.

This is the minimal end-to-end loop: create a `Sim`, reset it, apply a state command, and step
forward.
"""

import os

os.environ["SCIPY_ARRAY_API"] = "1"

import numpy as np
from scipy.spatial.transform import Rotation as R

from crazyflow.control import Control
from crazyflow.sim import Dynamics, Sim


def main():
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        dynamics=Dynamics.first_principles,
        control=Control.state,
        freq=500,
        attitude_freq=500,
        state_freq=100,
        device="cpu",
    )

    sim.reset()
    duration = 5.0
    fps = 60

    # State cmd is [x, y, z, vx, vy, vz, ax, ay, az, qx, qy, qz, qw, wx, wy, wz]
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 16))
    cmd[..., 9:13] = R.from_euler("z", 0.0).as_quat()
    cmd[..., :3] = 0.1

    for i in range(int(duration * sim.control_freq)):
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
