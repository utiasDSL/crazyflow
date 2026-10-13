"""Example where 4 drones fly upwards in a spiral motion."""

import os

os.environ["SCIPY_ARRAY_API"] = "1"

import numpy as np
from scipy.spatial.transform import Rotation as R

from crazyflow.control import Control
from crazyflow.sim import Sim


def trajectory(start_xy: np.ndarray, t: float) -> np.ndarray:
    circle = np.array([np.cos(t) - 1, np.sin(t)])
    cmd = np.zeros((*start_xy.shape[:-1], 16))
    cmd[..., 9:13] = R.from_euler("z", 0.0).as_quat()
    cmd[..., :2] = start_xy + circle
    cmd[..., 2] = 0.2 * t
    return cmd


def main():
    sim = Sim(n_drones=4, control=Control.state, integrator="rk4", dynamics="first_principles")
    sim.reset()
    duration = 5.0
    fps = 60

    start_xy = sim.data.states.pos[..., :2]
    for i in range(int(duration * sim.control_freq)):
        sim.state_control(trajectory(start_xy, i / sim.control_freq))
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
