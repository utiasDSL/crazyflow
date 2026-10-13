"""Command roll, pitch, yaw, and collective thrust directly.

This level bypasses the Mellinger position loop and is typical for RL agents that output attitude
targets.
"""

import os

os.environ["SCIPY_ARRAY_API"] = "1"

from functools import partial

import numpy as np
from scipy.spatial.transform import Rotation as R

from crazyflow.control import Control, parametrize
from crazyflow.control.mellinger import state2attitude
from crazyflow.sim import Sim


def trajectory(t: float, pos_start: np.ndarray) -> np.ndarray:
    """Compute the full state command of a circle with a slow climb."""
    cmd = np.zeros(16)
    cmd[:3] = pos_start + np.array([np.cos(t) - 1, np.sin(t), 0.2 * t])
    cmd[3:6] = np.array([-np.sin(t), np.cos(t), 0.2])
    cmd[6:9] = np.array([-np.cos(t), -np.sin(t), 0.0])
    cmd[9:13] = R.from_euler("z", t).as_quat()
    return cmd


def main():
    sim = Sim(control=Control.attitude)
    sim.reset()
    duration = 6.5
    fps = 60

    # We use the Mellinger position controller to generate attitude commands. This could be any
    # controller that outputs [roll, pitch, yaw, thrust], e.g. a learned policy.
    position_ctrl = partial(parametrize(state2attitude, sim.drone), ctrl_freq=sim.control_freq)
    pos_err_i = np.zeros(3)
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))  # [roll, pitch, yaw, thrust]
    pos_start = np.asarray(sim.data.states.pos[0, 0])
    for i in range(int(duration * sim.control_freq)):
        pos, quat = np.asarray(sim.data.states.pos[0, 0]), np.asarray(sim.data.states.quat[0, 0])
        vel = np.asarray(sim.data.states.vel[0, 0])
        ref = trajectory(i / sim.control_freq, pos_start)
        cmd[0, 0, :], pos_err_i = position_ctrl(pos, quat, vel, ref, pos_err_i)
        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
