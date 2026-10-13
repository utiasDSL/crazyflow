"""Command body-frame angular rates and collective thrust.

The body rate controller keeps the attitude terms of the Mellinger controller, which level the
drone at its current yaw and counteract the commanded rates. The example sets their `kR` and `ki_m`
gains to zero to track the body rates.
"""

import os

os.environ["SCIPY_ARRAY_API"] = "1"

from functools import partial

import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation as R

from crazyflow.control import Control, parametrize
from crazyflow.control.mellinger import state2attitude
from crazyflow.sim import Sim

kp_att = 8.0  # Proportional gain from the attitude error to body rates


def trajectory(t: float, pos_start: np.ndarray) -> np.ndarray:
    """Compute the full state command of a circle with a slow climb."""
    cmd = np.zeros(16)
    cmd[:3] = pos_start + np.array([np.cos(t) - 1, np.sin(t), 0.2 * t])
    cmd[3:6] = np.array([-np.sin(t), np.cos(t), 0.2])
    cmd[6:9] = np.array([-np.cos(t), -np.sin(t), 0.0])
    cmd[9:13] = R.from_euler("z", t).as_quat()
    return cmd


def control(quat: np.ndarray, rpyt: np.ndarray) -> np.ndarray:
    """Convert an attitude command into body rates with a proportional attitude loop."""
    rot_err = (R.from_quat(quat).inv() * R.from_euler("xyz", rpyt[:3])).as_rotvec()
    return np.concatenate([kp_att * rot_err, rpyt[3:]])


def main():
    sim = Sim(control=Control.body_rate, body_rate_freq=250)
    # The attitude terms of the body rate controller level the drone at the current yaw and would
    # counteract the commanded rates. Disable them to track body rates.
    body_rate = sim.data.controls.body_rate
    params = body_rate.params | {"kR": jnp.zeros(3), "ki_m": jnp.zeros(3)}
    controls = sim.data.controls.replace(body_rate=body_rate.replace(params=params))
    sim.data = sim.data.replace(controls=controls)
    sim.build_default_data()
    sim.reset()
    duration = 6.5
    fps = 60

    # We use the Mellinger position controller to generate attitude commands, which we then convert
    # to body rates. This could be any controller that outputs [w_x, w_y, w_z, thrust].
    position_ctrl = partial(parametrize(state2attitude, sim.drone), ctrl_freq=sim.control_freq)
    pos_err_i = np.zeros(3)
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))  # [wx, wy, wz, thrust]
    pos_start = np.asarray(sim.data.states.pos[0, 0])
    for i in range(int(duration * sim.control_freq)):
        pos, quat = np.asarray(sim.data.states.pos[0, 0]), np.asarray(sim.data.states.quat[0, 0])
        vel = np.asarray(sim.data.states.vel[0, 0])
        ref = trajectory(i / sim.control_freq, pos_start)
        rpyt, pos_err_i = position_ctrl(pos, quat, vel, ref, pos_err_i)
        cmd[0, 0, :] = control(quat, rpyt)
        sim.body_rate_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
