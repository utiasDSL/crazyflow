from enum import Enum

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.spatial.transform import Rotation as R


GRAVITY = 9.81
SYS_ID_PARAMS = {
    "acc_k1": 20.91,
    "acc_k2": 3.65,
    "roll_alpha": -3.96,
    "roll_beta": 4.08,
    "pitch_alpha": -6.00,
    "pitch_beta": 6.21,
    "yaw_alpha": 0.00,
    "yaw_beta": 0.00,
}


class Physics(Enum):
    DEFAULT = "mujoco"
    MUJOCO = "mujoco"
    ANALYTICAL = "analytical"
    SYS_ID = "sys_id"


def identified_dynamics(cmd: Array, pos: Array, quat: Array, vel: Array, ang_vel: Array, dt: float):
    """Dynamics model identified from data collected on the real drone.

    Contrary to the other physics implementations, this function is not based on a physical model.
    Instead, we fit a linear model to the data collected on the real drone, and predict the next
    state based on the control inputs and the current state.

    Note:
        We do not explicitly simulate the onboard controller for this model. Instead, we assume that
        its dynamics are implicitly captured by the linear model.

    Args:
        cmd: The 4D control input consisting of the desired collective thrust and attitude.
    """
    collective_thrust, attitude = cmd[0], cmd[1:]
    thrust = R.from_quat(quat).apply(jnp.array([0, 0, collective_thrust]))
    acc = thrust * SYS_ID_PARAMS["acc_k1"] + jnp.array([0, 0, SYS_ID_PARAMS["acc_k2"]])
    # TODO: Include acceleration dynamics from Haocheng's thesis
    roll_cmd, pitch_cmd, yaw_cmd = attitude
    roll_rate = SYS_ID_PARAMS["roll_alpha"] * quat[0] + SYS_ID_PARAMS["roll_beta"] * roll_cmd
    pitch_rate = SYS_ID_PARAMS["pitch_alpha"] * quat[1] + SYS_ID_PARAMS["pitch_beta"] * pitch_cmd
    yaw_rate = SYS_ID_PARAMS["yaw_alpha"] * quat[2] + SYS_ID_PARAMS["yaw_beta"] * yaw_cmd
    rpy_rates = jnp.array([roll_rate, pitch_rate, yaw_rate])
    next_pos = pos + vel * dt
    next_quat = R.from_euler("xyz", R.from_quat(quat).as_euler("xyz") + ang_vel * dt).as_quat()
    next_vel = vel + acc * dt
    next_ang_vel = ang_vel + rpy_rates * dt
    return next_pos, next_quat, next_vel, next_ang_vel
