from enum import Enum

import jax.numpy as jnp
from jax import Array
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.control.controller import ARM_LEN, KF, KM

GRAVITY = 9.81
MASS = 0.027
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


class Physics(str, Enum):
    """Physics mode for the simulation."""

    mujoco = "mujoco"
    analytical = "analytical"
    sys_id = "sys_id"
    default = mujoco


def identified_dynamics(
    cmd: Array, pos: Array, quat: Array, vel: Array, ang_vel: Array, dt: float
) -> tuple[Array, Array, Array, Array]:
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
    rot = R.from_quat(quat)
    thrust = rot.apply(jnp.array([0, 0, collective_thrust]))
    drift = rot.apply(jnp.array([0, 0, 1]))
    a1, a2 = SYS_ID_PARAMS["acc_k1"], SYS_ID_PARAMS["acc_k2"]
    acc = thrust * a1 + drift * a2 - jnp.array([0, 0, GRAVITY])
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


def analytical_dynamics(
    rpms: Array,
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    mass: Array,
    J: Array,
    J_INV: Array,
    dt: float,
) -> tuple[Array, Array, Array, Array]:
    """Analytical dynamics model."""
    # TODO: Remove rpms, use forces and torques directly.
    rot = R.from_quat(quat)
    rpy_rates = rot.apply(ang_vel, inverse=True)  # Now in body frame
    # Compute forces and torques.
    forces = jnp.array(rpms**2) * KF
    thrust = jnp.array([0, 0, jnp.sum(forces)])
    thrust_world_frame = rot.apply(thrust)
    force_world_frame = thrust_world_frame - jnp.array([0, 0, GRAVITY * mass[0]])
    z_torques = jnp.array(rpms**2) * KM
    z_torque = z_torques[0] - z_torques[1] + z_torques[2] - z_torques[3]
    x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (ARM_LEN / jnp.sqrt(2))
    y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * (ARM_LEN / jnp.sqrt(2))
    torques = jnp.array([x_torque, y_torque, z_torque])
    torques = torques - jnp.cross(rpy_rates, jnp.dot(J, rpy_rates))
    rpy_rates_deriv = jnp.dot(J_INV, torques)
    acc = force_world_frame / mass
    # Update state.
    next_pos = pos + vel * dt
    next_vel = vel + acc * dt
    rpy_rates = rpy_rates + rpy_rates_deriv * dt
    next_quat = R.from_euler("xyz", R.from_quat(quat).as_euler("xyz") + ang_vel * dt).as_quat()
    next_ang_vel = rot.apply(rpy_rates)
    return next_pos, next_quat, next_vel, next_ang_vel
