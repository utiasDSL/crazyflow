from enum import Enum

import jax.numpy as jnp
from jax import Array
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.constants import ARM_LEN, GRAVITY, SIGN_MIX_MATRIX
from crazyflow.control.controller import KF, KM

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
    default = analytical


def identified_dynamics(
    cmd: Array, pos: Array, quat: Array, vel: Array, rpy_rates: Array, dt: float
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
        pos: The current position.
        quat: The current orientation.
        vel: The current velocity.
        rpy_rates: The current roll, pitch, and yaw rates.
        dt: The simulation time step.
    """
    collective_thrust, attitude = cmd[0], cmd[1:]
    rot = R.from_quat(quat)
    thrust = rot.apply(jnp.array([0, 0, collective_thrust]))
    drift = rot.apply(jnp.array([0, 0, 1]))
    a1, a2 = SYS_ID_PARAMS["acc_k1"], SYS_ID_PARAMS["acc_k2"]
    acc = thrust * a1 + drift * a2 - jnp.array([0, 0, GRAVITY])
    roll_cmd, pitch_cmd, yaw_cmd = attitude
    rpy = rot.as_euler("xyz")
    roll_rate = SYS_ID_PARAMS["roll_alpha"] * rpy[0] + SYS_ID_PARAMS["roll_beta"] * roll_cmd
    pitch_rate = SYS_ID_PARAMS["pitch_alpha"] * rpy[1] + SYS_ID_PARAMS["pitch_beta"] * pitch_cmd
    yaw_rate = SYS_ID_PARAMS["yaw_alpha"] * rpy[2] + SYS_ID_PARAMS["yaw_beta"] * yaw_cmd
    rpy_rates = jnp.array([roll_rate, pitch_rate, yaw_rate])
    next_pos = pos + vel * dt
    next_rot = R.from_euler("xyz", rpy + rpy_rates * dt)
    next_quat = next_rot.as_quat()
    next_vel = vel + acc * dt
    next_rpy_rates = next_rot.apply(rpy_rates)
    return next_pos, next_quat, next_vel, next_rpy_rates


def analytical_dynamics(
    forces: Array,
    torques: Array,
    pos: Array,
    quat: Array,
    vel: Array,
    rpy_rates: Array,
    mass: Array,
    J_INV: Array,
    dt: float,
) -> tuple[Array, Array, Array, Array]:
    """Analytical dynamics model."""
    # Convert rotational quantities to local frame
    rot = R.from_quat(quat)
    torques_local = rot.apply(torques, inverse=True)
    rpy_rates_local = rot.apply(rpy_rates, inverse=True)
    # Compute acceleration in global frame, rpy_rates in local frame
    acc = forces / mass - jnp.array([0, 0, GRAVITY])
    rpy_rates_deriv_local = J_INV @ torques_local
    # Update state.
    next_pos = pos + vel * dt
    next_vel = vel + acc * dt
    next_rot = R.from_euler("xyz", R.from_quat(quat).as_euler("xyz") + rpy_rates_local * dt)
    next_quat = next_rot.as_quat()
    # Convert rpy rates back to global frame
    next_rpy_rates_local = rpy_rates_local + rpy_rates_deriv_local * dt
    next_rpy_rates = next_rot.apply(next_rpy_rates_local)  # Always give rpy rates in world frame
    return next_pos, next_quat, next_vel, next_rpy_rates


def rpms2collective_wrench(
    rpms: Array, quat: Array, rpy_rates: Array, J: Array
) -> tuple[Array, Array]:
    """Convert RPMs to forces and torques in the global frame."""
    rot = R.from_quat(quat)
    # Forces
    forces = rpms**2 * KF
    force = jnp.array([0, 0, jnp.sum(forces)])
    # Torques
    rpy_rates = rot.apply(rpy_rates, inverse=True)  # Now in body frame
    z_torques = jnp.array(rpms**2) * KM
    z_torque = SIGN_MIX_MATRIX[..., 3] @ z_torques
    x_torque = SIGN_MIX_MATRIX[..., 0] @ forces * (ARM_LEN / jnp.sqrt(2))
    y_torque = SIGN_MIX_MATRIX[..., 1] @ forces * (ARM_LEN / jnp.sqrt(2))
    torques = jnp.array([x_torque, y_torque, z_torque])
    torques = torques - jnp.cross(rpy_rates, J @ rpy_rates)
    return rot.apply(force), rot.apply(torques)
