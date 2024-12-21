from enum import Enum
from functools import partial

import jax.numpy as jnp
from jax import Array
from jax.numpy import vectorize
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.constants import ARM_LEN, GRAVITY, SIGN_MIX_MATRIX
from crazyflow.control.control import KF, KM

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

    # mujoco = "mujoco"  TODO: Implement
    analytical = "analytical"
    sys_id = "sys_id"
    default = analytical


@partial(vectorize, signature="(4),(4),(3),(1),(3,3)->(3),(3)", excluded=[5])
def virtual_identified_collective_wrench(
    controls: Array, quat: Array, rpy_rates: Array, mass: Array, J: Array, dt: float
) -> tuple[Array, Array]:
    """Virtual collective wrench for the identified dynamics model.

    Contrary to the other physics implementations, this function is not based on a physical model.
    Instead, we fit a linear model to the data collected on the real drone, and predict the next
    state based on the control inputs and the current state. Since we do not have a physical model,
    we cannot compute the actual forces and torques required by the simulation pipeline. Instead, we
    return virtual forces and torques that result in the desired acceleration and rpy rates
    derivative after converting back to the state derivative.

    Warning:
        The identified dynamics model does not include second-order derivatives of the orientation.
        Since the integration interface requires derivatives for all states, we leverage the
        rpy_rates_deriv that will integrate to the model's rpy_rates instead.

    Args:
        controls: The 4D control input consisting of the desired collective thrust and attitude.
        quat: The current orientation.
        rpy_rates: The current roll, pitch, and yaw rates.
        mass: The drone's mass.
        J: The drone's inertia matrix.
        dt: The simulation time step.
    """
    collective_thrust, attitude = controls[0], controls[1:]
    rot = R.from_quat(quat)
    thrust = rot.apply(jnp.array([0, 0, collective_thrust]))
    drift = rot.apply(jnp.array([0, 0, 1]))
    prev_rpy_rates = rot.apply(rpy_rates, inverse=True)
    a1, a2 = SYS_ID_PARAMS["acc_k1"], SYS_ID_PARAMS["acc_k2"]
    acc = thrust * a1 + drift * a2
    # rpy_rates_deriv have no real meaning in this context, since the identified dynamics set the
    # rpy_rates to the commanded values directly. However, since we use a unified integration
    # interface for all physics models, we cannot access states directly. Instead, we calculate
    # which rpy_rates_deriv would have resulted in the desired rpy_rates, and return that.
    roll_cmd, pitch_cmd, yaw_cmd = attitude
    rpy = rot.as_euler("xyz")
    roll_rate = SYS_ID_PARAMS["roll_alpha"] * rpy[0] + SYS_ID_PARAMS["roll_beta"] * roll_cmd
    pitch_rate = SYS_ID_PARAMS["pitch_alpha"] * rpy[1] + SYS_ID_PARAMS["pitch_beta"] * pitch_cmd
    yaw_rate = SYS_ID_PARAMS["yaw_alpha"] * rpy[2] + SYS_ID_PARAMS["yaw_beta"] * yaw_cmd
    rpy_rates_local = jnp.array([roll_rate, pitch_rate, yaw_rate])
    rpy_rates_local_deriv = (rpy_rates_local - prev_rpy_rates) / dt
    # The identified dynamics model does not use forces or torques, because we assume no knowledge
    # of the drone's mass and inertia. However, to remain compatible with the physics pipeline, we
    # return virtual forces and torques that result in the desired acceleration and rpy rates
    # derivative. When converting back to the state derivative, the mass and inertia will cancel
    # out, resulting in the correct acceleration and rpy rates derivative regardless of the model's
    # mass and inertia.
    virtual_torques = rot.apply(J @ rpy_rates_local_deriv)
    virtual_forces = acc * mass
    return virtual_forces, virtual_torques


@partial(vectorize, signature="(3),(1)->(3)")
def collective_force2acceleration(force: Array, mass: Array) -> Array:
    """Convert forces to acceleration."""
    return force / mass - jnp.array([0, 0, GRAVITY])


@partial(vectorize, signature="(3),(4),(3,3)->(3)")
def collective_torque2rpy_rates_deriv(torque: Array, quat: Array, J_INV: Array) -> Array:
    """Convert torques to rpy_rates_deriv."""
    return R.from_quat(quat).apply(J_INV @ torque)


@partial(vectorize, signature="(4),(4),(3),(3,3)->(3),(3)")
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
