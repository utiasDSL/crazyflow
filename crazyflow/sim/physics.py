from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.numpy import vectorize
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.constants import ARM_LEN, GRAVITY, SIGN_MIX_MATRIX
from crazyflow.control.control import KF, KM

SYS_ID_PARAMS = {
    "acc": np.array([20.907574256269616, 3.653687545690674]),
    "roll_acc": np.array([-130.3, -16.33, 119.3]),
    "pitch_acc": np.array([-99.94, -13.3, 84.73]),
    "yaw_acc": np.array([0.0, 0.0, 0.0]),
}


class Physics(str, Enum):
    """Physics mode for the simulation."""

    mujoco = "mujoco"
    analytical = "analytical"
    sys_id = "sys_id"
    default = analytical


@partial(vectorize, signature="(4),(4),(3),(1),(3,3)->(3),(3)", excluded=[5])
def surrogate_identified_collective_wrench(
    controls: Array, quat: Array, ang_vel: Array, mass: Array, J: Array, dt: float
) -> tuple[Array, Array]:
    """Surrogate collective wrench for the identified dynamics model.

    Contrary to the other physics implementations, this function is not based on a physical model.
    Instead, we use a predefined model structure, fit its parameters to data collected on a real
    drone, and predict the next state based on the control inputs and the current state. Since we do
    not have a physical model, we cannot compute the actual forces and torques required by the
    simulation pipeline. Instead, we return surrogate forces and torques that result in the desired
    acceleration and rpy rates derivative after converting back to the state derivative.

    Args:
        controls: The 4D control input consisting of the desired collective thrust and attitude.
        quat: The current orientation.
        ang_vel: The current angular velocity.
        mass: The drone's mass.
        J: The drone's inertia matrix.
        dt: The simulation time step.
    """
    collective_thrust, attitude = controls[0], controls[1:]
    rot = R.from_quat(quat)
    thrust = rot.apply(jnp.array([0, 0, collective_thrust]))
    drift = rot.apply(jnp.array([0, 0, 1]))
    k1, k2 = SYS_ID_PARAMS["acc"]
    acc = thrust * k1 + drift * k2
    rpy = rot.as_euler("xyz")
    # TODO: Move to an angular velocity based model to avoid this conversion.
    rpy_rates = ang_vel2rpy_rates(ang_vel, quat)
    k1, k2, k3 = SYS_ID_PARAMS["roll_acc"]
    roll_rate_deriv = k1 * rpy[0] + k2 * rpy_rates[0] + k3 * attitude[0]
    k1, k2, k3 = SYS_ID_PARAMS["pitch_acc"]
    pitch_rate_deriv = k1 * rpy[1] + k2 * rpy_rates[1] + k3 * attitude[1]
    k1, k2, k3 = SYS_ID_PARAMS["yaw_acc"]
    yaw_rate_deriv = k1 * rpy[2] + k2 * rpy_rates[2] + k3 * attitude[2]
    rpy_rates_deriv = jnp.array([roll_rate_deriv, pitch_rate_deriv, yaw_rate_deriv])
    # The identified dynamics model does not use forces or torques, because we assume no knowledge
    # of the drone's mass and inertia. However, to remain compatible with the physics pipeline, we
    # return surrogate forces and torques that result in the desired linear and angular velocity
    # derivative. When converting back to the state derivative, the mass and inertia will cancel
    # out, resulting in the correct linear and angular acceleration regardless of the model's mass
    # and inertia.
    # Rotate surrogate torques into global frame
    ang_vel_deriv = rpy_rates_deriv2ang_vel_deriv(rpy_rates_deriv, rpy_rates, quat)
    surrogate_torques = rot.apply(J @ ang_vel_deriv)
    surrogate_forces = acc * mass
    return surrogate_forces, surrogate_torques


@partial(vectorize, signature="(3),(1)->(3)")
def collective_force2acceleration(force: Array, mass: Array) -> Array:
    """Convert forces to acceleration."""
    return force / mass - jnp.array([0, 0, GRAVITY])


@partial(vectorize, signature="(3),(4),(3,3)->(3)")
def collective_torque2ang_vel_deriv(torque: Array, quat: Array, J_INV: Array) -> Array:
    """Convert torques to ang_vel_deriv."""
    return J_INV @ R.from_quat(quat).apply(torque, inverse=True)


@partial(vectorize, signature="(4),(4),(3),(3,3)->(3),(3)")
def rpms2collective_wrench(
    rpms: Array, quat: Array, ang_vel: Array, J: Array
) -> tuple[Array, Array]:
    """Convert RPMs to forces and torques in the global frame."""
    rot = R.from_quat(quat)
    motor_forces = rpms2motor_forces(rpms)
    body_force = jnp.array([0, 0, jnp.sum(motor_forces)])
    body_torque = rpms2body_torque(rpms, ang_vel, motor_forces, J)
    return rot.apply(body_force), rot.apply(body_torque)


@partial(vectorize, signature="(4)->(4)")
def rpms2motor_forces(rpms: Array) -> Array:
    """Convert RPMs to motor forces (body frame, along the z-axis)."""
    return rpms**2 * KF


@partial(vectorize, signature="(4)->(4)")
def rpms2motor_torques(rpms: Array) -> Array:
    """Convert RPMs to motor torques (body frame, around the z-axis)."""
    return rpms**2 * KM


@partial(vectorize, signature="(4),(3),(4),(3,3)->(3)")
def rpms2body_torque(rpms: Array, ang_vel: Array, motor_forces: Array, J: Array) -> Array:
    """Convert RPMs to torques in the body frame."""
    motor_torques = rpms2motor_torques(rpms)
    z_torque = SIGN_MIX_MATRIX[..., 2] @ motor_torques
    x_torque = SIGN_MIX_MATRIX[..., 0] @ motor_forces * (ARM_LEN / jnp.sqrt(2))
    y_torque = SIGN_MIX_MATRIX[..., 1] @ motor_forces * (ARM_LEN / jnp.sqrt(2))
    return jnp.array([x_torque, y_torque, z_torque]) - jnp.cross(ang_vel, J @ ang_vel)


@jax.jit
@partial(vectorize, signature="(3),(4)->(3)")
def ang_vel2rpy_rates(ang_vel: Array, quat: Array) -> Array:
    """Convert angular velocity to rpy rates.

    Args:
        ang_vel: The angular velocity in the body frame.
        quat: The current orientation.

    Returns:
        The rpy rates in the body frame, following the 'xyz' convention.
    """
    rpy = R.from_quat(quat).as_euler("xyz")
    sin_phi, cos_phi = jnp.sin(rpy[0]), jnp.cos(rpy[0])
    cos_theta, tan_theta = jnp.cos(rpy[1]), jnp.tan(rpy[1])
    conv_mat = jnp.array(
        [
            [1, sin_phi * tan_theta, cos_phi * tan_theta],
            [0, cos_phi, -sin_phi],
            [0, sin_phi / cos_theta, cos_phi / cos_theta],
        ]
    )
    return conv_mat @ ang_vel


@jax.jit
@partial(vectorize, signature="(3),(4)->(3)")
def rpy_rates2ang_vel(rpy_rates: Array, quat: Array) -> Array:
    """Convert rpy rates to angular velocity."""
    return rpy_rates2ang_vel_matrix(quat) @ rpy_rates


@jax.jit
@partial(vectorize, signature="(3),(3),(4)->(3)")
def rpy_rates_deriv2ang_vel_deriv(rpy_rates_deriv: Array, rpy_rates: Array, quat: Array) -> Array:
    r"""Convert rpy rates derivatives to angular velocity derivatives.

    .. math::
        \dot{\omega} = \mathbf{\dot{W}}\dot{\mathbf{\psi}} + \mathbf{W} \ddot{\mathbf{\psi}}
    """
    rpy = R.from_quat(quat).as_euler("xyz")
    # W_dot
    phi, theta = rpy[0], rpy[1]
    phi_dot, theta_dot = rpy_rates[0], rpy_rates[1]
    sin_phi, cos_phi = jnp.sin(phi), jnp.cos(phi)
    sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
    # fmt: off
    conv_mat_dot = jnp.array(
        [[0,                  0,                                           -cos_theta * theta_dot],
         [0, -sin_phi * phi_dot,  cos_phi * phi_dot * cos_theta - sin_phi * sin_theta * theta_dot],
         [0, -cos_phi * phi_dot, -sin_phi * phi_dot * cos_theta - cos_phi * sin_theta * theta_dot]]
    )
    # fmt: on
    return conv_mat_dot @ rpy_rates + rpy_rates2ang_vel_matrix(quat) @ rpy_rates_deriv


def ang_vel2rpy_rates_matrix(quat: Array) -> Array:
    """Calculate the conversion matrix from angular velocities to rpy rates."""
    rpy = R.from_quat(quat).as_euler("xyz")
    sin_phi, cos_phi = jnp.sin(rpy[0]), jnp.cos(rpy[0])
    cos_theta, tan_theta = jnp.cos(rpy[1]), jnp.tan(rpy[1])
    # fmt: off
    conv_mat = jnp.array(
        [[1, sin_phi * tan_theta, cos_phi * tan_theta],
         [0,             cos_phi,            -sin_phi],
         [0, sin_phi / cos_theta, cos_phi / cos_theta]]
    )
    # fmt: on
    return conv_mat


def rpy_rates2ang_vel_matrix(quat: Array) -> Array:
    """Calculate the conversion matrix from rpy rates to angular velocities."""
    rpy = R.from_quat(quat).as_euler("xyz")
    sin_phi, cos_phi = jnp.sin(rpy[0]), jnp.cos(rpy[0])
    cos_theta, sin_theta = jnp.cos(rpy[1]), jnp.sin(rpy[1])
    # fmt: off
    conv_mat = jnp.array(
        [[1,        0,          -sin_theta],
         [0,  cos_phi, sin_phi * cos_theta],
         [0, -sin_phi, cos_phi * cos_theta]]
    )
    # fmt: on
    return conv_mat
