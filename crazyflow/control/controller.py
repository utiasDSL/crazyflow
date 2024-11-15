"""Functional programming implementation of the onboard controller.

We reimplement the onboard controller for two reasons:
- We cannot use the C++ bindings of the firmware to differentiate through the onboard controller.
- We need to implement it with JAX to enable efficient, batched computations.

Since our controller is a PID controller, it requires integration of the error over time. We opt for
a functional implementation to avoid storing any state in the class. Doing so would either prevent
us from easily scaling across batches and drones with JAX's `vmap`, or require us to support batches
and multiple drones explicitly in the controller.
"""

from enum import Enum

import jax.numpy as jnp
from jax import Array
from jax.scipy.spatial.transform import Rotation as R


class Control(str, Enum):
    """Control type of the simulated onboard controller."""

    state = "state"
    attitude = "attitude"
    thrust = "thrust"
    default = attitude


class Controller(Enum):
    """Controller type of the simulated onboard controller."""

    pycffirmware = "pycffirmware"
    emulatefirmware = "emulatefirmware"
    default = emulatefirmware


GRAVITY = 9.81
KF: float = 3.16e-10
KM: float = 7.94e-12
ARM_LEN: float = 0.46
MASS: float = 0.027
P_F: Array = jnp.array([0.4, 0.4, 1.25])
I_F: Array = jnp.array([0.05, 0.05, 0.05])
D_F: Array = jnp.array([0.2, 0.2, 0.5])
P_T: Array = jnp.array([70000.0, 70000.0, 60000.0])
I_T: Array = jnp.array([0.0, 0.0, 500.0])
D_T: Array = jnp.array([20000.0, 20000.0, 12000.0])
J: Array = jnp.array([[2.3951e-5, 0, 0], [0, 2.3951e-5, 0], [0, 0, 3.2347e-5]])
J_INV: Array = jnp.linalg.inv(J)
PWM2RPM_SCALE: float = 0.2685
PWM2RPM_CONST: float = 4070.3
MIN_PWM: float = 20000
MAX_PWM: float = 65535
MIN_RPM: float = PWM2RPM_SCALE * MIN_PWM + PWM2RPM_CONST
MAX_RPM: float = PWM2RPM_SCALE * MAX_PWM + PWM2RPM_CONST
MIN_THRUST: float = KF * MIN_RPM**2
MAX_THRUST: float = KF * MAX_RPM**2
MIX_MATRIX: Array = jnp.array([[0.5, -0.5, -1], [0.5, 0.5, 1], [-0.5, 0.5, -1], [-0.5, -0.5, 1]])


def state2attitude(
    pos: Array,
    vel: Array,
    quat: Array,
    des_pos: Array,
    des_vel: Array,
    des_quat: Array,
    pos_err_i: Array,
    t: float,
) -> tuple[Array, Array, Array]:
    """Compute the desired collective thrust and attitude for a reference trajectory."""
    rot_now = R.from_quat(quat)
    pos_err = des_pos - pos
    vel_err = des_vel - vel
    pos_err_i = jnp.clip(pos_err_i + pos_err * t, -2.0, 2.0)
    pos_err_i = jnp.clip(pos_err_i[2], -0.15, 0.15)
    # PID target thrust.
    des_thrust = P_F * pos_err + I_F * pos_err_i + D_F * vel_err + jnp.array([0, 0, GRAVITY])
    collective_thrust = jnp.maximum(0.0, jnp.dot(des_thrust, rot_now.as_matrix()[:, 2]))
    # PID attitude
    rpy = R.from_quat(des_quat).as_euler("xyz")
    z_axis = des_thrust / jnp.linalg.norm(des_thrust)
    helper_axis = jnp.array([jnp.cos(rpy[2]), jnp.sin(rpy[2]), 0])
    y_axis = (y_axis := jnp.cross(z_axis, helper_axis)) / jnp.linalg.norm(y_axis)
    x_axis = jnp.cross(y_axis, z_axis)
    attitude = R.from_matrix(jnp.stack([x_axis, y_axis, z_axis])).as_euler("xyz")
    return collective_thrust, attitude, pos_err_i


def attitude2rpm(
    cmd: Array, quat: Array, ang_vel: Array, rpy_err_i: Array, dt: float
) -> tuple[Array, Array]:
    """Convert the desired attitude and quaternion into motor RPMs."""
    rot = R.from_quat(quat)
    target_rot = R.from_euler("xyz", cmd[..., 1:], degrees=False)  # Or XYZ ?
    cur_rpy = rot.as_euler("xyz")
    rot_matrix_e = jnp.dot((target_rot.as_matrix().transpose()), rot.as_matrix()) - jnp.dot(
        rot.as_matrix().transpose(), target_rot.as_matrix()
    )
    rot_e = jnp.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
    # We assume zero rpy rates target. Otherwise: rpy_rates_target -(cur_rpy - last_rpy) / dt
    rpy_rates_e = -rot.apply(ang_vel, inverse=True)  # Now in body frame
    rpy_err_i = rpy_err_i - rot_e * dt
    rpy_err_i = jnp.clip(rpy_err_i, -1500.0, 1500.0)
    rpy_err_i = rpy_err_i.at[:2].set(jnp.clip(rpy_err_i[:2], -1.0, 1.0))
    # PID target torques.
    target_torques = -P_T * rot_e + D_T * rpy_rates_e + I_T * rpy_err_i
    target_torques = jnp.clip(target_torques, -3200, 3200)
    pwm = cmd[..., 0] + jnp.dot(MIX_MATRIX, target_torques)
    pwm = jnp.clip(pwm, MIN_PWM, MAX_PWM)
    return PWM2RPM_CONST + PWM2RPM_SCALE * pwm, rpy_err_i


def thrust2rpm(thrust: Array) -> Array:
    """Convert the desired thrust into motor RPMs.

    Args:
        thrust: The desired thrust per motor. Shape (..., 4).

    Returns:
        The motors' RPMs to apply to the quadrotor.
    """
    assert thrust.shape[-1] == 4, "Thrust must have 4 motors in the last dimension"
    thrust = jnp.clip(thrust, min=MIN_THRUST, max=MAX_THRUST)
    pwms = jnp.clip((jnp.sqrt(thrust / KF) - PWM2RPM_CONST) / PWM2RPM_SCALE, MIN_PWM, MAX_PWM)
    return PWM2RPM_CONST + PWM2RPM_SCALE * pwms
