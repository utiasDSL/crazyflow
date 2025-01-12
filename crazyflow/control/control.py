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
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.constants import GRAVITY, MASS, MIX_MATRIX


class Control(str, Enum):
    """Control type of the simulated onboard controller."""

    state = "state"
    """State control takes [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate].
    
    Note:
        Recommended frequency is >=20 Hz.

    Warning:
        Currently, we only use positions, velocities, and yaw. The rest of the state is ignored.
        This is subject to change in the future.
    """
    attitude = "attitude"
    """Attitude control takes [collective thrust, roll, pitch, yaw].

    Note:
        Recommended frequency is >=100 Hz.
    """
    thrust = "thrust"
    """Thrust control takes [thrust1, thrust2, thrust3, thrust4] for each drone motor.

    Note:
        Recommended frequency is >=500 Hz.
    """
    default = attitude


KF: float = 3.16e-10
KM: float = 7.94e-12
P_F: Array = np.array([0.4, 0.4, 1.25])
I_F: Array = np.array([0.05, 0.05, 0.05])
D_F: Array = np.array([0.2, 0.2, 0.5])
I_F_RANGE: Array = np.array([2.0, 2.0, 0.4])
P_T: Array = np.array([70000.0, 70000.0, 60000.0])
I_T: Array = np.array([0.0, 0.0, 500.0])
D_T: Array = np.array([20000.0, 20000.0, 12000.0])
PWM2RPM_SCALE: float = 0.2685
PWM2RPM_CONST: float = 4070.3
MIN_PWM: float = 20000
MAX_PWM: float = 65535
MIN_RPM: float = PWM2RPM_SCALE * MIN_PWM + PWM2RPM_CONST
MAX_RPM: float = PWM2RPM_SCALE * MAX_PWM + PWM2RPM_CONST
MIN_THRUST: float = KF * MIN_RPM**2
MAX_THRUST: float = KF * MAX_RPM**2
# Thrust curve parameters for brushed motors
THRUST_CURVE_A: float = -1.1264
THRUST_CURVE_B: float = 2.2541
THRUST_CURVE_C: float = 0.0209


@partial(jnp.vectorize, signature="(3),(3),(4),(3),(3),(1),(3)->(4),(3)", excluded=[7])
def state2attitude(
    pos: Array,
    vel: Array,
    quat: Array,
    des_pos: Array,
    des_vel: Array,
    des_yaw: Array,
    i_error: Array,
    dt: float,
) -> tuple[Array, Array]:
    """Compute the next desired collective thrust and roll/pitch/yaw of the drone."""
    pos_error, vel_error = des_pos - pos, des_vel - vel
    # Update integral error
    i_error = jnp.clip(i_error + pos_error * dt, -I_F_RANGE, I_F_RANGE)
    # Compute target thrust
    thrust = P_F * pos_error + I_F * i_error + D_F * vel_error
    thrust = thrust.at[2].add(MASS * GRAVITY)
    # Update z_axis to the current orientation of the drone
    z_axis = R.from_quat(quat).as_matrix()[:, 2]
    # Project the thrust onto the z-axis
    thrust_desired = jnp.clip(thrust @ z_axis, 0.3 * MASS * GRAVITY, 1.8 * MASS * GRAVITY)
    # Update the desired z-axis
    z_axis = thrust / jnp.linalg.norm(thrust)
    yaw_axis = jnp.concatenate([jnp.cos(des_yaw), jnp.sin(des_yaw), jnp.array([0.0])])
    y_axis = jnp.cross(z_axis, yaw_axis)
    y_axis = y_axis / jnp.linalg.norm(y_axis)
    x_axis = jnp.cross(y_axis, z_axis)
    euler_desired = R.from_matrix(jnp.vstack([x_axis, y_axis, z_axis]).T).as_euler("xyz")
    return jnp.concatenate([jnp.atleast_1d(thrust_desired), euler_desired]), i_error


@partial(jnp.vectorize, signature="(4),(4),(3),(3)->(4),(3)", excluded=[4])
def attitude2rpm(
    controls: Array, quat: Array, last_rpy: Array, rpy_err_i: Array, dt: float
) -> tuple[Array, Array]:
    """Convert the desired collective thrust and attitude into motor RPMs."""
    rot = R.from_quat(quat)
    target_rot = R.from_euler("xyz", controls[1:])
    drot = (target_rot.inv() * rot).as_matrix()
    # Extract the anti-symmetric part of the relative rotation matrix.
    rot_e = jnp.array([drot[2, 1] - drot[1, 2], drot[0, 2] - drot[2, 0], drot[1, 0] - drot[0, 1]])
    # TODO: Assumes zero rpy_rates targets for now, use the actual target instead.
    rpy_rates_e = -(rot.as_euler("xyz") - last_rpy) / dt
    rpy_err_i = rpy_err_i - rot_e * dt
    rpy_err_i = jnp.clip(rpy_err_i, -1500.0, 1500.0)
    rpy_err_i = rpy_err_i.at[:2].set(jnp.clip(rpy_err_i[:2], -1.0, 1.0))
    # PID target torques.
    target_torques = -P_T * rot_e + D_T * rpy_rates_e + I_T * rpy_err_i
    target_torques = jnp.clip(target_torques, -3200, 3200)
    thrust_per_motor = jnp.atleast_1d(controls[0]) / 4
    pwm = jnp.clip(thrust2pwm(thrust_per_motor) + MIX_MATRIX @ target_torques, MIN_PWM, MAX_PWM)
    return pwm2rpm(pwm), rpy_err_i


@partial(jnp.vectorize, signature="(4)->(4)")
def thrust2pwm(thrust: Array) -> Array:
    """Convert the desired thrust into motor PWM.

    Args:
        thrust: The desired thrust per motor.

    Returns:
        The motors' PWMs to apply to the quadrotor.
    """
    thrust = jnp.clip(thrust, MIN_THRUST, MAX_THRUST)  # Protect against NaN values
    return jnp.clip((jnp.sqrt(thrust / KF) - PWM2RPM_CONST) / PWM2RPM_SCALE, MIN_PWM, MAX_PWM)


@partial(jnp.vectorize, signature="(4)->(4)")
def pwm2rpm(pwm: Array) -> Array:
    """Convert the motors' PWMs into RPMs."""
    return PWM2RPM_CONST + PWM2RPM_SCALE * pwm


@jax.jit
def thrust_curve(thrust: Array) -> Array:
    """Compute the quadratic thrust curve of the crazyflie.

    Warning:
        This function is not used by the simulation. It is only used as interface to the firmware.

    Todo:
        Find out where this function is used in the firmware and emulate its use in our onboard
        controller reimplementation.

    Args:
        thrust: The desired motor thrust.

    Returns:
        The motors' PWMs to apply to the quadrotor.
    """
    tau = THRUST_CURVE_A * thrust**2 + THRUST_CURVE_B * thrust + THRUST_CURVE_C
    return jnp.clip(tau * MAX_PWM, MIN_PWM, MAX_PWM)
