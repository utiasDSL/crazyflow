"""Functional programming implementation of the onboard controller.

We reimplement the onboard controller for two reasons:
- We cannot use the C++ bindings of the firmware to differentiate through the onboard controller.
- We need to implement it with JAX to enable efficient, batched computations.

Since our controller is a PID controller, it requires integration of the error over time. We opt for
a functional implementation to avoid storing any state in the class. Doing so would either prevent
us from easily scaling across batches and drones with JAX's `vmap`, or require us to support batches
and multiple drones explicitly in the controller.
"""

import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as R
from jax.typing import Array

GRAVITY = 9.81
KF: float = 3.16e-10
KM: float = 7.94e-12
P_F: Array = jnp.array([0.4, 0.4, 1.25])
I_F: Array = jnp.array([0.05, 0.05, 0.05])
D_F: Array = jnp.array([0.2, 0.2, 0.5])
P_T: Array = jnp.array([70000.0, 70000.0, 60000.0])
I_T: Array = jnp.array([0.0, 0.0, 500.0])
D_T: Array = jnp.array([20000.0, 20000.0, 12000.0])
PWM2RPM_SCALE: float = 0.2685
PWM2RPM_CONST: float = 4070.3
MIN_PWM: float = 20000
MAX_PWM: float = 65535
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


def attitude2torques(thrust: Array, attitude: Array, quat: Array) -> Array:
    """Compute the torques given a desired collective thrust and attitude of the drone."""
    rot_now = R.from_quat(quat)
    des_rot = R.from_euler("xyz", attitude)
    rot_err = rot_now.inv() * des_rot
    raise NotImplementedError
