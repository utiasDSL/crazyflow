from enum import Enum
from typing import Callable

from jax import Array
from scipy.spatial.transform import Rotation as R

from crazyflow.sim.structs import SimControls


class Integrator(str, Enum):
    euler = "euler"
    # TODO: Implement rk4
    default = euler  # TODO: Replace with rk4


def euler(
    deriv_fn: Callable[[Array, Array, Array, Array, SimControls, float], tuple[Array, Array]],
    pos: Array,
    quat: Array,
    vel: Array,
    rpy_rates: Array,
    control: SimControls,
    dt: float,
) -> tuple[Array, Array, Array, Array]:
    acc, rpy_rates_deriv = deriv_fn(pos, quat, vel, rpy_rates, control, dt)
    return _apply(pos, quat, vel, rpy_rates, acc, rpy_rates_deriv, dt)


def rk4(
    deriv_fn: Callable[[Array, Array, Array, Array, SimControls, float], tuple[Array, Array]],
    pos: Array,
    quat: Array,
    vel: Array,
    rpy_rates: Array,
    control: SimControls,
    dt: float,
) -> tuple[Array, Array, Array, Array]:
    raise NotImplementedError("RK4 not implemented")


def _apply(
    pos: Array,
    quat: Array,
    vel: Array,
    rpy_rates: Array,
    acc: Array,
    rpy_rates_deriv: Array,
    dt: float,
) -> tuple[Array, Array, Array, Array]:
    rot = R.from_quat(quat)
    rpy_rates_local = rot.apply(rpy_rates, inverse=True)
    rpy_rates_deriv_local = rot.apply(rpy_rates_deriv, inverse=True)
    next_pos = pos + vel * dt
    next_vel = vel + acc * dt
    next_rot = R.from_euler("xyz", rot.as_euler("xyz") + rpy_rates_local * dt)
    next_rpy_rates_local = rpy_rates_local + rpy_rates_deriv_local * dt
    return next_pos, next_rot.as_quat(), next_vel, next_rot.apply(next_rpy_rates_local)
