from enum import Enum
from functools import partial
from typing import Callable

import jax
from jax import Array
from jax.numpy import vectorize
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.sim.structs import SimData


class Integrator(str, Enum):
    euler = "euler"
    rk4 = "rk4"  # TODO: Confirm the RK4 implementation correctness
    default = euler  # TODO: Replace with rk4


def euler(data: SimData, deriv_fn: Callable[[SimData], SimData]) -> SimData:
    return integrate(data, deriv_fn(data), dt=1 / data.core.freq)


def rk4(data: SimData, deriv_fn: Callable[[SimData], SimData]) -> SimData:
    dt = 1 / data.core.freq
    data_d1 = deriv_fn(data)
    data_d2 = deriv_fn(integrate(data, data_d1, dt=dt / 2))
    data_d3 = deriv_fn(integrate(data, data_d2, dt=dt / 2))
    data_d4 = deriv_fn(integrate(data, data_d3, dt=dt))
    return integrate(data, rk4_average(data_d1, data_d2, data_d3, data_d4), dt=dt)


def rk4_average(k1: SimData, k2: SimData, k3: SimData, k4: SimData) -> SimData:
    """Average four derivatives according to the RK4 rules."""
    data = k1
    k1, k2, k3, k4 = k1.states_deriv, k2.states_deriv, k3.states_deriv, k4.states_deriv
    states_deriv = jax.tree.map(
        lambda x1, x2, x3, x4: (x1 + 2 * x2 + 2 * x3 + x4) / 6, k1, k2, k3, k4
    )
    return data.replace(states_deriv=states_deriv)


def integrate(data: SimData, deriv: SimData, dt: float) -> SimData:
    """Integrate the dynamics forward in time."""
    states, states_deriv = data.states, deriv.states_deriv
    pos, quat, vel, rpy_rates = states.pos, states.quat, states.vel, states.rpy_rates
    dpos, drot = states_deriv.dpos, states_deriv.drot
    dvel, drpy_rates = states_deriv.dvel, states_deriv.drpy_rates
    next_pos, next_quat, next_vel, next_rpy_rates = _integrate(
        pos, quat, vel, rpy_rates, dpos, drot, dvel, drpy_rates, dt
    )
    return data.replace(
        states=states.replace(pos=next_pos, quat=next_quat, vel=next_vel, rpy_rates=next_rpy_rates)
    )


@partial(vectorize, signature="(3),(4),(3),(3),(3),(3),(3),(3)->(3),(4),(3),(3)", excluded=[8])
def _integrate(
    pos: Array,
    quat: Array,
    vel: Array,
    rpy_rates: Array,
    dpos: Array,
    drot: Array,
    dvel: Array,
    drpy_rates: Array,
    dt: float,
) -> SimData:
    """Integrate the dynamics forward in time.

    Args:
        data: The simulation data structure.
        vel: The velocity of the drone.
        rpy_rates: The roll, pitch, and yaw rates of the drone.
        acc: The acceleration of the drone.
        rpy_rates_deriv: The derivative of the roll, pitch, and yaw rates of the drone.
        dt: The time step to integrate over.
    """
    rot = R.from_quat(quat)
    rpy_rates_local = rot.apply(rpy_rates, inverse=True)
    drot_local = rot.apply(drot, inverse=True)  # Also rpy rates, but from the derivative data
    drpy_rates_local = rot.apply(drpy_rates, inverse=True)
    next_pos = pos + dpos * dt
    next_rot = R.from_euler("xyz", rot.as_euler("xyz") + drot_local * dt)
    next_quat = next_rot.as_quat()
    next_vel = vel + dvel * dt
    next_rpy_rates_local = rpy_rates_local + drpy_rates_local * dt
    next_rpy_rates = next_rot.apply(next_rpy_rates_local)
    return next_pos, next_quat, next_vel, next_rpy_rates
