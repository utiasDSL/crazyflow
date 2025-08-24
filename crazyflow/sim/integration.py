from enum import Enum
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.numpy import vectorize
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.sim.structs import SimData


class Integrator(str, Enum):
    euler = "euler"
    rk4 = "rk4"
    symplectic_euler = "symplectic_euler"
    default = euler


def euler(data: SimData, deriv_fn: Callable[[SimData], SimData]) -> SimData:
    """Explicit Euler integration.

    Args:
        data: The simulation data structure.
        deriv_fn: The function to compute the derivative of the dynamics.

    Returns:
        The integrated simulation data structure.
    """
    return integrate(data, deriv_fn(data), dt=1 / data.core.freq)


def rk4(data: SimData, deriv_fn: Callable[[SimData], SimData]) -> SimData:
    """Runge-Kutta 4 integration.

    Args:
        data: The simulation data structure.
        deriv_fn: The function to compute the derivative of the dynamics.

    Returns:
        The integrated simulation data structure.
    """
    dt = 1 / data.core.freq
    data_d1 = deriv_fn(data)
    data_d2 = deriv_fn(integrate(data, data_d1, dt=dt / 2))
    data_d3 = deriv_fn(integrate(data, data_d2, dt=dt / 2))
    data_d4 = deriv_fn(integrate(data, data_d3, dt=dt))
    return integrate(data, rk4_average(data_d1, data_d2, data_d3, data_d4), dt=dt)


def symplectic_euler(data: SimData, deriv_fn: Callable[[SimData], SimData]) -> SimData:
    """Symplectic Euler integration.

    Args:
        data: The simulation data structure.
        deriv_fn: The function to compute the derivative of the dynamics.
    """
    return integrate_symplectic(data, deriv_fn(data), dt=1 / data.core.freq)


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

    pos, quat, vel, ang_vel = states.pos, states.quat, states.vel, states.ang_vel
    rotor_vel = states.rotor_vel
    dpos, drot = states_deriv.vel, states_deriv.ang_vel
    dvel, dang_vel, drotor_vel = states_deriv.acc, states_deriv.ang_acc, states_deriv.rotor_acc

    next_pos, next_quat, next_vel, next_ang_vel, next_rotor_vel = _integrate(
        pos, quat, vel, ang_vel, rotor_vel, dpos, drot, dvel, dang_vel, drotor_vel, dt
    )
    states = states.replace(
        pos=next_pos, quat=next_quat, vel=next_vel, ang_vel=next_ang_vel, rotor_vel=next_rotor_vel
    )
    return data.replace(states=states)


def integrate_symplectic(data: SimData, deriv: SimData, dt: float) -> SimData:
    """Integrate the dynamics forward in time."""
    states, states_deriv = data.states, deriv.states_deriv

    pos, quat, vel, ang_vel = states.pos, states.quat, states.vel, states.ang_vel
    rotor_vel = states.rotor_vel
    dvel, dang_vel, drotor_vel = states_deriv.vel, states_deriv.ang_vel, states_deriv.rotor_acc

    next_pos, next_quat, next_vel, next_ang_vel, next_rotor_vel = _integrate_symplectic(
        pos, quat, vel, ang_vel, rotor_vel, dvel, dang_vel, drotor_vel, dt
    )
    states = states.replace(
        pos=next_pos, quat=next_quat, vel=next_vel, ang_vel=next_ang_vel, rotor_vel=next_rotor_vel
    )
    return data.replace(states=states)


@partial(
    vectorize,
    signature="(3),(4),(3),(3),(M),(3),(3),(3),(3),(M)->(3),(4),(3),(3),(M)",
    excluded=[10],
)
def _integrate(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    rotor_vel: Array,
    dpos: Array,
    drot: Array,
    dvel: Array,
    dang_vel: Array,
    drotor_vel: Array,
    dt: float,
) -> tuple[Array, Array, Array, Array]:
    """Integrate the dynamics forward in time.

    Args:
        pos: The position of the drone.
        quat: The orientation of the drone as a quaternion.
        vel: The velocity of the drone.
        ang_vel: The angular velocity of the drone.
        rotor_vel: The rotor velocity of the drone.
        dpos: The derivative of the position of the drone.
        drot: The derivative of the quaternion of the drone (3D angular velocity).
        dvel: The derivative of the velocity of the drone.
        dang_vel: The derivative of the angular velocity of the drone.
        drotor_vel: The derivative of the rotor velocity of the drone.
        dt: The time step to integrate over.

    Returns:
        The next position, quaternion, velocity, and roll, pitch, and yaw rates of the drone.
    """
    next_pos = pos + dpos * dt
    # Prevent NaN gradients by setting extremely small rotations to 0. This should not be necessary
    # because of the small-angle approximation in from_rotvec, but jax still gives us NaN gradients
    # otherwise.
    drot = jnp.where(jnp.abs(drot) < jnp.finfo(pos.dtype).smallest_normal, 0, drot)
    next_quat = (R.from_quat(quat) * R.from_rotvec(drot * dt)).as_quat()
    next_vel = vel + dvel * dt
    next_ang_vel = ang_vel + dang_vel * dt
    next_rotor_vel = rotor_vel + drotor_vel * dt
    return next_pos, next_quat, next_vel, next_ang_vel, next_rotor_vel


@partial(vectorize, signature="(3),(4),(3),(3),(M),(3),(3),(M)->(3),(4),(3),(3),(M)", excluded=[7])
def _integrate_symplectic(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    rotor_vel: Array,
    dvel: Array,
    dang_vel: Array,
    drotor_vel: Array,
    dt: float,
) -> tuple[Array, Array, Array, Array]:
    """Integrate the dynamics forward in time using symplectic integration.

    See e.g. https://adamsturge.github.io/Engine-Blog/mydoc_symplectic_euler.html for an explanation
    of the differences between explicit and symplectic integration.

    Args:
        pos: The position of the drone.
        quat: The orientation of the drone as a quaternion.
        vel: The velocity of the drone.
        ang_vel: The angular velocity of the drone.
        rotor_vel: The rotor velocity of the drone.
        dpos: The derivative of the position of the drone.
        drot: The derivative of the quaternion of the drone (3D angular velocity).
        dvel: The derivative of the velocity of the drone.
        dang_vel: The derivative of the angular velocity of the drone.
        drotor_vel: The derivative of the rotor velocity of the drone.
        dt: The time step to integrate over.

    Returns:
        The next position, quaternion, velocity, and roll, pitch, and yaw rates of the drone.
    """
    next_vel = vel + dvel * dt
    next_ang_vel = ang_vel + dang_vel * dt
    next_rotor_vel = rotor_vel + drotor_vel * dt
    next_pos = pos + next_vel * dt
    # See comment in _integrate.
    next_ang_vel = jnp.where(
        jnp.abs(next_ang_vel) < jnp.finfo(pos.dtype).smallest_normal, 0, next_ang_vel
    )
    next_quat = (R.from_quat(quat) * R.from_rotvec(next_ang_vel * dt)).as_quat()
    return next_pos, next_quat, next_vel, next_ang_vel, next_rotor_vel
