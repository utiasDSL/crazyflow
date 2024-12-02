import jax.numpy as jnp
from flax.struct import dataclass
from jax import Array, Device


@dataclass
class SimState:
    pos: Array  # (N, M, 3)
    quat: Array  # (N, M, 4)
    vel: Array  # (N, M, 3)
    ang_vel: Array  # (N, M, 3)
    rpy_rates: Array  # (N, M, 3)


def default_state(n_worlds: int, n_drones: int, device: Device) -> SimState:
    """Create a default set of states for the simulation."""
    pos = jnp.zeros((n_worlds, n_drones, 3), device=device)
    quat = jnp.zeros((n_worlds, n_drones, 4), device=device)
    quat = quat.at[..., -1].set(1.0)
    rpy_rates = jnp.zeros((n_worlds, n_drones, 3), device=device)
    vel = jnp.zeros((n_worlds, n_drones, 3), device=device)
    ang_vel = jnp.zeros((n_worlds, n_drones, 3), device=device)
    return SimState(pos=pos, quat=quat, vel=vel, ang_vel=ang_vel, rpy_rates=rpy_rates)


@dataclass
class SimControls:
    state: Array  # (N, M, 13)
    staged_state: Array  # (N, M, 13)
    attitude: Array  # (N, M, 4)
    staged_attitude: Array  # (N, M, 4)
    thrust: Array  # (N, M, 4)
    rpms: Array  # (N, M, 4)
    rpy_err_i: Array  # (N, M, 3)
    pos_err_i: Array  # (N, M, 3)
    last_rpy: Array  # (N, M, 3)


def default_controls(n_worlds: int, n_drones: int, device: Device) -> SimControls:
    """Create a default set of controls for the simulation."""
    return SimControls(
        state=jnp.zeros((n_worlds, n_drones, 13), device=device),
        staged_state=jnp.zeros((n_worlds, n_drones, 13), device=device),
        attitude=jnp.zeros((n_worlds, n_drones, 4), device=device),
        staged_attitude=jnp.zeros((n_worlds, n_drones, 4), device=device),
        thrust=jnp.zeros((n_worlds, n_drones, 4), device=device),
        rpms=jnp.zeros((n_worlds, n_drones, 4), device=device),
        rpy_err_i=jnp.zeros((n_worlds, n_drones, 3), device=device),
        pos_err_i=jnp.zeros((n_worlds, n_drones, 3), device=device),
        last_rpy=jnp.zeros((n_worlds, n_drones, 3), device=device),
    )


@dataclass
class SimParams:
    mass: Array  # (N, M, 1)
    J: Array  # (N, M, 3, 3)
    J_INV: Array  # (N, M, 3, 3)


def default_params(
    n_worlds: int, n_drones: int, mass: float, J: Array, J_INV: Array, device: Device
) -> SimParams:
    """Create a default set of parameters for the simulation."""
    mass = jnp.ones((n_worlds, n_drones, 1), device=device) * mass
    J = jnp.tile(J[None, None, :, :], (n_worlds, n_drones, 1, 1))
    J_INV = jnp.tile(J_INV[None, None, :, :], (n_worlds, n_drones, 1, 1))
    return SimParams(mass=mass, J=J, J_INV=J_INV)


@dataclass
class SimData:
    states: SimState
    controls: SimControls
    params: SimParams
