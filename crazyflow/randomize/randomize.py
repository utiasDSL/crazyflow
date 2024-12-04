import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.sim.core import Sim
from crazyflow.sim.structs import SimParams


def randomize_mass(sim: Sim, mass: Array, mask: Array | None = None):
    """Randomize mass from a new masses.

    Args:
        sim: The simulation object.
        mass: The new masses. The shape always needs to be (n_worlds, n_drones).
        mask: Boolean array of shape (n_worlds, ) that indicates which worlds to reset. If None,
            all worlds are reset.

    Warning:
        This currently only works for analytical dynamics.
    """
    # TODO: Domain randomization when using MuJoCo as dynamics engine
    # TODO: recompile sim._sync_mjx? sim._mjx_model = _randomize_mass(sim._mjx_model, mass)
    # TODO: Parameters randomization for sys_id model
    sim.params = _randomize_mass_params(sim.params, mass, mask)


def randomize_inertia(sim: Sim, new_j: Array, mask: Array | None = None):
    """Randomize inertia tensor from a new inertia tensors.

    Args:
        sim: The simulation object.
        new_j: The new inertia tensors of shape (n_worlds, n_drones, 3, 3).
        mask: Boolean array of shape (n_worlds, ) that indicates which worlds to reset. If None,
            all worlds are reset.

    Warning:
        This currently only works for analytical dynamics.
    """
    assert new_j.shape[2:] == (3, 3), "Inertia tensor must be of shape (n_worlds, n_drones, 3, 3)"
    sim.params = _randomize_inertia_params(sim.params, new_j, mask)


@jax.jit
def _randomize_mass_params(params: SimParams, mass: Array, mask: Array | None = None) -> SimParams:
    mass = jnp.atleast_3d(mass)
    assert mass.shape[2] == 1, f"Expected shape (n_worlds, n_drones, 1), is {mass.shape}"
    mask = mask.squeeze() if mask is not None else jnp.ones(mass.shape[0], dtype=bool)
    return params.replace(mass=jnp.where(mask[:, None, None], mass, params.mass))


@jax.jit
def _randomize_inertia_params(
    params: SimParams, new_j: Array, mask: Array | None = None
) -> SimParams:
    mask = mask if mask is not None else jnp.ones(new_j.shape[0], dtype=bool)
    new_j_inv = jnp.linalg.inv(new_j)
    mask_4d = mask[:, None, None, None]
    return params.replace(
        J=jnp.where(mask_4d, new_j, params.J), J_INV=jnp.where(mask_4d, new_j_inv, params.J_INV)
    )
