import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.sim import Sim
from crazyflow.sim.structs import SimData
from crazyflow.utils import leaf_replace


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
    # TODO: Parameters randomization for sys_id model
    sim.data = _randomize_mass_params(sim.data, mass, mask)


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
    sim.data = _randomize_inertia_params(sim.data, new_j, mask)


@jax.jit
def _randomize_mass_params(data: SimData, mass: Array, mask: Array | None = None) -> SimData:
    mass = jnp.atleast_3d(mass)
    assert mass.shape[2] == 1, f"Expected shape (n_worlds, n_drones | 1, 1), is {mass.shape}"
    return data.replace(params=leaf_replace(data.params, mask, mass=mass))


@jax.jit
def _randomize_inertia_params(data: SimData, new_j: Array, mask: Array | None = None) -> SimData:
    new_j_inv = jnp.linalg.inv(new_j)
    return data.replace(params=leaf_replace(data.params, mask, J=new_j, J_INV=new_j_inv))
