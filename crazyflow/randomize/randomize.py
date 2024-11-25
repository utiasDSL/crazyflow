import jax
from jax import Array
from mujoco.mjx import Model


def randomize_mass(sim, new_masses):
    """This is some docstring.

    Warning:
        This currently only works for analytical dynamics.
    """
    # TODO: Do we need to recompile sim._sync_mjx after this?
    # Needed if we use MuJoCo as dynamics engine (currently not implemented)
    sim._mjx_model = _randomize_mass(sim._mjx_model, new_masses)
    # Needed if we use analytical dynamics. This is what we use right now.
    sim.params = sim.params.replace(mass=new_masses)  # TODO: Can this be JITed?
    # sys_id model: You cannot change the mass in the sys_id model, it's given by the identified
    # parameters.


@jax.jit
@jax.vmap
def _randomize_mass(model: Model, new_mass: Array):
    # Double check that at 1 is the correct thing to set. Is this being vmapped correctly? Possibly
    # base plate it at 0, drone0 is at 1, drone2 is at 2, etc.
    model.body_mass = model.body_mass.at[1].set(new_mass)
    return model


@jax.jit
def _randomize_mass_params(params: Params, new_masses: Array):
    return params.replace(mass=new_masses)
