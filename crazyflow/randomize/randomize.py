import jax
from jax import Array
from crazyflow.sim.structs import SimParams

def randomize_mass(sim, new_masses):
    """Randomize mass from a new masses. 
    
        Args:
            sim: The simulation object.
            new_masses: The new masses.
        
        Warning:
            This currently only works for analytical dynamics.
    """
    #TODO: Domain randomization when using MuJoCo as dynamics engine
    #TODO: recompile sim._sync_mjx? sim._mjx_model = _randomize_mass(sim._mjx_model, new_masses)
    #TODO: Parameters randomization for sys_id model
    randomized_params = _randomize_mass_params(sim.params, new_masses)
    sim.defaults["params"] = randomized_params
    sim.params = randomized_params

def randomize_inertia(sim, new_js, new_j_invs):
    """Randomize inertia tensor from a new inertia tensors. 
    
        Args:
            sim: The simulation object.
            new_js: The new inertia tensors.
        
        Warning:
            This currently only works for analytical dynamics.
    """
    #TODO: same as randomize_mass
    randomized_params = _randomize_inertia_params(sim.params, new_js, new_j_invs)
    sim.defaults["params"] = randomized_params
    sim.params = randomized_params

@jax.jit
def _randomize_mass_params(params: SimParams, new_masses: Array):
    return params.replace(mass=new_masses)

@jax.jit
def _randomize_inertia_params(params: SimParams, new_js: Array, new_j_invs: Array):
    return params.replace(J=new_js, J_INV=new_j_invs)