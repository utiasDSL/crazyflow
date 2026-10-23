"""Unit tests for simulation integrators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.sim import Dynamics, Sim
from crazyflow.sim.integration import Integrator


@pytest.mark.unit
@pytest.mark.parametrize("integrator", Integrator)
@pytest.mark.parametrize("dynamics", Dynamics)
def test_integrators_close_to_default(dynamics: Dynamics, integrator: Integrator, device: str):
    """Every integrator stays close to the default integrator after two simulation steps."""
    sim = Sim(dynamics=dynamics, integrator=integrator, device=device)
    default_sim = Sim(dynamics=dynamics, device=device)
    for s in (sim, default_sim):
        # Spawn the drone in the air to induce motion during the steps
        states = s.data.states.replace(pos=s.data.states.pos.at[..., 2].set(2.0))
        s.data = s.data.replace(states=states)
    sim.step(2)
    default_sim.step(2)

    data, structure = jax.tree.flatten_with_path(sim.data)
    default_data, default_structure = jax.tree.flatten(default_sim.data)
    assert structure == default_structure, "Simulation data structures must match"
    for (path, value), default_value in zip(data, default_data, strict=True):
        name = jax.tree_util.keystr(path)
        if isinstance(value, jnp.ndarray):
            if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
                np.testing.assert_array_equal(
                    jax.random.key_data(value), jax.random.key_data(default_value), err_msg=name
                )
            elif jnp.issubdtype(value.dtype, jnp.inexact):
                # Rotor speeds differ by several percent between Euler and RK4.
                np.testing.assert_allclose(
                    value, default_value, rtol=2e-2, atol=1e-4, err_msg=name, strict=True
                )
            else:
                np.testing.assert_array_equal(value, default_value, err_msg=name, strict=True)
        else:
            assert value == default_value, f"{name}: Value mismatch"
    sim.close()
    default_sim.close()
