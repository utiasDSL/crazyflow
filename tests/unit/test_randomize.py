import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.serialization import to_state_dict

from crazyflow.control import Control
from crazyflow.randomize import randomize_inertia, randomize_mass
from crazyflow.sim import Sim


@pytest.mark.unit
@pytest.mark.parametrize("n_worlds", [1, 3])
def test_randomize_mass(n_worlds: int):
    sim = Sim(n_worlds=n_worlds, n_drones=4, control=Control.state)

    add_on_mass = np.random.uniform(-0.005, 0.005, size=(sim.n_worlds, sim.n_drones))
    masses = np.ones((sim.n_worlds, sim.n_drones)) * 0.025
    randomized_masses = masses + add_on_mass
    randomize_mass(sim, randomized_masses)

    for k, v in to_state_dict(sim.data.params).items():
        default = getattr(sim.default_data.params, k)
        if not isinstance(v, jnp.ndarray) or not isinstance(default, jnp.ndarray):
            continue
        assert v.shape == default.shape, f"{k} shape mismatch"
        assert v.device == default.device, f"{k} device mismatch"
        if k == "mass":
            randomized_masses = randomized_masses.reshape(sim.n_worlds, sim.n_drones, 1)
            assert np.allclose(v, randomized_masses), f"{k} value mismatch, {v - randomized_masses}"


@pytest.mark.unit
def test_randomize_mass_masked():
    sim = Sim(n_worlds=3, n_drones=4, control=Control.state)

    add_on_mass = np.random.uniform(-0.005, 0.005, size=(sim.n_worlds, sim.n_drones))
    masses = np.ones((sim.n_worlds, sim.n_drones)) * 0.025
    randomized_masses = masses + add_on_mass
    randomized_masses = randomized_masses.reshape(sim.n_worlds, sim.n_drones, 1)
    randomized_masses = jnp.array(randomized_masses, device=sim.device)

    mask = jnp.array([True, False, True])
    randomize_mass(sim, randomized_masses, mask)

    for k, v in to_state_dict(sim.data.params).items():
        default = getattr(sim.default_data.params, k)
        if not isinstance(v, jnp.ndarray) or not isinstance(default, jnp.ndarray):
            continue
        assert v.shape == default.shape, f"{k} shape mismatch"
        assert v.device == default.device, f"{k} device mismatch"
        if k == "mass":
            # Check that masked worlds match randomized masses
            assert np.all(v[0] == randomized_masses[0]), "First world should be randomized"
            assert np.all(v[1] == default[1]), "Second world should not be randomized"
            assert np.all(v[2] == randomized_masses[2]), "Third world should be randomized"


@pytest.mark.unit
def test_randomize_inertia():
    sim = Sim(n_worlds=2, n_drones=4, control=Control.state)

    J_residual = jax.random.normal(jax.random.key(0), (sim.n_worlds, sim.n_drones, 3, 3)) * 1e-5
    J = sim.data.params.J + J_residual
    randomize_inertia(sim, J)

    for k, v in to_state_dict(sim.data.params).items():
        default = getattr(sim.default_data.params, k)
        if not isinstance(v, jnp.ndarray) or not isinstance(default, jnp.ndarray):
            continue
        assert v.shape == default.shape, f"{k} shape mismatch"
        assert v.device == default.device, f"{k} device mismatch"
        if k == "J":
            assert jnp.all(v == J), f"{k} value mismatch"


@pytest.mark.unit
def test_randomize_inertia_masked():
    sim = Sim(n_worlds=3, n_drones=4, control=Control.state)

    J_residual = jax.random.normal(jax.random.key(0), (sim.n_worlds, sim.n_drones, 3, 3)) * 1e-5
    J = sim.data.params.J + J_residual
    mask = jnp.array([True, False, True])
    randomize_inertia(sim, J, mask)

    for k, v in to_state_dict(sim.data.params).items():
        default = getattr(sim.default_data.params, k)
        if not isinstance(v, jnp.ndarray) or not isinstance(default, jnp.ndarray):
            continue
        assert v.shape == default.shape, f"{k} shape mismatch"
        assert v.device == default.device, f"{k} device mismatch"
        if k == "J":
            # Check that masked worlds match randomized inertias
            assert jnp.all(v[0] == J[0]), "First world should be randomized"
            assert jnp.all(v[1] == default[1]), "Second world should not be randomized"
            assert jnp.all(v[2] == J[2]), "Third world should be randomized"
