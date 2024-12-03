import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.serialization import to_state_dict

from crazyflow.control.controller import J
from crazyflow.control.controller import Control, Controller
from crazyflow.randomize import randomize_mass, randomize_inertia
from crazyflow.sim.core import Sim
from crazyflow.sim.physics import Physics

from tests.unit.test_sim import skip_unavailable_device

@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_randomize_mass(device: str):
    skip_unavailable_device(device)
    sim = Sim(
        n_worlds=1,
        n_drones=4,
        physics=Physics.analytical,
        control=Control.state,
        controller=Controller.emulatefirmware,
        device=device,
    )
    key = jax.random.PRNGKey(0)
    add_on_mass = jax.random.uniform(key, shape=(sim.n_worlds, sim.n_drones, 1), minval=-0.005, maxval=0.005)
    masses = jnp.ones((sim.n_worlds, sim.n_drones, 1)) * 0.025
    randomized_masses = masses + add_on_mass

    randomize_mass(sim, randomized_masses)

    for k, v in to_state_dict(sim.params).items():
        default = getattr(sim.defaults["params"], k)
        if not isinstance(v, jnp.ndarray) or not isinstance(default, jnp.ndarray):
            continue
        assert v.shape == default.shape, f"{k} shape mismatch"
        assert v.device == default.device, f"{k} device mismatch"
        assert jnp.all(v[0] == default[0]), f"{k} value mismatch"
        if k == "mass":
            assert jnp.all(v == randomized_masses), f"{k} value mismatch"

@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_randomize_inertia(device: str):
    skip_unavailable_device(device)
    sim = Sim(
        n_worlds=1,
        n_drones=4,
        physics=Physics.analytical,
        control=Control.state,
        controller=Controller.emulatefirmware,
        device=device,
    )
    key = jax.random.PRNGKey(0)
    add_on_j = jax.random.uniform(key, shape=(sim.n_worlds, sim.n_drones, 3, 3), minval=-1.5e-5, maxval=1.5e-5)
    j = jnp.array(J, device=sim.device)
    randomized_j = j + add_on_j
    randomized_j_inv = jnp.linalg.inv(randomized_j)

    randomize_inertia(sim, randomized_j, randomized_j_inv)

    for k, v in to_state_dict(sim.params).items():
        default = getattr(sim.defaults["params"], k)
        if not isinstance(v, jnp.ndarray) or not isinstance(default, jnp.ndarray):
            continue
        assert v.shape == default.shape, f"{k} shape mismatch"
        assert v.device == default.device, f"{k} device mismatch"
        assert jnp.all(v[0] == default[0]), f"{k} value mismatch"
        if k == "J":
            assert jnp.all(v == randomized_j), f"{k} value mismatch"
        if k == "J_INV":
            assert jnp.all(v == randomized_j_inv), f"{k} value mismatch"