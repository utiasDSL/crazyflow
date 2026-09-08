from dataclasses import fields

import jax.numpy as jnp
import numpy as np
import pytest

import crazyflow  # noqa: F401, register gymnasium envs
from crazyflow.control import Control
from crazyflow.sim import Dynamics, Sim
from crazyflow.sim.data import SimData
from crazyflow.utils import CORE_NDIM_KEY


def vectorize_state(data: SimData) -> jnp.ndarray:
    """Stack the drone states into a (n_worlds, n_drones, 17) array."""
    s = data.states
    return jnp.concat([s.pos, s.quat, s.vel, s.ang_vel, s.rotor_vel], axis=-1)


@pytest.mark.integration
@pytest.mark.parametrize("dynamics", Dynamics)
def test_reset_during_simulation(dynamics: Dynamics):
    """Test reset behavior during an active simulation."""
    sim = Sim(dynamics=dynamics, control=Control.attitude)
    n_steps = 3
    random_cmds = np.random.rand(n_steps, 1, 1, 4)
    # Run simulation once
    for cmd in random_cmds:
        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)
    final_state = vectorize_state(sim.data)

    sim.reset()
    assert jnp.all(sim.data.core.steps == 0)
    assert jnp.all(vectorize_state(sim.data) == vectorize_state(sim.default_data))

    # Verify simulation is identical when running again
    for i in range(n_steps):
        sim.attitude_control(random_cmds[i])
        sim.step(sim.freq // sim.control_freq)
    assert jnp.all(vectorize_state(sim.data) == final_state)


@pytest.mark.integration
@pytest.mark.parametrize("dynamics", Dynamics)
def test_reset_multi_world(dynamics: Dynamics):
    """Test reset behavior with multiple worlds."""
    n_worlds, n_drones = 2, 2
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, dynamics=dynamics, control=Control.attitude)
    n_steps = 3
    random_cmds = np.random.rand(n_steps, n_worlds, n_drones, 4)
    # Run simulation once
    for i in range(n_steps):
        sim.attitude_control(random_cmds[i])
        assert isinstance(sim.data.controls.attitude.staged_cmd, jnp.ndarray)
        assert isinstance(sim.data.controls.attitude.cmd, jnp.ndarray)
        sim.step(sim.freq // sim.control_freq)
    final_state = vectorize_state(sim.data)

    sim.reset()
    assert jnp.all(sim.data.core.steps == 0)
    assert jnp.all(vectorize_state(sim.data) == vectorize_state(sim.default_data))

    # Verify simulation is identical when running again
    for cmd in random_cmds:
        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)
    assert jnp.all(vectorize_state(sim.data) == final_state)


@pytest.mark.integration
@pytest.mark.parametrize("dynamics", Dynamics)
def test_reset_masked_batched_params(dynamics: Dynamics):
    """Masked reset restores per-world parameter arrays only for the masked worlds."""
    n_worlds, n_drones, n_steps = 3, 2, 5
    sim = Sim(n_worlds, n_drones, dynamics=dynamics, control=Control.attitude)
    random_cmds = np.random.rand(n_steps, n_worlds, n_drones, 4)
    # Reference trajectory with the shared default parameters
    for cmd in random_cmds:
        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)
    states_default = vectorize_state(sim.data)
    sim.reset()

    # Give every drone its own copy of all parameters, scaled per world
    scale = jnp.array([1.0, 1.1, 0.9])[:, None] * jnp.ones((n_worlds, n_drones))
    params = sim.data.params
    batched_params = {
        f.name: getattr(params, f.name)
        * scale.reshape(scale.shape + (1,) * f.metadata[CORE_NDIM_KEY])
        for f in fields(params)
    }
    batched_params["J_inv"] = jnp.linalg.inv(batched_params["J"])
    sim.data = sim.data.replace(params=params.replace(**batched_params))

    for cmd in random_cmds:
        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)
    states_param_change = vectorize_state(sim.data)
    assert jnp.allclose(states_param_change[0], states_default[0])  # Should stay the same
    assert not jnp.allclose(states_param_change[1], states_default[1])  # Must have changed
    assert not jnp.allclose(states_param_change[2], states_default[2])

    mask = np.array([False, True, False])
    sim.reset(mask=mask)
    # Only world 1 is restored to the default parameters, the other worlds keep theirs
    for name, value in batched_params.items():
        current, default = getattr(sim.data.params, name), getattr(sim.default_data.params, name)
        assert current.shape == value.shape, f"{name}: masked reset changed the shape"
        assert jnp.allclose(current[1], jnp.broadcast_to(default, value.shape)[1]), name
        assert jnp.allclose(current[~mask], value[~mask]), name
    # The same holds for the states
    assert jnp.all(vectorize_state(sim.data)[1] == vectorize_state(sim.default_data)[1])
    assert jnp.all(vectorize_state(sim.data)[~mask] == states_param_change[~mask])
    assert jnp.all(sim.data.core.steps[1] == 0)
    assert jnp.all(sim.data.core.steps[~mask] > 0)

    # A full reset restores the shared default parameters
    sim.reset()
    for name in batched_params:
        current, default = getattr(sim.data.params, name), getattr(sim.default_data.params, name)
        assert current.shape == default.shape, f"{name}: full reset did not restore the shape"
        assert jnp.all(current == default), name
