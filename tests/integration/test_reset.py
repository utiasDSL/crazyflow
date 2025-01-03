import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.control import Control
from crazyflow.sim import Physics, Sim


@pytest.mark.integration
@pytest.mark.parametrize("physics", Physics)
def test_reset_during_simulation(physics: Physics):
    """Test reset behavior during an active simulation."""
    sim = Sim(physics=physics, control=Control.attitude)
    # Run simulation
    n_steps = 3
    random_cmds = np.random.rand(n_steps, 1, 1, 4)
    # Run simulation once
    for cmd in random_cmds:
        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)
    final_pos = sim.data.states.pos.copy()
    final_quat = sim.data.states.quat.copy()

    sim.reset()
    assert jnp.all(sim.data.core.steps == 0)
    assert jnp.all(sim.data.states.pos == sim.default_data.states.pos)
    assert jnp.all(sim.data.states.quat == sim.default_data.states.quat)

    # Verify simulation is identical when running again
    for i in range(n_steps):
        sim.attitude_control(random_cmds[i])
        sim.step(sim.freq // sim.control_freq)
    assert jnp.all(sim.data.states.pos == final_pos)
    assert jnp.all(sim.data.states.quat == final_quat)


@pytest.mark.integration
@pytest.mark.parametrize("physics", Physics)
def test_reset_multi_world(physics: Physics):
    """Test reset behavior with multiple worlds."""
    n_worlds, n_drones = 2, 2
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=physics, control=Control.attitude)

    n_steps = 3
    random_cmds = np.random.rand(n_steps, n_worlds, n_drones, 4)
    # Run simulation once
    for i in range(n_steps):
        sim.attitude_control(random_cmds[i])
        assert isinstance(sim.data.controls.attitude, jnp.ndarray)
        sim.step(sim.freq // sim.control_freq)
    final_pos = sim.data.states.pos.copy()
    final_quat = sim.data.states.quat.copy()

    sim.reset()
    assert jnp.all(sim.data.core.steps == 0)
    assert jnp.all(sim.data.states.pos == sim.default_data.states.pos)
    assert jnp.all(sim.data.states.quat == sim.default_data.states.quat)

    # Verify simulation is identical when running again
    for cmd in random_cmds:
        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)
    assert jnp.all(sim.data.states.pos == final_pos)
    assert jnp.all(sim.data.states.quat == final_quat)
