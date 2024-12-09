import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.control.controller import Control
from crazyflow.sim.core import Physics, Sim


@pytest.mark.integration
@pytest.mark.parametrize("physics", Physics)
def test_reset_during_simulation(physics: Physics):
    """Test reset behavior during an active simulation."""
    if physics == Physics.mujoco:
        pytest.skip("Mujoco physics not implemented")  # TODO: Remove once MuJoCo is implemented

    sim = Sim(physics=physics, control=Control.attitude)
    # Run simulation
    n_steps = 3
    random_cmds = np.random.rand(n_steps, 1, 1, 4)
    # Run simulation once
    for cmd in random_cmds:
        sim.attitude_control(cmd)
        sim.step()
    final_pos = sim.states.pos.copy()
    final_quat = sim.states.quat.copy()

    sim.reset()
    assert jnp.all(sim.steps == 0)
    assert jnp.all(sim.states.pos == sim.defaults["states"].pos)
    assert jnp.all(sim.states.quat == sim.defaults["states"].quat)

    # Verify simulation is identical when running again
    for i in range(n_steps):
        sim.attitude_control(random_cmds[i])
        sim.step()
    assert jnp.all(sim.states.pos == final_pos)
    assert jnp.all(sim.states.quat == final_quat)


@pytest.mark.integration
@pytest.mark.parametrize("physics", [Physics.analytical, Physics.sys_id])
def test_reset_multi_world(physics: Physics):
    """Test reset behavior with multiple worlds."""
    n_worlds, n_drones = 2, 2
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=physics, control=Control.attitude)

    n_steps = 3
    random_cmds = np.random.rand(n_steps, n_worlds, n_drones, 4)
    # Run simulation once
    for i in range(n_steps):
        sim.attitude_control(random_cmds[i])
        assert isinstance(sim.controls.attitude, jnp.ndarray)
        sim.step()
    final_pos = sim.states.pos.copy()
    final_quat = sim.states.quat.copy()

    sim.reset()
    assert jnp.all(sim.steps == 0)
    assert jnp.all(sim.states.pos == sim.defaults["states"].pos)
    assert jnp.all(sim.states.quat == sim.defaults["states"].quat)

    # Verify simulation is identical when running again
    for cmd in random_cmds:
        sim.attitude_control(cmd)
        sim.step()
    assert jnp.all(sim.states.pos == final_pos)
    assert jnp.all(sim.states.quat == final_quat)
