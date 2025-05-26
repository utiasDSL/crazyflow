import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.control import Control
from crazyflow.sim import Physics, Sim


import gymnasium
import numpy as np
from gymnasium.wrappers.vector import JaxToNumpy  # , JaxToTorch
import crazyflow  # noqa: F401, register gymnasium envs


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
    
@pytest.mark.integration
def test_gymnasium_reset():
    """Test reset behavior of the DroneReachPos-v0 environment."""
    SEED = 42
    envs = gymnasium.make_vec("DroneReachPos-v0", num_envs=1, freq=50, time_horizon_in_seconds=2)

    envs = JaxToNumpy(envs)
    obs, _ = envs.reset(
        seed=SEED,
        options={
            "pos_min": np.array([-1.0, 1.0, 1.0]),
            "pos_max": np.array([-1.0, 1.0, 1.0]),
            "vel_min": 0.0,
            "vel_max": 0.0,
            "goal_pos_min": np.array([-1.0, 1.0, 1.0]),
            "goal_pos_max": np.array([-1.0, 1.0, 1.0]),
        },
    )
    assert np.all(obs["pos"] == np.array([[-1.0, 1.0, 1.0]]))
    assert np.all(obs["difference_to_goal"] == np.array([[.0, .0, .0]]))
    assert np.all(obs["vel"] == np.array([[0.0, 0.0, 0.0]]))
    