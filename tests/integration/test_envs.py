import gymnasium
import numpy as np
import pytest
from gymnasium.wrappers.vector import JaxToNumpy

import crazyflow  # noqa: F401, register gymnasium envs


@pytest.mark.integration
def test_gymnasium_reset():
    """Test reset behavior of the DroneReachPos-v0 environment."""
    SEED = 42
    envs = gymnasium.make_vec(
        "DroneReachPos-v0",
        num_envs=1,
        freq=50,
        max_episode_time=2,
        pos_min=np.array([-1.0, 1.0, 1.0]),
        pos_max=np.array([-1.0, 1.0, 1.0]),
        vel_min=0.0,
        vel_max=0.0,
    )

    envs = JaxToNumpy(envs)
    obs, _ = envs.reset(
        seed=SEED,
        options={
            "goal_pos_min": np.array([-1.0, 1.0, 1.0]),
            "goal_pos_max": np.array([-1.0, 1.0, 1.0]),
        },
    )
    assert np.all(obs["pos"] == np.array([[-1.0, 1.0, 1.0]]))
    assert np.all(obs["difference_to_goal"] == np.array([[0.0, 0.0, 0.0]]))
    assert np.all(obs["vel"] == np.array([[0.0, 0.0, 0.0]]))
