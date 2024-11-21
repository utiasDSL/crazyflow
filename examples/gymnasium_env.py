import gymnasium
import numpy as np
import crazyflow.crazyflow_env.envs
from ml_collections import config_dict

from crazyflow.control.controller import Control, Controller
from crazyflow.sim.core import Sim
from crazyflow.sim.physics import Physics


# set config for simulation
sim_config = config_dict.ConfigDict()
sim_config.device = "cpu"
sim_config.physics = Physics.sys_id
sim_config.control = Control.default
sim_config.controller = Controller.default
sim_config.freq = 60
sim_config.n_drones=10
sim_config.n_worlds=20

envs = gymnasium.make_vec(
    "crazyflow_env/CrazyflowVectorEnv-v0",
    max_episode_steps=200,
    return_datatype="numpy",
    num_envs=sim_config.n_worlds,
    **sim_config,
)

# action for going up (in attitude control)
action = np.array(
    [[[0.3, 0, 0, 0] for _ in range(sim_config.n_drones)] for _ in range(sim_config.n_worlds)], dtype=np.float32
).reshape(sim_config.n_worlds, -1)

obs, info = envs.reset(seed=42)

# Step through the environment
for _ in range(1500):
    observation, reward, terminated, truncated, info = envs.step(action)
    envs.render()
    
envs.close()
