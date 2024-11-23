import gymnasium
import numpy as np
from ml_collections import config_dict

import crazyflow.crazyflow_env.envs
from crazyflow.control.controller import Control, Controller
from crazyflow.sim.physics import Physics

# set config for simulation
sim_config = config_dict.ConfigDict()
sim_config.device = "cpu"
sim_config.physics = Physics.sys_id
sim_config.control = Control.default
sim_config.controller = Controller.default
sim_config.freq = 60
sim_config.n_drones=1
sim_config.n_worlds=20

SEED=42

envs = gymnasium.make_vec(
    "crazyflow_env/CrazyflowVectorEnvReachGoal-v0",
    max_episode_steps=1000,
    return_datatype="numpy",
    num_envs=sim_config.n_worlds, 
    jax_random_key=SEED,
    **sim_config,
)

# action for going up (in attitude control)
action = np.array(
    [[[0.3, 0, 0, 0] for _ in range(sim_config.n_drones)] for _ in range(sim_config.n_worlds)], dtype=np.float32
).reshape(sim_config.n_worlds, -1)

obs, info = envs.reset_all(seed=SEED)

# Step through the environment
for _ in range(1500):
    observation, reward, terminated, truncated, info = envs.step(action)
    envs.render()
    
envs.close()
