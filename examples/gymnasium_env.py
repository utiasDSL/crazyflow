import gymnasium
import numpy as np
from ml_collections import config_dict

from crazyflow.control.controller import Control, Controller
from crazyflow.sim.physics import Physics

# set config for simulation
sim_config = config_dict.ConfigDict()
sim_config.device = "cpu"
sim_config.physics = Physics.sys_id
sim_config.control = Control.default
sim_config.controller = Controller.default
sim_config.control_freq = 50
sim_config.n_drones = 1
sim_config.n_worlds = 20

SEED = 42

envs = gymnasium.make_vec(
    "DroneLanding-v0",
    time_horizon_in_seconds=2,
    return_datatype="numpy",
    num_envs=sim_config.n_worlds,
    jax_random_key=SEED,
    **sim_config,
)

# action for going up (in attitude control). NOTE actions are rescaled in the environment
action = np.array(
    [[[-0.2, 0, 0, 0] for _ in range(sim_config.n_drones)] for _ in range(sim_config.n_worlds)],
    dtype=np.float32,
).reshape(sim_config.n_worlds, -1)

obs, info = envs.reset_all(seed=SEED)

# Step through the environment
for _ in range(1500):
    observation, reward, terminated, truncated, info = envs.step(action)
    envs.render()

envs.close()
