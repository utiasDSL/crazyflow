import gymnasium
import numpy as np
import crazyflow.crazyflow_env.envs
from ml_collections import config_dict

from crazyflow.control.controller import Control, Controller
from crazyflow.sim.core import Sim
from crazyflow.sim.physics import Physics


num_envs = 20
num_drones_per_env = 10

# set config for simulation
sim_config = config_dict.ConfigDict()
sim_config.device = "cpu"
sim_config.physics = Physics.sys_id
sim_config.control = Control.default
sim_config.controller = Controller.default
sim_config.freq = 60

envs = gymnasium.make_vec(
    "crazyflow_env/CrazyflowVectorEnv-v0",
    num_envs=num_envs,
    num_drones_per_env=num_drones_per_env,
    max_episode_steps=200,
    **sim_config,
)

# determine init positions of drones as a grid
N = int(np.ceil(np.sqrt(num_drones_per_env)))
points = np.linspace(-0.5 * (N - 1), 0.5 * (N - 1), N)
x, y = np.meshgrid(points, points)
grid = np.stack((x.flatten(), y.flatten(), np.zeros_like(y).flatten()), axis=-1)
grid = grid[:num_drones_per_env]

# action for going up (in attitude control)
action = np.array(
    [[[0.3, 0, 0, 0] for _ in range(num_drones_per_env)] for _ in range(num_envs)], dtype=np.float32
).reshape(num_envs, -1)

obs, info = envs.reset(seed=42)

# Step through the environment
for _ in range(1500):
    observation, reward, terminated, truncated, info = envs.step(action)
    envs.render()
    
envs.close()
