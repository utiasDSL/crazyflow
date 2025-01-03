import gymnasium
import numpy as np
from gymnasium.wrappers.vector import JaxToNumpy  # , JaxToTorch
from ml_collections import config_dict

from crazyflow.control.controller import Control, Controller
from crazyflow.sim.physics import Physics


def main():
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
        "DroneLanding-v0",  # choose the environment
        time_horizon_in_seconds=2,
        num_envs=sim_config.n_worlds,
        **sim_config,
    )

    # RL wrapper
    # envs = CrazyflowRL(
    #     envs
    # )  # This wrapper to clips the actions to [-1, 1], and rescales them subsequently for use with common DRL libraries.

    envs = JaxToNumpy(
        envs
    )  #  This wrapper makes it possible to interact with the environment using numpy arrays, if desired. JaxToTorch is available as well.

    # dummy action for going up (in attitude control)
    action = np.zeros((sim_config.n_worlds * sim_config.n_drones, 4), dtype=np.float32)
    action[..., 0] = 0.4

    obs, info = envs.reset(seed=SEED)

    # Step through the environment
    for _ in range(300):
        observation, reward, terminated, truncated, info = envs.step(action)
        envs.render()

    envs.close()


if __name__ == "__main__":
    main()
