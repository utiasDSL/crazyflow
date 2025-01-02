import gymnasium
import numpy as np
from gymnasium.wrappers.vector import JaxToNumpy  # , JaxToTorch
from ml_collections import config_dict
from scipy.interpolate import splev

from crazyflow.control.controller import Control, Controller
from crazyflow.gymnasium_envs import CrazyflowRL
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

    # Create environment that contains a figure eight trajectory. You can parametrize the observation space, i.e., which part of the trajectory is contained in the observation. Please refer to the documentation of the environment for more information.
    envs = gymnasium.make_vec(
        "DroneFigureEightTrajectory-v0",
        n_trajectory_sample_points=10,
        dt_trajectory_sample_points=0.1,
        trajectory_time=10.0,
        render_trajectory_sample=True,  # useful for debug purposes
        time_horizon_in_seconds=10.0,
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
    action[..., 0] = 0.34

    obs, info = envs.reset(seed=SEED)

    # The trajectory is defined as a scipy spline. Its parameter can be retrieved using `envs.unwrapped.tck`. The spline can be reconstructed using scipy's splev.
    spline_params = envs.unwrapped.tck
    tau = envs.unwrapped.tau  # 1D parameters of the spline for the current timestep, in [0,1]
    value = splev(tau, spline_params)

    # Step through the environment
    for _ in range(1500):
        observation, reward, terminated, truncated, info = envs.step(action)
        envs.render()

    envs.close()


if __name__ == "__main__":
    main()
