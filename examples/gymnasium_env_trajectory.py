import gymnasium
import numpy as np
from gymnasium.wrappers.vector import JaxToNumpy  # , JaxToTorch

from crazyflow.gymnasium_envs import CrazyflowRL  # noqa: F401
from crazyflow.utils import enable_cache


def main():
    enable_cache()
    SEED = 42
    # Create environment that contains a figure eight trajectory. You can parametrize the
    # observation space, i.e., which part of the trajectory is contained in the observation. Please
    # refer to the documentation of the environment for more information.
    envs = gymnasium.make_vec(
        "DroneFigureEightTrajectory-v0",
        num_envs=20,
        freq=50,
        n_samples=10,
        samples_dt=0.1,
        trajectory_time=10.0,
        render_samples=True,
    )

    # RL wrapper to clip the actions to [-1, 1] and rescale them for use with common DRL libraries.
    # envs = CrazyflowRL(envs)

    # This wrapper makes it possible to interact with the environment using numpy arrays, if
    # desired. JaxToTorch is available as well.
    envs = JaxToNumpy(envs)

    # dummy action for going up (in attitude control)
    action = np.zeros((20, 4), dtype=np.float32)
    action[..., 0] = 0.31

    obs, info = envs.reset(seed=SEED)
    # Step through the environment
    for _ in range(500):
        observation, reward, terminated, truncated, info = envs.step(action)
        envs.render()

    envs.close()


if __name__ == "__main__":
    main()
