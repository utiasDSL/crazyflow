import gymnasium
import jax.numpy as jnp
import numpy as np
from gymnasium.wrappers.vector import JaxToNumpy  # , JaxToTorch

from crazyflow.envs import NormalizeActions  # noqa: F401
from crazyflow.utils import enable_cache


def main():
    enable_cache()
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
    )

    # NormalizeActions wrapper to clip the actions to [-1, 1] and rescale them for use with common
    # DRL libraries.
    envs = NormalizeActions(envs)
    envs = JaxToNumpy(envs)

    # dummy action for going up (in attitude control)
    action = np.zeros((20, 4), dtype=np.float32)
    action[..., 0] = -0.2

    obs, info = envs.reset()
    # Step through the environment
    for _ in range(1_000):
        # Prevent alignment warnings. Related issue: https://github.com/jax-ml/jax/issues/29810
        # TODO: Remove once https://github.com/jax-ml/jax/pull/29963 is merged.
        action = np.asarray(jnp.asarray(action))
        observation, reward, terminated, truncated, info = envs.step(action)
        envs.render()

    envs.close()


if __name__ == "__main__":
    main()
