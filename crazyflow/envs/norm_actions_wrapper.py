import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector import VectorWrapper
from gymnasium.vector.utils import batch_space
from jax import Array

from crazyflow.envs.drone_env import DroneEnv


class NormalizeActions(VectorWrapper):
    """Normalize the actions to be in the range [-1, 1].

    This wrapper clips the expected actions to [-1,1] and rescales them to the action space expected
    in simulation.
    """

    def __init__(self, env: DroneEnv):
        super().__init__(env)
        # Simulation action space bounds
        self.action_sim_low = self.single_action_space.low
        self.action_sim_high = self.single_action_space.high

        # Compute scale and mean for rescaling
        device = env.device
        self._scale = jnp.array((self.action_sim_high - self.action_sim_low) / 2.0, device=device)
        self._mean = jnp.array((self.action_sim_high + self.action_sim_low) / 2.0, device=device)

        # Modify the wrapper's action space to [-1, 1]
        self.single_action_space.low = -np.ones_like(self.action_sim_low)
        self.single_action_space.high = np.ones_like(self.action_sim_high)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def step(self, actions: Array) -> tuple[dict, Array, Array, Array, dict]:
        action = self._scale_actions(actions, self._scale, self._mean)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    @staticmethod
    @jax.jit
    def _scale_actions(actions: Array, scale: Array, mean: Array) -> Array:
        """Rescale and clip actions from [-1, 1] to [action_sim_low, action_sim_high]."""
        return jnp.clip(actions, -1.0, 1.0) * scale + mean
