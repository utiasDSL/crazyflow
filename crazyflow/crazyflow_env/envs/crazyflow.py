from typing import Optional, Tuple, Literal, Dict, Union
import numpy as np
import jax
import jax.numpy as jnp
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from crazyflow.control.controller import Control
from crazyflow.sim.core import Sim
import math


class CrazyflowVectorEnv(VectorEnv):
    """JAX Gymnasium environment for Crazyflie simulation."""

    def __init__(
        self,
        num_envs: int = 1,
        num_drones_per_env: int = 1,
        max_episode_steps: int = 1000,
        return_datatype: Literal["numpy", "jax"] = "jax",
        **kwargs: dict,
    ):
        """
        Args:
            num_envs (int): The number of parallel environments (simulations) to run.
            num_drones_per_env (int): The number of drones in each environment.
            max_episode_steps (int): The maximum number of steps per episode.
            return_datatype (Literal["numpy", "jax"]): The data type for returned arrays, either "numpy" or "jax". If specified as "numpy", the returned arrays will be numpy arrays on the CPU. If specified as "jax", the returned arrays will be jax arrays on the "device" specifiedf or the simulation.
            **kwargs: Takes arguments that are passed to the Crazyfly simulation .
        """

        self.num_envs = num_envs
        self.num_drones_per_env = num_drones_per_env
        self.max_episode_steps = max_episode_steps
        self.return_datatype = return_datatype

        self.sim = Sim(n_worlds=num_envs, n_drones=num_drones_per_env, **kwargs)

        self.steps = jnp.zeros((num_envs), dtype=jnp.int32)
        self.prev_done = jnp.zeros((num_envs), dtype=jnp.bool_)

        self.single_action_space = spaces.Box(
            -1,
            1,
            shape=(math.prod(self.sim._controls[self.sim.control].shape[1:]),),
            dtype=jnp.float32,
        )
        self.action_space = batch_space(self.single_action_space, num_envs)

        _obs_size = 0
        for key in self.sim.states.keys():
            _obs_size += math.prod(self.sim.states[key].shape[1:])
        self.single_observation_space = spaces.Box(
            -jnp.inf, jnp.inf, shape=(_obs_size,), dtype=jnp.float32
        )
        self.observation_space = batch_space(self.single_observation_space, num_envs)

    def step(
        self, action: Union[jnp.ndarray, np.ndarray]
    ) -> Tuple[
        Union[jnp.ndarray, np.ndarray],
        Union[jnp.ndarray, np.ndarray],
        Union[jnp.ndarray, np.ndarray],
        Union[jnp.ndarray, np.ndarray],
        Dict,
    ]:
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        action = self._maybe_to_jax(action)
        action = action.reshape((self.num_envs, self.num_drones_per_env, -1))

        if self.sim.control == Control.state:
            self.sim.state_control(action)
        elif self.sim.control == Control.attitude:
            self.sim.attitude_control(action)
        elif self.sim.control == Control.thrust:
            self.sim.thrust_control(action)
        else:
            raise ValueError(f"Invalid control type {self.sim.control}")

        self.sim.step()

        terminated = self._get_terminated()
        reward = self._get_reward()
        truncated = self.steps >= self.max_episode_steps

        # Reset all environments which terminated or were truncated in the last step
        self.sim.reset(mask=self.prev_done)

        # Compute the new states without in-place mutation
        # TODO: check if this is the most performance way to do this
        self.steps = jax.lax.select(self.prev_done, jnp.zeros_like(self.steps), self.steps + 1)
        reward = jax.lax.select(self.prev_done, jnp.zeros_like(reward), reward)
        terminated = jax.lax.select(self.prev_done, jnp.full_like(terminated, False), terminated)
        truncated = jax.lax.select(self.prev_done, jnp.full_like(truncated, False), truncated)

        self.prev_done = jnp.logical_or(terminated, truncated)

        return (
            self._get_obs(),
            self._maybe_to_numpy(reward),
            self._maybe_to_numpy(terminated),
            self._maybe_to_numpy(truncated),
            {},
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Resets ALL (!) environments
        if options is None:
            options = {}
        self.sim.reset()

        self.steps = jnp.zeros(self.num_envs, dtype=jnp.int32)
        self.prev_done = jnp.zeros(self.num_envs, dtype=jnp.bool_)

        return self._get_obs(), {}

    def render(self):
        self.sim.render()

    def _get_obs(self) -> Dict[str, jnp.ndarray]:
        # Returns observations for all environments
        obs = {
            key: self._maybe_to_numpy(value.reshape(*value.shape[:-2], -1))
            for key, value in self.sim.states.items()
        }
        return obs

    def _get_reward(self) -> jnp.ndarray:
        # Returns rewards for all environments
        return jnp.zeros((self.num_envs), dtype=jnp.float32)

    def _get_terminated(self) -> jnp.ndarray:
        # Returns termination status for all environments
        return jnp.zeros((self.num_envs), dtype=jnp.bool_)

    def _maybe_to_numpy(self, data: Union[jnp.ndarray, np.ndarray]) -> np.ndarray:
        if self.return_datatype == "numpy" and not isinstance(data, np.ndarray):
            return jax.device_get(data)
        return data
    
    def _maybe_to_jax(self, data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        # Potentially dont need to check for "self.return_datatype == "jax"", as simulation works with jax arrays in any case
        if self.return_datatype == "jax" and not isinstance(data, jnp.ndarray):
            return jax.device_put(data)
        return data
