from typing import Optional, Tuple, Union, List
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import jax

from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from crazyflow.control.controller import Control, Controller
from crazyflow.sim.core import Sim
from crazyflow.sim.physics import Physics


# Example for Env and VectorEnv here: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py#L476
class CrazyflowVectorEnv(VectorEnv):

    def __init__(
        self,
        num_envs: int = 1,
        num_drones_per_env: int = 1,
        max_episode_steps: int = 1000,
        **kwargs: dict,
    ):

        self.num_envs = num_envs
        self.num_drones_per_env = num_drones_per_env
        self.max_episode_steps = max_episode_steps

        self.sim = Sim(
            n_worlds=num_envs,
            n_drones=num_drones_per_env,
            **kwargs,
        )

        self.steps = np.zeros((num_envs), dtype=np.int32)
        self.prev_done = np.zeros((num_envs), dtype=np.bool_)

        self.single_action_space = spaces.Box(
            -1, 1, shape=(math.prod(self.sim._controls[self.sim.control].shape[1:]),), dtype=np.float32
        )  # num_drones_per_env * num_controls
        self.action_space = batch_space(self.single_action_space, num_envs)

        _obs_size = 0
        for key in self.sim.states.keys():
            _obs_size += math.prod(
                self.sim.states[key].shape[1:]
            )  # num_obs_per_drone * num_controls
        self.single_observation_space = spaces.Box(
            -np.inf, np.inf, shape=(_obs_size,), dtype=np.float32
        )  # set limits to np.inf as we are expecting a normalization wrapper
        self.observation_space = batch_space(self.single_observation_space, num_envs)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        action = action.reshape((self.num_envs, self.num_drones_per_env, -1))

        # TODO: rescale actions
        if self.sim.control == Control.state:
            # TODO Do I need to check self.sim.controllable() ?
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

        self.steps += 1

        truncated = self.steps >= self.max_episode_steps

        # TODO Reset all environments which terminated or were truncated in the last step

        self.steps[self.prev_done] = 0
        reward[self.prev_done] = 0.0
        terminated[self.prev_done] = False
        truncated[self.prev_done] = False

        self.prev_done = np.logical_or(terminated, truncated)

        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Resets all (!) envs
        super().reset(seed=seed)
        self.sim.reset()

        if "pos" in options:
            self.sim.states["pos"] = options["pos"]

        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_done = np.zeros(self.num_envs, dtype=np.bool_)

        return self._get_obs(), {}
    
   
    def _get_obs(self) -> np.ndarray:
        # Returns observations for all envs in a dictionary. Each entry is of shape (num_envs, num_drones_per_env * num_obs_per_drone)
        return {key: jax.device_get(value.reshape(*value.shape[:-2], -1)) for key, value in self.sim.states.items()}
    
    def _get_reward(self) -> np.ndarray:
        # Returns rewards for all envs
        return np.zeros((self.num_envs), dtype=np.float32)

    def _get_terminated(self) -> np.ndarray:
        # Returns termination status for all envs
        return np.zeros((self.num_envs), dtype=np.bool_)

    def render(self):
        self.sim.render()
