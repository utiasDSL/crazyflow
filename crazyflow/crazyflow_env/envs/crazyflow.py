import math
from dataclasses import fields
from typing import Dict, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from jax import Array

from crazyflow.control.controller import Control
from crazyflow.sim.core import Sim


class CrazyflowVectorEnv(VectorEnv):
    """JAX Gymnasium environment for Crazyflie simulation."""

    def __init__(
        self,
        *,
        num_envs: int = 1,
        max_episode_steps: int = 1000,
        return_datatype: Literal["numpy", "jax"] = "jax",
        **kwargs: dict,
    ):
        """Summary: Initializes the CrazyflowVectorEnv.

        Args:
        max_episode_steps (int): The maximum number of steps per episode.
            return_datatype (Literal["numpy", "jax"]): The data type for returned arrays, either "numpy" or "jax". If specified as "numpy", the returned arrays will be numpy arrays on the CPU. If specified as "jax", the returned arrays will be jax arrays on the "device" specifiedf or the simulation.
            **kwargs: Takes arguments that are passed to the Crazyfly simulation .
        """
        assert "n_worlds" in kwargs, "n_worlds must be specified in kwargs"
        assert "n_drones" in kwargs, "n_drones must be specified in kwargs"
        assert num_envs == kwargs["n_worlds"], "num_envs must be equal to n_worlds"

        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.return_datatype = return_datatype
        self.device = jax.devices(kwargs["device"])[0]

        self.sim = Sim(**kwargs)

        self.prev_done = jnp.zeros((self.sim.n_worlds), dtype=jnp.bool_, device=self.device)

        self.single_action_space = spaces.Box(
            -1,
            1,
            shape=(math.prod(getattr(self.sim.controls, self.sim.control).shape[1:]),),
            dtype=jnp.float32,
        )
        self.action_space = batch_space(self.single_action_space, self.sim.n_worlds)

        self.states_to_exclude_from_obs = ["step", "device"]
        _obs_size = 0
        for field in fields(self.sim.states):
            field_name = field.name
            if field_name in self.states_to_exclude_from_obs:
                continue
            _obs_size += math.prod(getattr(self.sim.states, field_name).shape[1:])
        self.single_observation_space = spaces.Box(
            -jnp.inf, jnp.inf, shape=(_obs_size,), dtype=jnp.float32
        )
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)

    def step(self, action: Array) -> Tuple[Array, Array, Array, Array, Dict]:
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        action = jnp.array(action, device=self.device).reshape(
            (self.sim.n_worlds, self.sim.n_drones, -1)
        )

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
        truncated = self.sim.steps >= self.max_episode_steps

        # Reset all environments which terminated or were truncated in the last step
        self.sim.reset(mask=self.prev_done)

        terminated = jnp.where(self.prev_done, False, terminated)
        truncated = jnp.where(self.prev_done, False, truncated)
        reward = self.reward

        self.prev_done = jnp.logical_or(terminated, truncated)

        return (
            self._get_obs(),
            reward,
            self._maybe_to_numpy(terminated),
            self._maybe_to_numpy(truncated),
            {},
        )

    # return self.obs, self.reward, self.terminated, self.truncated, self.info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Resets ALL (!) environments
        if options is None:
            options = {}
        self.sim.reset()

        self.prev_done = jnp.zeros((self.sim.n_worlds), dtype=jnp.bool_)
        return self._get_obs(), {}

    @property
    def reward(self):
        return self._reward(self.prev_done)

    @staticmethod
    @jax.jit
    def _reward(done: jax.Array )-> jnp.ndarray:
        return jnp.where(done, 0.0, 1.0)

    def render(self):
        self.sim.render()

    def _get_obs(self) -> Dict[str, jnp.ndarray]:
        # Returns observations for all environments
        obs = {
            field.name: self._maybe_to_numpy(getattr(self.sim.states, field.name))
            for field in fields(self.sim.states)
            if field.name not in self.states_to_exclude_from_obs
        }
        return obs

    def _get_terminated(self) -> jnp.ndarray:
        # Returns termination status for all environments
        return jnp.zeros((self.sim.n_worlds), dtype=jnp.bool_)

    def _maybe_to_numpy(self, data: Array) -> np.ndarray:
        if self.return_datatype == "numpy" and not isinstance(data, np.ndarray):
            return jax.device_get(data)
        return data
