import math
from dataclasses import fields
from functools import partial
from typing import Dict, Literal, Optional, Tuple

import jax
from jax.scipy.spatial.transform import Rotation
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from jax import Array

from crazyflow.control.controller import Control
from crazyflow.sim.core import Sim
from crazyflow.sim.structs import SimState


class CrazyflowVectorBaseEnv(VectorEnv):
    """JAX Gymnasium environment for Crazyflie simulation."""

    def __init__(
        self,
        *,
        jax_random_key, # required for jax random number generator
        num_envs: int = 1, # required for VectorEnv
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

        self.jax_key = jax.random.key(jax_random_key)

        self.num_envs = num_envs
        self.return_datatype = return_datatype
        self.device = jax.devices(kwargs["device"])[0]
        self.max_episode_steps = jnp.array(max_episode_steps, dtype=jnp.int32, device=self.device)

        self.sim = Sim(**kwargs)

        self.prev_done = jnp.zeros((self.sim.n_worlds), dtype=jnp.bool_, device=self.device)

        self.single_action_space = spaces.Box(
            -1,
            1,
            shape=(math.prod(getattr(self.sim.controls, self.sim.control).shape[1:]),),
            dtype=jnp.float32,
        )
        self.action_space = batch_space(self.single_action_space, self.sim.n_worlds)

        self.states_to_exclude_from_obs = ["step", "device", "pos"]  # exclude pos from observation
        self._obs_size = 0
        for field in fields(self.sim.states):
            if field.name in self.states_to_exclude_from_obs:
                continue
            self._obs_size += math.prod(getattr(self.sim.states, field.name).shape[1:])
        self.single_observation_space = spaces.Box(
            -jnp.inf, jnp.inf, shape=(self._obs_size,), dtype=jnp.float32
        )
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)

    def step(self, action: Array) -> Tuple[Array, Array, Array, Array, Dict]:
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        action = jnp.array(action, device=self.device).reshape(
            (self.sim.n_worlds, self.sim.n_drones, -1)
        )

        action = self._rescale_action(action, self.sim.control)

        if self.sim.control == Control.state:
            raise NotImplementedError("State control not implemented. Should use relative states for control.")
            # self.sim.state_control(action)
        elif self.sim.control == Control.attitude:
            self.sim.attitude_control(action)
        elif self.sim.control == Control.thrust:
            self.sim.thrust_control(action)
        else:
            raise ValueError(f"Invalid control type {self.sim.control}")

        self.sim.step()

        # Reset all environments which terminated or were truncated in the last step
        if jnp.any(self.prev_done):
            self.reset(mask=self.prev_done)

        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated

        self.prev_done = jnp.logical_or(terminated, truncated)

        return (
            self._get_obs(),
            reward,
            self._maybe_to_numpy(terminated),
            self._maybe_to_numpy(truncated),
            {},
        )

    @staticmethod
    @partial(
        jax.jit, static_argnames=["control_type"]
    )  # static_argnames assumes they dont change during runtime
    def _rescale_action(action: Array, control_type: str) -> Array:
        """Rescales actions based on the control type.

        Args:
            action (Array): Input actions to rescale.
            control_type (str): The type of control (`state`, `attitude`, or `thrust`).

        Returns:
            Array: Rescaled actions.
        """

        def scale_state(action):
            scale_factor = jnp.ones(13)  # Example scaling for state control
            return action * scale_factor

        def scale_attitude(action):
            scale_factor = jnp.ones(4)  # Example scaling for attitude control
            return action * scale_factor

        def scale_thrust(action):
            scale_factor = jnp.ones(4)  # Example scaling for thrust control
            return action * scale_factor

        # Use Python if-statements (works with static_argnames)
        if control_type == "state":
            return scale_state(action)
        elif control_type == "attitude":
            return scale_attitude(action)
        elif control_type == "thrust":
            return scale_thrust(action)
        else:
            raise ValueError(f"Unsupported control type: {control_type}")

    def reset_all(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Resets ALL (!) environments
        if options is None:
            options = {}

        self.reset(mask=jnp.ones((self.sim.n_worlds), dtype=jnp.bool_))

        self.prev_done = jnp.zeros((self.sim.n_worlds), dtype=jnp.bool_)

        return self._get_obs(), {}

    def reset(self, mask: Array) -> None:
        self.sim.reset(mask=mask)

        mask3d = mask[:, None, None]

        # Sample initial pos
        self.jax_key, subkey = jax.random.split(self.jax_key)
        init_pos = jax.random.uniform(
            key=subkey,
            shape=(self.sim.n_worlds, self.sim.n_drones, 3),
            minval=jnp.array([-1.0, -1.0, 0.5]),  # x,y,z
            maxval=jnp.array([1.0, 1.0, 1.5]),  # x,y,z
        )
        self.sim.states = self.sim.states.replace(
            pos=jnp.where(mask3d, init_pos, self.sim.states.pos)
        )

        # Sample initial vel
        self.jax_key, subkey = jax.random.split(self.jax_key)
        init_vel = jax.random.uniform(
            key=subkey, shape=(self.sim.n_worlds, self.sim.n_drones, 3), minval=-0.1, maxval=0.1
        )
        self.sim.states = self.sim.states.replace(
            vel=jnp.where(mask3d, init_vel, self.sim.states.vel)
        )

        # Setting initial ryp_rate when using 

    @property
    def reward(self):
        return self._reward(self.terminated, self.sim.states)

    @property
    def terminated(self):
        return self._terminated(self.prev_done, self.sim.states)

    @property
    def truncated(self):
        return self._truncated(self.prev_done, self.sim.steps, self.max_episode_steps)

    @staticmethod
    @jax.jit
    def _reward(terminated: jax.Array, states: SimState) -> jnp.ndarray:
        z_coords = states.pos[..., 2]
        reward = jnp.sum(z_coords, axis=1)
        return jnp.where(terminated, -1.0, reward)

    @staticmethod
    @jax.jit
    def _terminated(dones: jax.Array, states: SimState) -> jnp.ndarray:
        # terminate if any drone goes below -0.1 in z
        z_coords = states.pos[..., 2]
        terminated = jnp.any(z_coords < -0.1, axis=1)
        return jnp.where(dones, False, terminated)

    @staticmethod
    @jax.jit
    def _truncated(dones: jax.Array, steps: jax.Array, max_episode_steps: jax.Array) -> jnp.ndarray:
        truncated = steps >= max_episode_steps
        return jnp.where(dones, False, truncated)

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

    def _maybe_to_numpy(self, data: Array) -> np.ndarray:
        if self.return_datatype == "numpy" and not isinstance(data, np.ndarray):
            return jax.device_get(data)
        return data


class CrazyflowVectorEnvReachGoal(CrazyflowVectorBaseEnv):
    """JAX Gymnasium environment for Crazyflie simulation."""

    def __init__(self, **kwargs: dict):
        assert kwargs["n_drones"] == 1, "Currently only supported for one drone"

        super().__init__(**kwargs)
        self._obs_size += 3  # difference to goal position
        self.single_observation_space = spaces.Box(
            -jnp.inf, jnp.inf, shape=(self._obs_size,), dtype=jnp.float32
        )
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)

        self.goal = jnp.zeros((kwargs["n_worlds"], 3), dtype=jnp.float32)

    @property
    def reward(self):
        return self._reward(self.terminated, self.sim.states, self.goal)

    @staticmethod
    @jax.jit
    def _reward(terminated: jax.Array, states: SimState, goal: jax.Array) -> jnp.ndarray:
        norm_distance = jnp.linalg.norm(states.pos - goal, axis=2)
        reward = jnp.exp(-2.0 * norm_distance)
        return jnp.where(terminated, -1.0, reward)

    def reset(self, mask: Array) -> None:
        super().reset(mask)

        # Generate new goals
        self.jax_key, subkey = jax.random.split(self.jax_key)
        new_goals = jax.random.uniform(
            key=subkey,
            shape=(self.sim.n_worlds, 3),
            minval=jnp.array([-1.0, -1.0, 0.5]),  # x,y,z
            maxval=jnp.array([1.0, 1.0, 1.5]),  # x,y,z
        )
        self.goal = self.goal.at[mask].set(new_goals[mask])

    def _get_obs(self) -> Dict[str, jnp.ndarray]:
        obs = super()._get_obs()
        obs["difference_to_goal"] = [self.goal - self.sim.states.pos]
        return obs
