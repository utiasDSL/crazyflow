from typing import Literal

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax import Array

from crazyflow.envs.drone_env import DroneEnv
from crazyflow.sim.physics import Physics
from crazyflow.sim.structs import SimState


class LandingEnv(DroneEnv):
    """Drone environment for landing at a target position."""

    def __init__(
        self,
        num_envs: int = 1,
        max_episode_time: float = 10.0,
        physics: Literal["sys_id", "analytical"] | Physics = Physics.sys_id,
        freq: int = 500,
        device: str = "cpu",
    ):
        super().__init__(
            num_envs=num_envs,
            max_episode_time=max_episode_time,
            physics=physics,
            freq=freq,
            device=device,
        )
        spec = {k: v for k, v in self.single_observation_space.items()}
        spec["difference_to_goal"] = spaces.Box(-np.inf, np.inf, shape=(3,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)
        self._goal = jnp.zeros((self.sim.n_worlds, 3), device=self.device)
        self._goal = self._goal.at[..., 2].set(0.1)  # 10cm above ground

    def render(self):
        if self.sim.viewer is not None:
            self.sim.viewer.viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.02, 0.02, 0.02]),
                pos=np.array(self._goal[0]),
                rgba=np.array([1, 0, 0, 0.5]),
            )
        self.sim.render()

    def obs(self) -> dict[str, Array]:
        obs = super().obs()
        # drop n_drones dimension, as it is always 1 for now
        obs["difference_to_goal"] = self._goal - self.sim.data.states.pos[:, 0, :]
        return obs

    def reward(self) -> Array:
        return self._reward(self.terminated(), self.sim.data.states, self._goal)

    @staticmethod
    @jax.jit
    def _reward(terminated: Array, states: SimState, goal: Array) -> Array:
        norm_distance = jnp.linalg.norm(states.pos[:, 0, :] - goal, axis=-1)
        speed = jnp.linalg.norm(states.vel[:, 0, :], axis=-1)
        reward = jnp.exp(-2.0 * norm_distance) * jnp.exp(-2.0 * speed)
        reward = jnp.where(terminated, -1.0, reward)
        return reward
