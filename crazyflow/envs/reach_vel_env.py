from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax import Array

from crazyflow.envs.drone_env import DroneEnv
from crazyflow.sim.physics import Physics


class ReachVelEnv(DroneEnv):
    """Drone environment for reaching a goal velocity."""

    def __init__(
        self,
        *,
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
        self.jax_key = jax.device_put(
            jax.random.key(int(self.np_random.random() * 2**32)), self.device
        )
        assert self.sim.n_drones == 1, "Currently only supported for one drone"
        spec = {k: v for k, v in self.single_observation_space.items()}
        spec["difference_to_target_vel"] = spaces.Box(-np.inf, np.inf, shape=(3,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)
        self._goal_vel = jnp.zeros((self.sim.n_worlds, 3), dtype=jnp.float32, device=self.device)

    def obs(self) -> dict[str, Array]:
        obs = super().obs()
        # drop n_drones dimension, as it is always 1 for now
        obs["difference_to_target_vel"] = self._goal_vel - self.sim.data.states.vel[:, 0, :]
        return obs

    def reward(self) -> Array:
        return self._reward(self.terminated(), self.sim.data.states.vel, self._goal_vel)

    @staticmethod
    @jax.jit
    def _reward(terminated: Array, vel: Array, target_vel: Array) -> Array:
        norm_distance = jnp.linalg.norm(vel[:, 0, :] - target_vel, axis=-1)
        reward = jnp.exp(-norm_distance)
        reward = jnp.where(terminated, -1.0, reward)
        return reward

    def _reset(self, mask: Array | None = None, options: dict | None = None) -> None:
        if options is None:
            options = {}
        self.jax_key, subkey = jax.random.split(self.jax_key)
        vmin, vmax = options.get("vel_min"), options.get("vel_max")
        self._goal_vel = self._sample_goal_vel(subkey, self._goal_vel, vmin, vmax, mask)
        self.sim.reset(mask=mask)

    @staticmethod
    @jax.jit
    def _sample_goal_vel(
        key: Array, goal_vel: Array, vmin: Array | None, vmax: Array | None, mask: Array | None
    ) -> Array:
        vmin = jnp.array([-1.0, -1.0, -1.0]) if vmin is None else vmin
        vmax = jnp.array([1.0, 1.0, 1.0]) if vmax is None else vmax
        new_goal_vel = jax.random.uniform(key, shape=goal_vel.shape, minval=vmin, maxval=vmax)
        if mask is not None:
            new_goal_vel = jnp.where(mask[..., None], new_goal_vel, goal_vel)
        return new_goal_vel
