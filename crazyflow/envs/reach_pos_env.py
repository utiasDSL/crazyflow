from functools import partial
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
from crazyflow.sim.structs import SimData
from crazyflow.utils import leaf_replace


class ReachPosEnv(DroneEnv):
    """Drone environment for reaching a goal position."""

    def __init__(
        self,
        pos_min: Array | None = None,
        pos_max: Array | None = None,
        vel_min: float = -1.0,
        vel_max: float = 1.0,
        num_envs: int = 1,
        max_episode_time: float = 10.0,
        physics: Literal["sys_id", "analytical"] | Physics = Physics.sys_id,
        freq: int = 500,
        device: str = "cpu",
    ):
        pos_min = jnp.array([-1.0, -1.0, 1.0]) if pos_min is None else pos_min
        pos_max = jnp.array([1.0, 1.0, 2.0]) if pos_max is None else pos_max
        reset_randomization = partial(
            self._reset_randomization, pmin=pos_min, pmax=pos_max, vmin=vel_min, vmax=vel_max
        )
        super().__init__(
            num_envs=num_envs,
            max_episode_time=max_episode_time,
            physics=physics,
            freq=freq,
            device=device,
            reset_randomization=reset_randomization,
        )
        self.jax_key = jax.device_put(
            jax.random.key(int(self.np_random.random() * 2**32)), self.device
        )
        spec = {k: v for k, v in self.single_observation_space.items()}
        spec["difference_to_goal"] = spaces.Box(-np.inf, np.inf, shape=(3,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)
        self._goal = jnp.zeros((self.sim.n_worlds, 3), dtype=jnp.float32, device=self.device)

    def _reset(self, mask: Array | None = None, options: dict | None = None):
        if options is None:
            options = {}
        self.jax_key, subkey = jax.random.split(self.jax_key)
        pmin, pmax = options.get("goal_pos_min"), options.get("goal_pos_max")
        self._goal = self._sample_goal(subkey, self._goal, pmin, pmax, mask)
        self.sim.reset(mask=mask)

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
        return self._reward(self.terminated(), self.sim.data.states.pos, self._goal)

    @staticmethod
    @jax.jit
    def _reward(terminated: Array, pos: Array, goal: Array) -> Array:
        norm_distance = jnp.linalg.norm(pos[:, 0, :] - goal, axis=-1)
        reward = jnp.exp(-2.0 * norm_distance)
        reward = jnp.where(terminated, -1.0, reward)
        return reward

    @staticmethod
    @jax.jit
    def _sample_goal(
        key: Array, goal: Array, pmin: Array | None, pmax: Array | None, mask: Array | None
    ) -> Array:
        pmin = jnp.array([-1.0, -1.0, 0.5]) if pmin is None else pmin
        pmax = jnp.array([1.0, 1.0, 1.5]) if pmax is None else pmax
        new_goal = jax.random.uniform(key, shape=goal.shape, minval=pmin, maxval=pmax)
        if mask is not None:
            new_goal = jnp.where(mask[..., None], new_goal, goal)
        return new_goal

    @staticmethod
    def _reset_randomization(
        data: SimData, mask: Array, pmin: Array, pmax: Array, vmin: float, vmax: float
    ) -> SimData:
        shape = (data.core.n_worlds, data.core.n_drones, 3)
        key, pos_key, vel_key = jax.random.split(data.core.rng_key, 3)
        data = data.replace(core=data.core.replace(rng_key=key))
        pos = jax.random.uniform(key=pos_key, shape=shape, minval=pmin, maxval=pmax)
        vel = jax.random.uniform(key=vel_key, shape=shape, minval=vmin, maxval=vmax)
        data = data.replace(states=leaf_replace(data.states, mask, pos=pos, vel=vel))
        return data
