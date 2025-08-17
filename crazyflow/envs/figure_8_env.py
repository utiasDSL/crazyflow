from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax import Array

from crazyflow.envs.drone_env import DroneEnv
from crazyflow.sim.physics import Physics
from crazyflow.sim.structs import SimData
from crazyflow.sim.visualize import draw_line, draw_points
from crazyflow.utils import leaf_replace


class FigureEightEnv(DroneEnv):
    """Drone environment for following a figure-eight trajectory.

    This environment is used to follow a figure-eight trajectory. The observations contain the
    relative position errors to the next `n_samples` points that are distanced by `samples_dt`. The
    reward is based on the distance to the next trajectory point.
    """

    def __init__(
        self,
        n_samples: int = 10,
        samples_dt: float = 0.1,
        trajectory_time: float = 10.0,
        *,
        num_envs: int = 1,
        max_episode_time: float = 10.0,
        physics: Literal["sys_id", "analytical"] | Physics = Physics.sys_id,
        freq: int = 500,
        device: str = "cpu",
    ):
        """Initialize the environment and create the figure-eight trajectory.

        Args:
            n_samples: Number of next trajectory points to sample for observations.
            samples_dt: Time between trajectory sample points in seconds.
            trajectory_time: Total time for completing the figure-eight trajectory in seconds.
            num_envs: Number of environments to run in parallel.
            max_episode_time: Maximum episode time in seconds.
            physics: Physics backend to use.
            freq: Frequency of the simulation.
            device: Device to use for the simulation.
        """
        super().__init__(
            num_envs=num_envs,
            max_episode_time=max_episode_time,
            physics=physics,
            freq=freq,
            device=device,
        )
        if trajectory_time < self.max_episode_time:
            raise ValueError("Trajectory time must be greater than max episode time")

        # Create the figure eight trajectory
        n_steps = int(np.ceil(trajectory_time * self.freq))
        t = np.linspace(0, 2 * np.pi, n_steps)
        radius = 1  # Radius for the circles
        y = np.zeros_like(t)  # x is 0 everywhere
        x = radius * np.sin(t)  # Scale amplitude for 1-meter diameter
        z = radius * np.sin(2 * t) + 1.2  # Scale amplitude for 1-meter diameter
        self.trajectory = np.array([x, y, z]).T

        # Define trajectory sampling parameters
        self.n_samples = n_samples
        self.samples_dt = samples_dt
        self.sample_offsets = np.array(np.arange(n_samples) * self.freq * samples_dt, dtype=int)

        # Update observation space
        spec = {k: v for k, v in self.single_observation_space.items()}
        spec["local_samples"] = spaces.Box(-np.inf, np.inf, shape=(3 * self.n_samples,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)

    def render(self):
        idx = (self.steps + self.sample_offsets[None, ...]) % self.trajectory.shape[0]
        next_trajectory = self.trajectory[idx, ...]
        draw_line(self.sim, next_trajectory[0], start_size=3.0, end_size=3.0)
        draw_points(self.sim, next_trajectory[0], rgba=np.array([1.0, 0, 0, 1]), size=0.01)
        self.sim.render()

    def obs(self) -> dict[str, Array]:
        obs = super().obs()
        idx = (self.steps + self.sample_offsets[None, ...]) % self.trajectory.shape[0]
        dpos = self.trajectory[idx, ...] - self.sim.data.states.pos
        obs["local_samples"] = dpos.reshape(-1, 3 * self.n_samples)
        return obs

    def reward(self) -> Array:
        return self._reward(
            self.terminated(), self.sim.data.states.pos, self.trajectory[self.steps.squeeze()]
        )

    @property
    def steps(self) -> Array:
        """The current step in the trajectory."""
        return self.sim.data.core.steps // (self.sim.freq // self.freq) - 1

    @staticmethod
    @jax.jit
    def _reward(terminated: Array, pos: Array, goal: Array) -> Array:
        # distance to next trajectory point
        norm_distance = jnp.linalg.norm(pos[:, 0, :] - goal, axis=-1)
        reward = jnp.exp(-2.0 * norm_distance)
        reward = jnp.where(terminated, -1.0, reward)
        return reward

    @staticmethod
    def _reset_randomization(data: SimData, mask: Array) -> SimData:
        """Randomize the initial position and velocity of the drones.

        This function will get compiled into the reset function of the simulation. Therefore, it
        must take data and mask as input arguments and must return a SimData object.
        """
        # Sample initial position
        shape = (data.core.n_worlds, data.core.n_drones, 3)
        pmin, pmax = jnp.array([-0.1, -0.1, 1.1]), jnp.array([0.1, 0.1, 1.3])
        key, pos_key, vel_key = jax.random.split(data.core.rng_key, 3)
        data = data.replace(core=data.core.replace(rng_key=key))
        pos = jax.random.uniform(key=pos_key, shape=shape, minval=pmin, maxval=pmax)
        vel = jax.random.uniform(key=vel_key, shape=shape, minval=-0.5, maxval=0.5)
        data = data.replace(states=leaf_replace(data.states, mask, pos=pos, vel=vel))
        return data
