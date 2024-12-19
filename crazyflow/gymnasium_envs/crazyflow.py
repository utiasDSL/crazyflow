import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv, VectorWrapper
from gymnasium.vector.utils import batch_space
from jax import Array

from crazyflow.control.control import MAX_THRUST, MIN_THRUST, Control
from crazyflow.sim import Sim
from crazyflow.sim.structs import SimState


def action_space(control_type: Control) -> spaces.Box:
    """Select the appropriate action space for a given control type.

    Args:
        control_type: The desired control mode.

    Returns:
        The action space.
    """
    match control_type:
        case Control.attitude:
            return spaces.Box(
                np.array([4 * MIN_THRUST, -np.pi, -np.pi, -np.pi], dtype=np.float32),
                np.array([4 * MAX_THRUST, np.pi, np.pi, np.pi], dtype=np.float32),
            )
        case Control.thrust:
            return spaces.Box(MIN_THRUST, MAX_THRUST, shape=(4,))
        case _:
            raise ValueError(f"Invalid control type {control_type}")


class CrazyflowBaseEnv(VectorEnv):
    """JAX Gymnasium environment for Crazyflie simulation.

    ## Action space
    We have three types of actions:
    - `attitude`: 4D vector consisting of [collective thrust, roll, pitch, yaw]
    - `thrust`: 4D vector consisting of the individual motor thrusts [f1, f2, f3, f4]
    - `state`: Currently not implemented

    The default action space is `attitude`.
    """

    obs_keys = ["pos", "quat", "vel", "rpy_rates"]

    def __init__(
        self,
        *,
        num_envs: int = 1,  # required for VectorEnv
        time_horizon_in_seconds: float = 10.0,
        **kwargs: dict,
    ):
        """Initialize the CrazyflowEnv.

        Args:
            num_envs: The number of environments to run in parallel.
            time_horizon_in_seconds: The time horizon after which episodes are truncated.
            **kwargs: Takes arguments that are passed to the Crazyfly simulation.
        """
        assert num_envs == kwargs["n_worlds"], "num_envs must be equal to n_worlds"
        # Set random initial seed for JAX. For seeding, people should use the reset function
        self.jax_key = jax.random.key(int(self.np_random.random() * 2**32))

        self.num_envs = num_envs
        self.device = jax.devices(kwargs["device"])[0]
        self.time_horizon_in_seconds = time_horizon_in_seconds
        self.sim = Sim(**kwargs)

        assert self.sim.freq >= self.sim.control_freq, "Sim freq must be higher than control freq"
        if not self.sim.freq % self.sim.control_freq == 0:
            # We can handle other cases, but it's not recommended
            warnings.warn("Simulation frequency should be a multiple of control frequency.")
        if self.sim.control == Control.state:
            raise NotImplementedError("State control currently not supported")

        self.n_substeps = self.sim.freq // self.sim.control_freq
        self.prev_done = jnp.zeros((self.sim.n_worlds), dtype=jnp.bool_, device=self.device)

        self.single_action_space = action_space(self.sim.control)
        self.action_space = batch_space(self.single_action_space, self.sim.n_worlds)

        self.single_observation_space = spaces.Dict(
            {
                "pos": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "quat": spaces.Box(-np.inf, np.inf, shape=(4,)),
                "vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "rpy_rates": spaces.Box(-np.inf, np.inf, shape=(3,)),
            }
        )
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        assert self.action_space.contains(np.array(action)), f"{action!r} ({type(action)}) invalid"
        action = self._sanitize_action(action, self.sim.n_worlds, self.sim.n_drones, self.device)
        self._apply_action(action)

        self.sim.step(self.n_substeps)
        # Reset all environments which terminated or were truncated in the last step
        if jnp.any(self.prev_done):
            self.reset_masked(mask=self.prev_done)

        terminated = self.terminated
        truncated = self.truncated
        # We need to calculate the reward before setting `self.prev_done`, because reward depends on
        # prev_done. Moving this line to after the write to prev_done results in wrong rewards.
        reward = self.reward
        self.prev_done = self._done(terminated, truncated)

        terminated = terminated
        truncated = truncated
        return self._obs(), reward, terminated, truncated, {}

    def _apply_action(self, action: Array):
        match self.sim.control:
            case Control.state:
                raise NotImplementedError("State control currently not supported")
            case Control.attitude:
                self.sim.attitude_control(action)
            case Control.thrust:
                self.sim.thrust_control(action)
            case _:
                raise ValueError(f"Invalid control type {self.sim.control}")

    @staticmethod
    @partial(jax.jit, static_argnames=["n_worlds", "n_drones", "device"])
    def _sanitize_action(action: Array, n_worlds: int, n_drones: int, device: str) -> Array:
        return jnp.array(action, device=device).reshape((n_worlds, n_drones, -1))

    @staticmethod
    @jax.jit
    def _done(terminated: Array, truncated: Array) -> Array:
        return jnp.logical_or(terminated, truncated)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Array], dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.jax_key = jax.random.key(seed)

        self.reset_masked(mask=jnp.ones((self.sim.n_worlds), dtype=bool, device=self.device))
        self.prev_done = jnp.zeros((self.sim.n_worlds), dtype=bool, device=self.device)
        return self._obs(), {}

    def reset_masked(self, mask: Array) -> None:
        self.sim.reset(mask=mask)
        mask3d = mask[:, None, None]
        # NOTE Setting initial ryp_rate when using physics.sys_id will not have an impact
        # Sample initial pos
        self.jax_key, subkey = jax.random.split(self.jax_key)
        init_pos = jax.random.uniform(
            key=subkey,
            shape=(self.sim.n_worlds, self.sim.n_drones, 3),
            minval=jnp.array([-1.0, -1.0, 1.0]),  # x,y,z
            maxval=jnp.array([1.0, 1.0, 2.0]),  # x,y,z
        )
        self.sim.data = self.sim.data.replace(
            states=self.sim.data.states.replace(
                pos=jnp.where(mask3d, init_pos, self.sim.data.states.pos)
            )
        )
        # Sample initial vel
        self.jax_key, subkey = jax.random.split(self.jax_key)
        init_vel = jax.random.uniform(
            key=subkey, shape=(self.sim.n_worlds, self.sim.n_drones, 3), minval=-1.0, maxval=1.0
        )
        self.sim.data = self.sim.data.replace(
            states=self.sim.data.states.replace(
                vel=jnp.where(mask3d, init_vel, self.sim.data.states.vel)
            )
        )

    @property
    def reward(self) -> Array:
        return self._reward(self.prev_done, self.terminated, self.sim.data.states)

    @property
    def terminated(self) -> Array:
        return self._terminated(self.prev_done, self.sim.data.states, self.sim.contacts())

    @property
    def truncated(self) -> Array:
        return self._truncated(self.prev_done, self.sim.time, self.time_horizon_in_seconds)

    def _reward() -> None:
        raise NotImplementedError

    @staticmethod
    @jax.jit
    def _terminated(dones: Array, states: SimState, contacts: Array) -> Array:
        contact = jnp.any(contacts, axis=1)
        z_coords = states.pos[..., 2]
        # Sanity check if we are below the ground. Should not be triggered due to collision checking
        below_ground = jnp.any(z_coords < -0.1, axis=1)
        terminated = jnp.logical_or(below_ground, contact)
        return jnp.where(dones, False, terminated)

    @staticmethod
    @jax.jit
    def _truncated(dones: Array, time: Array, time_horizon_in_seconds: float) -> Array:
        truncated = (time >= time_horizon_in_seconds).squeeze()
        return jnp.where(dones, False, truncated)

    def render(self):
        self.sim.render()

    def _obs(self) -> dict[str, Array]:
        fields = self.obs_keys
        states = [getattr(self.sim.data.states, field) for field in fields]
        return {k: v for k, v in zip(fields, states)}


class CrazyflowEnvReachGoal(CrazyflowBaseEnv):
    """JAX Gymnasium environment for Crazyflie simulation."""

    def __init__(self, **kwargs: dict):
        assert kwargs["n_drones"] == 1, "Currently only supported for one drone"

        super().__init__(**kwargs)
        spec = {k: v for k, v in self.single_observation_space.items()}
        spec["difference_to_goal"] = spaces.Box(-np.inf, np.inf, shape=(3,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)
        self.goal = jnp.zeros((self.sim.n_worlds, 3), dtype=jnp.float32, device=self.device)

    @property
    def reward(self) -> Array:
        return self._reward(self.prev_done, self.terminated, self.sim.data.states, self.goal)

    @staticmethod
    @jax.jit
    def _reward(prev_done: Array, terminated: Array, states: SimState, goal: Array) -> Array:
        norm_distance = jnp.linalg.norm(states.pos - goal, axis=2)
        reward = jnp.exp(-2.0 * norm_distance)
        reward = jnp.where(terminated, -1.0, reward)
        reward = jnp.where(prev_done, 0.0, reward)
        return reward

    def reset_masked(self, mask: Array) -> None:
        super().reset_masked(mask)

        # Generate new goals
        self.jax_key, subkey = jax.random.split(self.jax_key)
        new_goals = jax.random.uniform(
            key=subkey,
            shape=(self.sim.n_worlds, 3),
            minval=jnp.array([-1.0, -1.0, 0.5]),  # x,y,z
            maxval=jnp.array([1.0, 1.0, 1.5]),  # x,y,z
        )
        self.goal = self.goal.at[mask].set(new_goals[mask])

    def _obs(self) -> dict[str, Array]:
        obs = super()._obs()
        obs["difference_to_goal"] = [self.goal - self.sim.data.states.pos]
        return obs


class CrazyflowEnvTargetVelocity(CrazyflowBaseEnv):
    """JAX Gymnasium environment for Crazyflie simulation."""

    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        assert self.sim.n_drones == 1, "Currently only supported for one drone"
        spec = {k: v for k, v in self.single_observation_space.items()}
        spec["difference_to_target_vel"] = spaces.Box(-np.inf, np.inf, shape=(3,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)
        self.target_vel = jnp.zeros((self.sim.n_worlds, 3), dtype=jnp.float32)

    @property
    def reward(self) -> Array:
        return self._reward(self.prev_done, self.terminated, self.sim.data.states, self.target_vel)

    @staticmethod
    @jax.jit
    def _reward(prev_done: Array, terminated: Array, states: SimState, target_vel: Array) -> Array:
        norm_distance = jnp.linalg.norm(states.vel - target_vel, axis=2)
        reward = jnp.exp(-norm_distance)
        reward = jnp.where(terminated, -1.0, reward)
        reward = jnp.where(prev_done, 0.0, reward)
        return reward

    def reset_masked(self, mask: Array) -> None:
        super().reset_masked(mask)

        # Generate new target_vels
        self.jax_key, subkey = jax.random.split(self.jax_key)
        new_target_vel = jax.random.uniform(
            key=subkey,
            shape=(self.sim.n_worlds, 3),
            minval=jnp.array([-1.0, -1.0, -1.0]),  # x,y,z
            maxval=jnp.array([1.0, 1.0, 1.0]),  # x,y,z
        )
        self.target_vel = self.target_vel.at[mask].set(new_target_vel[mask])

    def _obs(self) -> dict[str, Array]:
        obs = super()._obs()
        obs["difference_to_target_vel"] = [self.target_vel - self.sim.data.states.vel]
        return obs


class CrazyflowEnvLanding(CrazyflowBaseEnv):
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
        self.goal = self.goal.at[..., 2].set(0.1)  # 10cm above ground

    @property
    def reward(self) -> Array:
        return self._reward(self.prev_done, self.terminated, self.sim.data.states, self.goal)

    @staticmethod
    @jax.jit
    def _reward(prev_done: Array, terminated: Array, states: SimState, goal: Array) -> Array:
        norm_distance = jnp.linalg.norm(states.pos - goal, axis=2)
        speed = jnp.linalg.norm(states.vel, axis=2)
        reward = jnp.exp(-2.0 * norm_distance) * jnp.exp(-2.0 * speed)
        reward = jnp.where(terminated, -1.0, reward)
        reward = jnp.where(prev_done, 0.0, reward)
        return reward

    def reset_masked(self, mask: Array) -> None:
        super().reset_masked(mask)

    def _get_obs(self) -> dict[str, Array]:
        obs = super()._get_obs()
        obs["difference_to_goal"] = [self.goal - self.sim.data.states.pos]
        return obs


class CrazyflowRL(VectorWrapper):
    """Wrapper to use the crazyflow JAX environments with common DRL frameworks.
    Currently, this wrapper clips the expected actions to [-1,1]
    and rescales them to the action space expected in simulation.
    """

    def __init__(self, env: VectorEnv):
        super().__init__(env)

        # Simulation action space bounds
        self.action_sim_low = self.single_action_space.low
        self.action_sim_high = self.single_action_space.high

        # Compute scale and mean for rescaling
        self.action_scale = jnp.array((self.action_sim_high - self.action_sim_low) / 2.0)
        self.action_mean = jnp.array((self.action_sim_high + self.action_sim_low) / 2.0)

        # Modify the wrapper's action space to [-1, 1]
        self.single_action_space.low = -np.ones_like(self.action_sim_low)
        self.single_action_space.high = np.ones_like(self.action_sim_high)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def step(self, actions: Array) -> tuple[Array, Array, Array, Array, dict]:
        actions = np.clip(actions, -1.0, 1.0)
        return self.env.step(self.actions(actions))

    def actions(self, actions):
        """Rescale and clip actions from [-1, 1] to [action_sim_low, action_sim_high]."""
        # Rescale actions using the computed scale and mean
        rescaled_actions = actions * self.action_scale + self.action_mean

        # Ensure actions are within the valid range of the simulation action space
        rescaled_actions = np.clip(rescaled_actions, self.action_sim_low, self.action_sim_high)

        return rescaled_actions
