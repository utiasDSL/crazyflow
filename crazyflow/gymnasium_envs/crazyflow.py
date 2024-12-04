import math
import warnings
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from jax import Array
from numpy.typing import NDArray

from crazyflow.control.controller import MAX_THRUST, MIN_THRUST, Control
from crazyflow.sim.core import Sim
from crazyflow.sim.structs import SimState


@dataclass
class RescaleParams:
    scale_factor: Array
    mean: Array


CONTROL_RESCALE_PARAMS = {
    "state": None,
    "thrust": None,
    "attitude": RescaleParams(
        scale_factor=jnp.array(
            [4 * (MAX_THRUST - MIN_THRUST) / 2, jnp.pi / 6, jnp.pi / 6, jnp.pi / 6]
        ),
        mean=jnp.array([4 * (MIN_THRUST + MAX_THRUST) / 2, 0.0, 0.0, 0.0]),
    ),
}


@partial(jax.jit, static_argnames=["convert"])
def maybe_to_numpy(data: Array, convert: bool) -> NDArray | Array:
    """Converts data to numpy array if convert is True."""
    return jax.lax.cond(convert, lambda: jax.device_get(data), lambda: data)


class CrazyflowBaseEnv(VectorEnv):
    """JAX Gymnasium environment for Crazyflie simulation."""

    def __init__(
        self,
        *,
        num_envs: int = 1,  # required for VectorEnv
        time_horizon_in_seconds: int = 10,
        return_datatype: Literal["numpy", "jax"] = "jax",
        **kwargs: dict,
    ):
        """Summary: Initializes the CrazyflowEnv.

        Args:
            num_envs: The number of environments to run in parallel.
            time_horizon_in_seconds: The time horizon after which episodes are truncated.
            return_datatype: The data type for returned arrays, either "numpy" or "jax". If "numpy",
                the returned arrays will be numpy arrays on the CPU. If "jax", the returned arrays
                will be jax arrays on the "device" specified for the simulation.
            **kwargs: Takes arguments that are passed to the Crazyfly simulation.
        """
        assert num_envs == kwargs["n_worlds"], "num_envs must be equal to n_worlds"

        # Set random initial seed for JAX. For seeding, people should use the reset function
        jax_seed = int(self.np_random.random() * 2**32)
        self.jax_key = jax.random.key(jax_seed)

        self.num_envs = num_envs
        self.return_datatype = return_datatype
        self.device = jax.devices(kwargs["device"])[0]
        self.time_horizon_in_seconds = jnp.array(
            time_horizon_in_seconds, dtype=jnp.int32, device=self.device
        )

        self.sim = Sim(**kwargs)

        assert (
            self.sim.freq >= self.sim.control_freq
        ), "Simulation frequency must be higher than control frequency"
        if not self.sim.freq % self.sim.control_freq == 0:
            warnings.warn(
                "Simulation frequency should be a multiple of control frequency. We can handle the other case, but we highly recommend to change the simulation frequency to a multiple of the control frequency."
            )

        self.n_substeps = self.sim.freq // self.sim.control_freq

        self.prev_done = jnp.zeros((self.sim.n_worlds), dtype=jnp.bool_, device=self.device)

        self.single_action_space = spaces.Box(
            -1,
            1,
            shape=(math.prod(getattr(self.sim.controls, self.sim.control).shape[1:]),),
            dtype=jnp.float32,
        )
        self.action_space = batch_space(self.single_action_space, self.sim.n_worlds)

        self.states_to_include_in_obs = ["pos", "quat", "vel", "ang_vel", "rpy_rates"]
        self._obs_size = 0
        for state in self.states_to_include_in_obs:
            if state == "pos":
                self._obs_size += math.prod(
                    getattr(self.sim.states, state)[0, :, 2].shape
                )  # exclude x-y coordinates
            else:
                self._obs_size += math.prod(getattr(self.sim.states, state).shape[1:])
        self.single_observation_space = spaces.Box(
            -jnp.inf, jnp.inf, shape=(self._obs_size,), dtype=jnp.float32
        )
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        action = self._sanitize_action(action, self.sim.n_worlds, self.sim.n_drones, self.device)
        action = self._rescale_action(action, self.sim.control)

        match self.sim.control:
            case Control.state:
                raise NotImplementedError(
                    "Possibly you want to control state differences instead of absolute states"
                )
            case Control.attitude:
                self.sim.attitude_control(action)
            case Control.thrust:
                self.sim.thrust_control(action)
            case _:
                raise ValueError(f"Invalid control type {self.sim.control}")

        for _ in range(self.n_substeps):
            self.sim.step()
        # Reset all environments which terminated or were truncated in the last step
        if jnp.any(self.prev_done):
            self.reset(mask=self.prev_done)

        terminated = self.terminated
        truncated = self.truncated
        self.prev_done = self._done(terminated, truncated)

        convert = self.return_datatype == "numpy"
        terminated = maybe_to_numpy(terminated, convert)
        truncated = maybe_to_numpy(truncated, convert)
        return self._obs(), self.reward, terminated, truncated, {}

    @staticmethod
    @partial(jax.jit, static_argnames=["n_worlds", "n_drones", "device"])
    def _sanitize_action(action: Array, n_worlds: int, n_drones: int, device: str) -> Array:
        return jnp.array(action, device=device).reshape((n_worlds, n_drones, -1))

    @staticmethod
    @partial(jax.jit, static_argnames=["control_type"])
    def _rescale_action(action: Array, control_type: str) -> Array:
        """Rescales actions based on the control type.

        Args:
            action (Array): Input actions to rescale.
            control_type (str): The type of control (`state`, `attitude`, or `thrust`).

        Returns:
            Array: Rescaled actions.
        """
        params = CONTROL_RESCALE_PARAMS.get(control_type)
        if params is None:
            raise NotImplementedError(
                f"Rescaling not implemented for control type '{control_type}'"
            )
        return action * params.scale_factor + params.mean

    @staticmethod
    @jax.jit
    def _done(terminated: Array, truncated: Array) -> Array:
        return jnp.logical_or(terminated, truncated)

    def reset_all(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Array], dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.jax_key = jax.random.key(seed)
        # Resets ALL (!) environments
        if options is None:
            options = {}

        self.reset(mask=jnp.ones((self.sim.n_worlds), dtype=jnp.bool_))

        self.prev_done = jnp.zeros((self.sim.n_worlds), dtype=jnp.bool_)

        return self._obs(), {}

    def reset(self, mask: Array) -> None:
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
        self.sim.states = self.sim.states.replace(
            pos=jnp.where(mask3d, init_pos, self.sim.states.pos)
        )

        # Sample initial vel
        self.jax_key, subkey = jax.random.split(self.jax_key)
        init_vel = jax.random.uniform(
            key=subkey, shape=(self.sim.n_worlds, self.sim.n_drones, 3), minval=-1.0, maxval=1.0
        )
        self.sim.states = self.sim.states.replace(
            vel=jnp.where(mask3d, init_vel, self.sim.states.vel)
        )

    @property
    def reward(self) -> Array:
        return self._reward(self.terminated, self.sim.states)

    @property
    def terminated(self) -> Array:
        return self._terminated(self.prev_done, self.sim.states, self.sim.contacts())

    @property
    def truncated(self) -> Array:
        return self._truncated(
            self.prev_done, self.sim.time, self.time_horizon_in_seconds, self.n_substeps
        )

    def _reward() -> None:
        raise NotImplementedError

    @staticmethod
    @jax.jit
    def _terminated(dones: Array, states: SimState, contacts: Array) -> Array:
        contact = jnp.any(contacts, axis=1)
        z_coords = states.pos[..., 2]
        below_ground = jnp.any(
            z_coords < -0.1, axis=1
        )  # Sanity check if we are below the ground. Should not be triggered due to collision checking
        terminated = jnp.logical_or(below_ground, contact)
        return jnp.where(dones, False, terminated)

    @staticmethod
    @jax.jit
    def _truncated(
        dones: Array, time: Array, time_horizon_in_seconds: Array, n_substeps: Array
    ) -> Array:
        truncated = time >= time_horizon_in_seconds
        return jnp.where(dones, False, truncated)

    def render(self):
        self.sim.render()

    def _obs(self) -> dict[str, Array]:
        convert = self.return_datatype == "numpy"
        fields = self.states_to_include_in_obs
        states = [maybe_to_numpy(getattr(self.sim.states, field), convert) for field in fields]
        obs = {k: v for k, v in zip(fields, states)}
        if "pos" in obs:
            obs["pos"] = obs["pos"][..., 2]
        return obs


class CrazyflowEnvReachGoal(CrazyflowBaseEnv):
    """JAX Gymnasium environment for Crazyflie simulation."""

    def __init__(self, **kwargs: dict):
        assert kwargs["n_drones"] == 1, "Currently only supported for one drone"

        super().__init__(**kwargs)
        self._obs_size += 3  # difference to goal position
        self.single_observation_space = spaces.Box(
            -jnp.inf, jnp.inf, shape=(self._obs_size,), dtype=jnp.float32
        )
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)
        self.goal = jnp.zeros((kwargs["n_worlds"], 3), dtype=jnp.float32, device=self.device)

    @property
    def reward(self) -> Array:
        return self._reward(self.terminated, self.sim.states, self.goal)

    @staticmethod
    @jax.jit
    def _reward(terminated: Array, states: SimState, goal: Array) -> Array:
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

    def _obs(self) -> dict[str, Array]:
        obs = super()._obs()
        obs["difference_to_goal"] = [self.goal - self.sim.states.pos]
        return obs


class CrazyflowEnvTargetVelocity(CrazyflowBaseEnv):
    """JAX Gymnasium environment for Crazyflie simulation."""

    def __init__(self, **kwargs: dict):
        assert kwargs["n_drones"] == 1, "Currently only supported for one drone"

        super().__init__(**kwargs)
        self._obs_size += 3  # difference to target velocity
        self.single_observation_space = spaces.Box(
            -jnp.inf, jnp.inf, shape=(self._obs_size,), dtype=jnp.float32
        )
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)

        self.target_vel = jnp.zeros((kwargs["n_worlds"], 3), dtype=jnp.float32)

    @property
    def reward(self) -> Array:
        return self._reward(self.terminated, self.sim.states, self.target_vel)

    @staticmethod
    @jax.jit
    def _reward(terminated: Array, states: SimState, target_vel: Array) -> Array:
        norm_distance = jnp.linalg.norm(states.vel - target_vel, axis=2)
        reward = jnp.exp(-norm_distance)
        return jnp.where(terminated, -1.0, reward)

    def reset(self, mask: Array) -> None:
        super().reset(mask)

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
        obs["difference_to_target_vel"] = [self.target_vel - self.sim.states.vel]
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
        return self._reward(self.terminated, self.sim.states, self.goal)

    @staticmethod
    @jax.jit
    def _reward(terminated: Array, states: SimState, goal: Array) -> Array:
        norm_distance = jnp.linalg.norm(states.pos - goal, axis=2)
        speed = jnp.linalg.norm(states.vel, axis=2)
        reward = jnp.exp(-2.0 * norm_distance) * jnp.exp(-2.0 * speed)
        return jnp.where(terminated, -1.0, reward)

    def reset(self, mask: Array) -> None:
        super().reset(mask)

    def _get_obs(self) -> dict[str, Array]:
        obs = super()._get_obs()
        obs["difference_to_goal"] = [self.goal - self.sim.states.pos]
        return obs
