import warnings
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
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
                np.array([4 * MIN_THRUST, -np.pi / 2, -np.pi / 2, -np.pi / 2], dtype=np.float32),
                np.array([4 * MAX_THRUST, np.pi / 2, np.pi / 2, np.pi / 2], dtype=np.float32),
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

    obs_keys = ["pos", "quat", "vel", "ang_vel"]
    # TODO: Once we switch to gymnasium >= 1.1.0, we should set the autoreset mode
    # metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP}

    def __init__(
        self,
        *,
        num_envs: int = 1,  # required for VectorEnv
        time_horizon_in_seconds: float = 10.0,
        physics: Literal["sys_id", "analytical"] = "sys_id",
        freq: int = 500,
        device: str = "cpu",
    ):
        """Initialize the CrazyflowEnv.

        Args:
            num_envs: The number of environments to run in parallel.
            time_horizon_in_seconds: The time horizon after which episodes are truncated.
            physics: The crazyflow physics simulation model.
            freq: The frequency at which the environment is run.
            device: The device of the environment and the simulation.
        """
        self.num_envs = num_envs
        self.device = jax.devices(device)[0]
        # Set random initial seed for JAX. For seeding, people should use the reset function
        self.jax_key = jax.device_put(
            jax.random.key(int(self.np_random.random() * 2**32)), self.device
        )

        self.time_horizon_in_seconds = time_horizon_in_seconds
        assert physics in ("sys_id", "analytical"), "Invalid physics type"
        self.sim = Sim(n_worlds=num_envs, n_drones=1, device=device, physics=physics)

        self.freq = freq
        assert self.sim.freq >= self.sim.control_freq, "Sim freq must be higher than control freq"
        if not self.sim.freq % self.freq == 0:
            # We can handle other cases, but it's not recommended
            warnings.warn("Simulation frequency should be a multiple of env frequency.")
        if self.sim.control == Control.state:
            raise NotImplementedError("State control currently not supported")

        self.n_substeps = self.sim.freq // self.freq
        self.prev_done = jnp.zeros((self.sim.n_worlds), dtype=jnp.bool_, device=self.device)

        self.single_action_space = action_space(self.sim.control)
        self.action_space = batch_space(self.single_action_space, self.sim.n_worlds)

        self.single_observation_space = spaces.Dict(
            {
                "pos": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "quat": spaces.Box(-np.inf, np.inf, shape=(4,)),
                "vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "ang_vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
            }
        )
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
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

    def reset_masked(self, mask: Array, reset_params: dict | None = None) -> None:
        default_reset_params = {
            "pos_min": jnp.array([-1.0, -1.0, 1.0]),  # x,y,z
            "pos_max": jnp.array([1.0, 1.0, 2.0]),  # x,y,z
            "vel_min": -1.0,
            "vel_max": 1.0,
        }

        if reset_params is not None:
            invalid_keys = set(reset_params.keys()) - set(default_reset_params.keys())
            if invalid_keys:
                raise ValueError(f"Invalid bounds keys: {invalid_keys}")
            default_reset_params.update(reset_params)

        self.sim.reset(mask=mask)
        mask3d = mask[:, None, None]
        # NOTE Setting initial ryp_rate when using physics.sys_id will not have an impact
        # Sample initial pos
        self.jax_key, subkey = jax.random.split(self.jax_key)
        init_pos = jax.random.uniform(
            key=subkey,
            shape=(self.sim.n_worlds, self.sim.n_drones, 3),
            minval=default_reset_params["pos_min"],
            maxval=default_reset_params["pos_max"],
        )
        self.sim.data = self.sim.data.replace(
            states=self.sim.data.states.replace(
                pos=jnp.where(mask3d, init_pos, self.sim.data.states.pos)
            )
        )
        # Sample initial vel
        self.jax_key, subkey = jax.random.split(self.jax_key)
        init_vel = jax.random.uniform(
            key=subkey,
            shape=(self.sim.n_worlds, self.sim.n_drones, 3),
            minval=default_reset_params["vel_min"],
            maxval=default_reset_params["vel_max"],
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
        return {k: v.squeeze() for k, v in zip(fields, states)}

    def close(self):
        self.sim.close()


class CrazyflowEnvReachGoal(CrazyflowBaseEnv):
    """JAX Gymnasium environment for Crazyflie simulation."""

    def __init__(self, render_goal_marker: bool = False, **kwargs: dict):
        super().__init__(**kwargs)
        self.render_goal_marker = render_goal_marker
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
        norm_distance = jnp.linalg.norm(states.pos, axis=2)
        reward = jnp.exp(-2.0 * norm_distance)
        reward = jnp.where(terminated.reshape(-1, 1), -1.0, reward)
        reward = jnp.where(prev_done.reshape(-1, 1), 0.0, reward)
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

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        if self.render_goal_marker:
            for i in range(self.sim.n_worlds):
                if hasattr(self.sim, "viewer") and self.sim.viewer is not None:
                    self.sim.viewer.viewer.add_marker(
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.02, 0.02, 0.02]),
                        pos=np.array(self.goal[i]),
                        rgba=np.array([1, 0, 0, 0.5]),
                    )
        return super().step(action)

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
        reward = jnp.where(terminated.reshape(-1, 1), -1.0, reward)
        reward = jnp.where(prev_done.reshape(-1, 1), 0.0, reward)
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

    def __init__(self, render_landing_marker: bool = False, **kwargs: dict):
        super().__init__(**kwargs)
        self.render_landing_target = render_landing_marker
        spec = {k: v for k, v in self.single_observation_space.items()}
        spec["difference_to_goal"] = spaces.Box(-np.inf, np.inf, shape=(3,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)
        self.goal = jnp.zeros((self.sim.n_worlds, 3), dtype=jnp.float32, device=self.device)
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
        reward = jnp.where(terminated.reshape(-1, 1), -1.0, reward)
        reward = jnp.where(prev_done.reshape(-1, 1), 0.0, reward)
        return reward

    def reset_masked(self, mask: Array) -> None:
        super().reset_masked(mask)

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        if self.render_landing_target:
            for i in range(self.sim.n_worlds):
                if hasattr(self.sim, "viewer") and self.sim.viewer is not None:
                    self.sim.viewer.viewer.add_marker(
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.02, 0.02, 0.02]),
                        pos=np.array(self.goal[i]),
                        rgba=np.array([1, 0, 0, 0.5]),
                    )
        return super().step(action)

    def _obs(self) -> dict[str, Array]:
        obs = super()._obs()
        obs["difference_to_goal"] = [self.goal - self.sim.data.states.pos]
        return obs


def render_trajectory(viewer: MujocoRenderer | None, pos: Array) -> None:
    """Render trajectory."""
    if viewer is None:
        return
    assert pos.ndim == 2 and pos.shape[1] == 3, f"Expected shape (n_points, 3), got {pos.shape}"
    for p in pos:
        viewer.viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.02, 0.02, 0.02]),
            pos=p,
            rgba=np.array([1, 0, 0, 0.8]),
        )


class CrazyflowEnvFigureEightTrajectory(CrazyflowBaseEnv):
    """JAX Gymnasium environment for Crazyfly simulation.

    This environment is used to follow a figure-eight trajectory. The observations contain the
    relative position errors to the next `n_samples` points that are distanced by `samples_dt`. The
    reward is based on the distance to the next trajectory point.
    """

    def __init__(
        self,
        n_samples: int = 10,
        samples_dt: float = 0.1,
        trajectory_time: float = 10.0,
        render_samples: bool = False,
        **kwargs: dict,
    ):
        """Initializes the environment.

        Args:
            n_samples: Number of next trajectory points to sample for observations.
            samples_dt: Time between trajectory sample points in seconds.
            trajectory_time: Total time for completing the figure-eight trajectory in seconds.
            render_samples: Flag to enable/disable rendering of the trajectory sample.
            **kwargs: Arguments passed to the Crazyfly simulation.
        """
        super().__init__(**kwargs)
        if trajectory_time < self.time_horizon_in_seconds:
            raise ValueError("Trajectory time must be greater than time horizon in seconds")

        self.render_samples = render_samples

        # Create the figure eight trajectory
        n_steps = int(np.ceil(trajectory_time * self.freq))
        t = np.linspace(0, 2 * np.pi, n_steps)
        radius = 1  # Radius for the circles
        y = np.zeros_like(t)  # x is 0 everywhere
        x = radius * np.sin(t)  # Scale amplitude for 1-meter diameter
        z = radius * np.sin(2 * t) + 1.2  # Scale amplitude for 1-meter diameter
        self.trajectory = np.array([x, y, z]).T

        self.sample_offsets = np.array(np.arange(n_samples) * self.freq * samples_dt, dtype=int)

        # Define trajectory sampling parameters
        self.n_samples = n_samples
        self.samples_dt = samples_dt

        spec = {k: v for k, v in self.single_observation_space.items()}
        spec["local_samples"] = spaces.Box(-np.inf, np.inf, shape=(3 * self.n_samples,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)

    @property
    def reward(self) -> Array:
        return self._reward(
            self.prev_done, self.terminated, self.sim.data.states, self.trajectory[self.steps]
        ).reshape(-1)

    @staticmethod
    @jax.jit
    def _reward(prev_done: Array, terminated: Array, states: SimState, goal: Array) -> Array:
        norm_distance = jnp.linalg.norm(
            states.pos - goal, axis=2
        )  # distance to next trajectory point
        reward = jnp.exp(-2.0 * norm_distance)
        reward = jnp.where(terminated.reshape(-1, 1), -1.0, reward)
        reward = jnp.where(prev_done.reshape(-1, 1), 0.0, reward)
        return reward

    def reset_masked(self, mask: Array) -> None:
        reset_params = {
            "pos_min": jnp.array([-0.1, -0.1, 1.1]),  # x,y,z
            "pos_max": jnp.array([0.1, 0.1, 1.3]),  # x,y,z
            "vel_min": -0.5,
            "vel_max": 0.5,
        }
        super().reset_masked(mask, reset_params)

    def _obs(self) -> dict[str, Array]:
        obs = super()._obs()
        idx = (self.steps + self.sample_offsets[None, ...]) % self.trajectory.shape[0]
        next_trajectory = self.trajectory[idx, ...]
        if self.render_samples:
            render_trajectory(self.sim.viewer, next_trajectory[0])
        dpos = next_trajectory - self.sim.data.states.pos
        obs["local_samples"] = dpos.reshape(-1, 3 * self.n_samples)
        return obs

    @property
    def steps(self) -> Array:
        """The parameters tau for the next trajectory. Must be in [0,1]."""
        return self.sim.data.core.steps // (self.sim.freq // self.freq) - 1


class FigureEightXY(CrazyflowBaseEnv):
    """JAX Gymnasium environment for Crazyflie simulation with a figure eight in the x-y plane.

    This environment has a single, predefined trajectory that the drone should follow. Each episode
    lasts exactly 10 seconds at 50Hz. The reward is based on the distance to the current trajectory
    point.
    """

    def __init__(self, num_envs: int = 1, device: str = "cpu"):
        """Initialize the fixed trajectory environment."""
        super().__init__(num_envs=num_envs, freq=50, device=device)
        # Create a fixed trajectory (a simple circle in the x-z plane)
        n_steps = int(self.time_horizon_in_seconds * self.freq)
        t = np.linspace(0, self.time_horizon_in_seconds, n_steps)

        traj_period = 5.0
        traj_freq = 2.0 * np.pi / traj_period
        x = np.sin(traj_freq * t)
        y = np.sin(traj_freq * t) * np.cos(traj_freq * t)
        z = np.ones_like(t)

        dims = 12  # x dx y dy z dz r p yaw dr dp dyaw
        self.trajectory = np.zeros((n_steps, dims))
        self.trajectory[:, 0] = x
        self.trajectory[:, 2] = y
        self.trajectory[:, 4] = z

        # Flag to enable/disable rendering of the trajectory
        self.render_trajectory = True

    @property
    def reward(self) -> Array:
        """Calculate reward based on distance to current trajectory point."""
        step = self.steps % self.trajectory.shape[0]
        return self._reward(
            self.prev_done, self.terminated, self.sim.data.states, self.trajectory[step, [0, 2, 4]]
        ).reshape(-1)

    @staticmethod
    @jax.jit
    def _reward(prev_done: Array, terminated: Array, states: SimState, target: Array) -> Array:
        """Calculate reward based on distance to current trajectory point and velocity alignment."""
        norm_distance = jnp.linalg.norm(states.pos - target, axis=-1)
        reward = jnp.exp(-2.0 * norm_distance)
        # Apply penalties for termination and previous done states
        reward = jnp.where(terminated.reshape(-1, 1), -1.0, reward)
        reward = jnp.where(prev_done.reshape(-1, 1), 0.0, reward)
        return reward

    def reset_masked(self, mask: Array) -> None:
        """Reset the environment with specific initial conditions.

        The drone starts near the beginning of the trajectory with low velocity.

        Args:
            mask: Boolean array indicating which environments to reset
        """
        # Set initial position near the start of the trajectory

        reset_params = {
            "pos_min": jnp.array([self.trajectory[0, i] - 0.1 for i in (0, 2, 4)]),
            "pos_max": jnp.array([self.trajectory[0, i] + 0.1 for i in (0, 2, 4)]),
            "vel_min": -0.2,
            "vel_max": 0.2,
        }
        super().reset_masked(mask, reset_params)

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        """Step the environment and render trajectory if enabled."""
        if self.render_trajectory and hasattr(self.sim, "viewer") and self.sim.viewer is not None:
            # Render the full trajectory
            render_trajectory(self.sim.viewer, self.trajectory[::5, [0, 2, 4]])

            # Highlight current target point
            step = self.steps % self.trajectory.shape[0]
            current_target = self.trajectory[step, [0, 2, 4]]
            self.sim.viewer.viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.05, 0.05, 0.05]),  # Larger marker for current target
                pos=np.array(current_target[0]),
                rgba=np.array([0, 1, 0, 0.8]),  # Green color for current target
            )

        return super().step(action)

    @property
    def steps(self) -> Array:
        """Get the current step index in the trajectory."""
        return self.sim.data.core.steps // (self.sim.freq // self.freq)


class CrazyflowRL(VectorWrapper):
    """Wrapper to use the crazyflow JAX environments with common DRL frameworks.

    Currently, this wrapper clips the expected actions to [-1,1] and rescales them to the action
    space expected in simulation.
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

    def step(self, actions: Array) -> tuple[dict, Array, Array, Array, dict]:
        actions = np.clip(actions, -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(self.actions(actions))
        return obs, reward, terminated, truncated, info

    def actions(self, actions: Array) -> Array:
        """Rescale and clip actions from [-1, 1] to [action_sim_low, action_sim_high]."""
        # Rescale actions using the computed scale and mean
        rescaled_actions = actions * self.action_scale + self.action_mean
        # Ensure actions are within the valid range of the simulation action space
        rescaled_actions = np.clip(rescaled_actions, self.action_sim_low, self.action_sim_high)
        return rescaled_actions
