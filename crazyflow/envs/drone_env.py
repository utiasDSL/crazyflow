import warnings
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space
from jax import Array
from numpy.typing import NDArray

from crazyflow.control import Control
from crazyflow.drones import Drone
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.pipeline import append_fn
from crazyflow.utils import leaf_replace


def action_space(control_type: Control, dynamics: Dynamics, drone: Drone) -> spaces.Box:
    """Select the appropriate action space for a given control type.

    Args:
        control_type: The desired control mode.
        dynamics: Dynamics of the environment.
        drone: Drone of the environment.

    Returns:
        The action space.
    """
    match control_type:
        case Control.attitude:
            params = load_params(dynamics, drone)
            thrust_min, thrust_max = params["thrust_min"] * 4, params["thrust_max"] * 4
            return spaces.Box(
                np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, thrust_min], dtype=np.float32),
                np.array([np.pi / 2, np.pi / 2, np.pi / 2, thrust_max], dtype=np.float32),
            )
        case Control.force_torque:
            return spaces.Box(-1.0, 1.0, shape=(6,))
        case _:
            raise ValueError(f"Invalid control type {control_type}")


class DroneEnv(VectorEnv):
    """JAX-based Gymnasium core environment for Crazyflow.

    ## Action space
    We have three types of actions:
    - `attitude`: 4D vector consisting of [collective thrust, roll, pitch, yaw]
    - `thrust`: 4D vector consisting of the individual motor thrusts [f1, f2, f3, f4]
    - `state`: Currently not implemented

    The default action space is `attitude`.
    """

    metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP}

    def __init__(
        self,
        *,
        num_envs: int = 1,
        max_episode_time: float = 10.0,
        dynamics: Dynamics = Dynamics.so_rpy,
        drone: Drone = Drone.cf2x_L250,
        freq: int = 500,
        device: str = "cpu",
        reset_randomization: Callable[[SimData, Array], SimData] | None = None,
    ):
        """Initialize the CrazyflowEnv.

        Args:
            num_envs: The number of environments to run in parallel.
            max_episode_time: The time horizon after which episodes are truncated (s).
            dynamics: The crazyflow dynamics.
            drone: Drone of the environment.
            freq: The frequency at which the environment is run.
            device: The device of the environment and the simulation.
            reset_randomization: A function that randomizes the initial state of the simulation. If
                None, the default randomization for pos and vel is used.
        """
        self.num_envs = num_envs
        self.device = jax.devices(device)[0]
        self.freq = freq
        self.max_episode_time = max_episode_time
        assert Dynamics(dynamics) in Dynamics, f"Invalid dynamics type {dynamics}"

        # Initialize the simulation
        self.sim = Sim(n_worlds=num_envs, n_drones=1, drone=drone, device=device, dynamics=dynamics)
        assert self.sim.freq >= self.sim.control_freq, "Sim freq must be higher than control freq"
        if not self.sim.freq % self.freq == 0:
            # We can handle other cases, but it's not recommended
            warnings.warn("Simulation frequency should be a multiple of env frequency.")
        if self.sim.control == Control.state:
            raise NotImplementedError("State control currently not supported")

        # Compile our domain randomization on reset into the sim pipeline
        if reset_randomization is None:
            reset_randomization = self._reset_randomization
        append_fn(self.sim.reset_pipeline, reset_randomization, name="reset_randomization")
        self.sim.build_reset_fn()

        self.n_substeps = self.sim.freq // self.freq
        self._marked_for_reset = jnp.zeros((self.sim.n_worlds), dtype=jnp.bool_, device=self.device)

        # Define action and observation spaces
        self.single_action_space = action_space(self.sim.control, self.sim.dynamics, self.sim.drone)
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

    def step(self, actions: Array) -> tuple[Array, Array, Array, Array, dict]:
        self._apply_action(actions)
        self.sim.step(self.n_substeps)
        # Reset all environments which terminated or were truncated in the last step
        self._reset(mask=self._marked_for_reset)
        terminated, truncated = self.terminated(), self.truncated()
        self._marked_for_reset = self._done(terminated, truncated)
        return self.obs(), self.reward(), terminated, truncated, {}

    def _apply_action(self, action: Array):
        low, high = self.action_space.low, self.action_space.high
        action = self._sanitize_action(action, low, high, self.sim.n_worlds, self.sim.device)
        match self.sim.control:
            case Control.state:
                raise NotImplementedError("State control currently not supported")
            case Control.attitude:
                self.sim.attitude_control(action)
            case Control.force_torque:
                self.sim.force_torque_control(action)
            case _:
                raise ValueError(f"Invalid control type {self.sim.control}")

    @staticmethod
    @partial(jax.jit, static_argnames=["n_worlds", "device"])
    def _sanitize_action(
        action: Array, low: NDArray, high: NDArray, n_worlds: int, device: str
    ) -> Array:
        action = jnp.clip(action, low, high)
        return jnp.array(action, device=device).reshape((n_worlds, 1, -1))

    @staticmethod
    @jax.jit
    def _done(terminated: Array, truncated: Array) -> Array:
        return terminated | truncated

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Array], dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.sim.seed(seed)
        self._reset(options=options)
        self._marked_for_reset = self._marked_for_reset.at[...].set(False)
        return self.obs(), {}

    def render(self):
        self.sim.render()

    def close_extras(self):
        self.sim.close()

    def _reset(self, mask: Array | None = None, options: dict | None = None) -> None:
        self.sim.reset(mask=mask)

    def reward(self) -> Array:
        raise NotImplementedError

    def terminated(self) -> Array:
        return self._terminated(self.sim.data.states.pos)

    def truncated(self) -> Array:
        return self._truncated(self.sim.time[..., 0], self.max_episode_time)

    @staticmethod
    @jax.jit
    def _terminated(pos: Array) -> Array:
        return pos[:, 0, 2] < 0  # Terminate if the drone has crashed into the ground

    @staticmethod
    @jax.jit
    def _truncated(time: Array, max_episode_time: float) -> Array:
        return time >= max_episode_time

    def obs(self) -> dict[str, Array]:
        obs = {
            "pos": self.sim.data.states.pos,
            "quat": self.sim.data.states.quat,
            "vel": self.sim.data.states.vel,
            "ang_vel": self.sim.data.states.ang_vel,
        }
        # drop n_drones dimension, as it is always 1 for now
        return {k: v[:, 0, :] for k, v in obs.items()}

    @staticmethod
    def _reset_randomization(data: SimData, _: SimData, mask: Array) -> SimData:
        """Randomize the initial position and velocity of the drones.

        This function will get compiled into the reset function of the simulation. Therefore, it
        must take data and mask as input arguments and must return a SimData object.
        """
        # Sample initial position
        shape = (data.core.n_worlds, data.core.n_drones, 3)
        pos_min = jnp.array([-1.0, -1.0, 1.0])
        pos_max = jnp.array([1.0, 1.0, 2.0])
        key, pos_key, vel_key = jax.random.split(data.core.rng_key, 3)
        data = data.replace(core=data.core.replace(rng_key=key))
        pos = jax.random.uniform(key=pos_key, shape=shape, minval=pos_min, maxval=pos_max)
        # Sample initial velocity
        vel = jax.random.uniform(key=vel_key, shape=shape, minval=-1.0, maxval=1.0)
        data = data.replace(states=leaf_replace(data.states, mask, pos=pos, vel=vel))
        return data
