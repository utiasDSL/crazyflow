from enum import Enum
from typing import Optional, Tuple, Union, List
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from crazyflow.control.controller import Control, Controller
from crazyflow.sim.core import Sim
from crazyflow.sim.physics import Physics


# observations corresponding to simulation states
class Observations(Enum):
    pos = "pos"
    quat = "quat"
    vel = "vel"
    ang_vel = "ang_vel"
    rpy_rates = "rpy_rates"
    default = [pos, quat, vel, ang_vel]


# Example for Env and VectorEnv here: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py#L476
class CrazyflowVectorEnv(VectorEnv):
    # TODO dynamically set render_fps
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        # Env parameters
        num_envs: int = 1,
        num_drones_per_env: int = 1,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
        observations: List[Observations] = Observations.default,
        # Sim parameters
        # TODO: pass as kwargs
        control: Control = Control.default,
        physics=Physics.analytical,
        controller=Controller.emulatefirmware,
        freq=500,
        control_freq=500,
        device="cpu",
    ):
        # Env parameters
        self.num_envs = num_envs
        self.num_drones_per_env = num_drones_per_env
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.observations = observations
        
        self.control = control
        self.controller = controller
        self.freq = freq
        self.control_freq = control_freq
        self.device = device

        # Init physics sim
        self.sim = Sim(
            n_worlds=num_envs,
            n_drones=num_drones_per_env,
            # TODO pass as kwargs
            physics=physics,
            control=control,
            controller=controller,
            freq=freq,
            control_freq=control_freq,
            device=device,
        )

        # Env setup
        self.steps = np.zeros((num_envs, num_drones_per_env), dtype=np.int32)
        self.prev_done = np.zeros((num_envs, num_drones_per_env), dtype=np.bool_)

        # TODO: scale action spaces
        self.single_action_space = spaces.Box(
            -1, 1, shape=(math.prod(self.sim._controls[control].shape[1:]),), dtype=np.float32
        ) # num_drones_per_env * num_controls
        self.action_space = batch_space(self.single_action_space, num_envs)

        _obs_size = 0
        for obs in self.observations.value:
            _obs_size += math.prod(self.sim.states[obs].shape[1:]) # num_obs_per_drone * num_controls
        self.single_observation_space = spaces.Box(-np.inf, np.inf, shape=(_obs_size,), dtype=np.float32) # set limits to np.inf as we are expecting a normalization wrapper
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        def step(
            self, action: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
            assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
            assert self.state is not None, "Call reset before using step method."
            pass

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            # Resets all (!) envs
            super().reset(seed=seed)
            pass

        def render(self):
            pass
