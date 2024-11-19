from enum import Enum
from typing import Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from crazyflow.control.controller import Control, Controller
from crazyflow.sim.core import Sim
from crazyflow.sim.physics import Physics


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

        # Sim parameters, TODO: pass as kwargs
        physics=Physics.analytical,
        control=Control.state,
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

        # Init physics sim
        self.sim = Sim(
            n_worlds=num_envs,
            n_drones=num_drones_per_env,
            physics=physics,
            control=control,
            controller=controller,
            freq=freq,
            control_freq=control_freq,
            device=device,
        )

        # More env parameters
        self.steps = np.zeros(num_envs, dtype=np.int32)
        self.prev_done = np.zeros(num_envs, dtype=np.bool_)

        # TODO: set observation_space and action_space
        self.single_action_space = spaces.Discrete(2)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.single_observation_space = spaces.Box(-np.inf, np.inf, dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        
        self.state = None
        self.steps_beyond_terminated = None

        def step(
            self, action: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid"
            assert self.state is not None, "Call reset before using step method."
            pass

        def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
        ):
            # Resets all (!) envs
            super().reset(seed=seed)
            pass

        def render(self):
            pass