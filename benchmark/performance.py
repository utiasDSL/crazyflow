from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium
import jax
import numpy as np
from ml_collections import config_dict
from pyinstrument import Profiler
from pyinstrument.renderers.html import HTMLRenderer

import crazyflow  # noqa: F401, ensure gymnasium envs are registered
from crazyflow.sim import Sim

if TYPE_CHECKING:
    from crazyflow.gymnasium_envs import CrazyflowEnvReachGoal


def profile_step(sim_config: config_dict.ConfigDict, n_steps: int, device: str):
    sim = Sim(**sim_config)
    device = jax.devices(device)[0]
    ndim = 13 if sim.control == "state" else 4
    control_fn = sim.state_control if sim.control == "state" else sim.attitude_control
    cmd = np.zeros((sim.n_worlds, sim.n_drones, ndim))
    # Ensure JIT compiled dynamics and control
    sim.reset()
    control_fn(cmd)
    sim.step()
    jax.block_until_ready(sim.data)

    profiler = Profiler()
    profiler.start()

    for _ in range(n_steps):
        control_fn(cmd)
        # sim.reset()
        sim.step()
        jax.block_until_ready(sim.data)
    profiler.stop()
    renderer = HTMLRenderer()
    renderer.open_in_browser(profiler.last_session)


def profile_gym_env_step(sim_config: config_dict.ConfigDict, n_steps: int, device: str):
    device = jax.devices(device)[0]

    envs: CrazyflowEnvReachGoal = gymnasium.make_vec(
        "DroneReachPos-v0", time_horizon_in_seconds=2, num_envs=sim_config.n_worlds, **sim_config
    )

    # Action for going up (in attitude control)
    action = np.zeros((sim_config.n_worlds, 4), dtype=np.float32)
    action[..., 0] = 0.3

    # Step through env once to ensure JIT compilation.
    envs.reset(seed=42)
    envs.step(action)
    envs.step(action)  # Ensure all paths have been taken at least once
    envs.reset(seed=42)
    jax.block_until_ready(envs.unwrapped.sim.data)

    profiler = Profiler()
    profiler.start()

    for _ in range(n_steps):
        envs.step(action)
        jax.block_until_ready(envs.unwrapped.sim.data)

    profiler.stop()
    renderer = HTMLRenderer()
    renderer.open_in_browser(profiler.last_session)
    envs.close()


def main():
    device = "cpu"
    sim_config = config_dict.ConfigDict()
    sim_config.n_worlds = 1
    sim_config.n_drones = 1
    sim_config.physics = "analytical"
    sim_config.control = "attitude"
    sim_config.device = device

    profile_step(sim_config, 1000, device)
    profile_gym_env_step(sim_config, 1000, device)


if __name__ == "__main__":
    main()
