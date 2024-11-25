import jax
import jax.numpy as jnp
import numpy as np
import gymnasium
from ml_collections import config_dict
from pyinstrument import Profiler
from pyinstrument.renderers.html import HTMLRenderer

from crazyflow.sim.core import Sim
import crazyflow.gymnasium_envs


def profile_step(sim_config: config_dict.ConfigDict, n_steps: int, device: str):
    sim = Sim(
        **sim_config
    )
    device = jax.devices(device)[0]
    ndim = 13 if sim.control == "state" else 4
    control_fn = sim.state_control if sim.control == "state" else sim.attitude_control
    cmd = np.zeros((sim.n_worlds, sim.n_drones, ndim))
    # Ensure JIT compiled dynamics and control
    sim.reset()
    control_fn(cmd)
    sim.step()
    control_fn(cmd)
    sim.step()
    sim.reset()
    jax.block_until_ready(sim.states.pos)

    profiler = Profiler()
    profiler.start()

    for _ in range(n_steps):
        control_fn(cmd)
        # sim.reset()
        sim.step()
        jax.block_until_ready(sim.states.pos)
    profiler.stop()
    renderer = HTMLRenderer()
    renderer.open_in_browser(profiler.last_session)

def profile_gym_env_step(sim_config: config_dict.ConfigDict, n_steps: int, device: str):
    device = jax.devices(device)[0]

    envs = gymnasium.make_vec(
        "CrazyflowEnvReachGoal-v0",
        max_episode_steps=200,
        return_datatype="numpy",
        num_envs=sim_config.n_worlds,
        jax_random_key=42,
        **sim_config,
    )

    # Action for going up (in attitude control)
    action = np.array(
        [[[-0.3, 0, 0, 0] for _ in range(sim_config.n_drones)] for _ in range(sim_config.n_worlds)],
        dtype=np.float32,
    ).reshape(sim_config.n_worlds, -1)

    # step through env once to ensure JIT compilation
    _, _ = envs.reset_all(seed=42)
    _, _, _, _, _ = envs.step(action)
    _, _ = envs.reset_all(seed=42)
    _, _, _, _, _ = envs.step(action)
    _, _ = envs.reset_all(seed=42)
    _, _, _, _, _ = envs.step(action)
    _, _ = envs.reset_all(seed=42)
    
    jax.block_until_ready(envs.unwrapped.sim._mjx_data)  # Ensure JIT compiled dynamics

    profiler = Profiler()
    profiler.start()

    for _ in range(n_steps):
        _, _, _, _, _ = envs.step(action)
        jax.block_until_ready(envs.unwrapped.sim._mjx_data)
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
    sim_config.controller = "emulatefirmware"
    sim_config.device = device

    profile_step(sim_config, 1000, device)
    # old | new
    # sys_id + attitude:
    # 0.61 reset, 0.61 step  |  0.61 reset, 0.61 step
    # sys_id + state:
    # 14.53 step, 0.53 reset |  0.75 reset, 0.88 step

    # Analytical + attitude:
    # 0.75 reset, 9.38 step  |  0.75 reset, 0.80 step
    # Analytical + state:
    # 0.75 reset, 15.1 step  |  0.75 reset, 0.82 step

    profile_gym_env_step(sim_config, 1000, device)



if __name__ == "__main__":
    main()
