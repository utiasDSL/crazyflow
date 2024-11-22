import time
from ml_collections import config_dict

import jax
import jax.numpy as jnp
import numpy as np
import gymnasium
from crazyflow.sim.core import Sim
import crazyflow.crazyflow_env.envs


def analyze_timings(times: list[float], n_steps: int, n_worlds: int, freq: float) -> None:
    """Analyze timing results and print performance metrics."""
    if not times:
        raise ValueError("The list of timing results is empty.")
    
    tmin, idx_tmin = np.min(times), np.argmin(times)
    tmax, idx_tmax = np.max(times), np.argmax(times)
    
    # Check for significant variance
    if tmax / tmin > 5:
        print("Warning: step time varies by more than 5x. Is JIT compiling during the benchmark?")
        print(f"Times: max {tmax:.2e}@{idx_tmax}, min {tmin:.2e}@{idx_tmin}")
    
    # Performance metrics
    n_frames = n_steps * n_worlds  # Number of frames simulated
    total_time = np.sum(times)
    avg_step_time = np.mean(times)
    step_time_std = np.std(times)
    fps = n_frames / total_time
    real_time_factor = (n_steps / freq) * n_worlds / total_time
    
    print(
        f"Avg step time: {avg_step_time:.2e}s, std: {step_time_std:.2e}"
        f"\nFPS: {fps:.3e}, Real time factor: {real_time_factor:.2e}"
    )


def profile_gym_env_step(sim_config: config_dict.ConfigDict, n_steps: int, device: str):
    """Profile the Crazyflow gym environment step performance."""
    times = []
    device = jax.devices(device)[0]

    envs = gymnasium.make_vec(
        "crazyflow_env/CrazyflowVectorEnv-v0",
        max_episode_steps=200,
        return_datatype="numpy",
        num_envs=sim_config.n_worlds,
        **sim_config,
    )

    # Action for going up (in attitude control)
    action = np.array(
        [[[0.3, 0, 0, 0] for _ in range(sim_config.n_drones)] for _ in range(sim_config.n_worlds)],
        dtype=np.float32,
    ).reshape(sim_config.n_worlds, -1)

    # step through env once to ensure JIT compilation
    _, _ = envs.reset(seed=42)
    _, _, _, _, _ = envs.step(action)
    _, _ = envs.reset(seed=42)
    _, _, _, _, _ = envs.step(action)
    _, _ = envs.reset(seed=42)
    
    jax.block_until_ready(envs.unwrapped.sim._mjx_data)  # Ensure JIT compiled dynamics

    # Step through the environment
    for _ in range(n_steps):
        tstart = time.perf_counter()
        _, _, _, _, _ = envs.step(action)
        jax.block_until_ready(envs.unwrapped.sim._mjx_data)
        times.append(time.perf_counter() - tstart)

    envs.close()

    analyze_timings(times, n_steps, envs.unwrapped.sim.n_worlds, envs.unwrapped.sim.freq)


def profile_step(sim_config: config_dict.ConfigDict, n_steps: int, device: str):
    """Profile the Crazyflow simulator step performance."""
    sim = Sim(**sim_config)
    times = []
    device = jax.devices(device)[0]

    cmd = jnp.zeros((sim.n_worlds, sim.n_drones, 4), device=device)
    cmd = cmd.at[0, 0, 0].set(1)
    
    sim.reset()
    sim.attitude_control(cmd)
    sim.step()
    sim.reset()
    jax.block_until_ready(sim._mjx_data)  # Ensure JIT compiled dynamics

    for _ in range(n_steps):
        tstart = time.perf_counter()
        sim.attitude_control(cmd)
        sim.step()
        jax.block_until_ready(sim._mjx_data)
        times.append(time.perf_counter() - tstart)

    analyze_timings(times, n_steps, sim.n_worlds, sim.freq)


def main():
    """Main entry point for profiling."""
    device = "cpu"
    sim_config = config_dict.ConfigDict()
    sim_config.n_worlds = 1
    sim_config.n_drones = 1
    sim_config.physics = "sys_id"
    sim_config.control = "attitude"
    sim_config.controller = "emulatefirmware"
    sim_config.device = device

    profile_step(sim_config, 100, device)
    profile_gym_env_step(sim_config, 100, device)


if __name__ == "__main__":
    main()
