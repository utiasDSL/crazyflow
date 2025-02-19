import time

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict

import crazyflow  # noqa: F401, ensure gymnasium envs are registered
from crazyflow.sim import Sim


def analyze_timings(times: list[float], n_steps: int, n_worlds: int, freq: float) -> None:
    """Analyze timing results and print performance metrics."""
    if not times:
        raise ValueError("The list of timing results is empty.")

    tmin, idx_tmin = np.min(times), np.argmin(times)
    tmax, idx_tmax = np.max(times), np.argmax(times)

    # Check for significant variance
    if tmax / tmin > 10:
        print("Warning: Fn time varies by more than 10x. Is JIT compiling during the benchmark?")
        print(f"Times: max {tmax:.2e} @ {idx_tmax}, min {tmin:.2e} @ {idx_tmin}")

    # Performance metrics
    n_frames = n_steps * n_worlds  # Number of frames simulated
    total_time = np.sum(times)
    avg_step_time = np.mean(times)
    step_time_std = np.std(times)
    fps = n_frames / total_time
    real_time_factor = (n_steps / freq) * n_worlds / total_time

    print(
        f"Avg fn time: {avg_step_time:.2e}s, std: {step_time_std:.2e}"
        f"\nFPS: {fps:.3e}, Real time factor: {real_time_factor:.2e}\n"
    )


def profile_gym_env_step(sim_config: config_dict.ConfigDict, n_steps: int, device: str):
    """Profile the Crazyflow gym environment step performance."""
    times = []
    device = jax.devices(device)[0]

    envs = gymnasium.make_vec(
        "DroneReachPos-v0",
        time_horizon_in_seconds=3,
        num_envs=sim_config.n_worlds,
        device=sim_config.device,
        freq=sim_config.attitude_freq,
        physics=sim_config.physics,
    )

    # Action for going up (in attitude control)
    action = np.zeros((sim_config.n_worlds, 4), dtype=np.float32)
    action[..., 0] = 0.3
    # Step through env once to ensure JIT compilation
    envs.reset(seed=42)
    envs.step(action)

    jax.block_until_ready(envs.unwrapped.sim.data)  # Ensure JIT compiled dynamics

    # Step through the environment
    for _ in range(n_steps):
        tstart = time.perf_counter()
        envs.step(action)
        jax.block_until_ready(envs.unwrapped.sim.data)
        times.append(time.perf_counter() - tstart)

    envs.close()
    print("Gym env step performance:")
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
    sim.step(sim.freq // sim.control_freq)
    jax.block_until_ready(sim.data)  # Ensure JIT compiled dynamics

    for _ in range(n_steps):
        tstart = time.perf_counter()
        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        jax.block_until_ready(sim.data)
        times.append(time.perf_counter() - tstart)

    print("Sim step performance:")
    analyze_timings(times, n_steps, sim.n_worlds, sim.freq)


def profile_reset(sim_config: config_dict.ConfigDict, n_steps: int, device: str):
    """Profile the Crazyflow simulator reset performance."""
    sim = Sim(**sim_config)
    times = []
    times_masked = []
    device = jax.devices(device)[0]

    # Ensure JIT compiled reset
    sim.reset()
    jax.block_until_ready(sim.data)

    # Test full reset
    for _ in range(n_steps):
        tstart = time.perf_counter()
        sim.reset()
        jax.block_until_ready(sim.data)
        times.append(time.perf_counter() - tstart)

    # Test masked reset (only reset first world)
    mask = jnp.zeros(sim.n_worlds, dtype=bool, device=device)
    mask = mask.at[0].set(True)
    sim.reset(mask)
    jax.block_until_ready(sim.data)

    for _ in range(n_steps):
        tstart = time.perf_counter()
        sim.reset(mask)
        jax.block_until_ready(sim.data)
        times_masked.append(time.perf_counter() - tstart)

    print("Sim reset performance:")
    analyze_timings(times, n_steps, sim.n_worlds, sim.freq)
    print("Sim masked reset performance:")
    analyze_timings(times_masked, n_steps, sim.n_worlds, sim.freq)


def main():
    """Main entry point for profiling."""
    device = "cpu"
    sim_config = config_dict.ConfigDict()
    sim_config.n_worlds = 1
    sim_config.n_drones = 1
    sim_config.physics = "analytical"
    sim_config.control = "attitude"
    sim_config.attitude_freq = 500
    sim_config.device = device

    print("Simulator performance\n")
    profile_step(sim_config, 1000, device)
    profile_reset(sim_config, 1000, device)

    print("Gymnasium environment performance\n")
    profile_gym_env_step(sim_config, 1000, device)


if __name__ == "__main__":
    main()
