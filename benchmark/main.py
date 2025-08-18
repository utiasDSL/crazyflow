import csv
import time
from datetime import datetime
from pathlib import Path

import fire
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from jax.errors import JaxRuntimeError
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


def profile_gym_env_step(
    sim_config: config_dict.ConfigDict, n_steps: int, device: str, print_summary: bool = True
) -> list[float]:
    """Profile the Crazyflow gym environment step performance."""
    times = []
    device = jax.devices(device)[0]

    envs = gymnasium.make_vec(
        "DroneReachPos-v0",
        max_episode_time=3,
        num_envs=sim_config.n_worlds,
        device=sim_config.device,
        freq=sim_config.freq,
        physics=sim_config.physics,
    )

    # Action for going up (in attitude control)
    action = np.zeros((sim_config.n_worlds, 4), dtype=np.float32)
    action[..., 0] = 0.3
    # Step through env once to ensure JIT compilation
    envs.reset()
    envs.step(action)

    jax.block_until_ready(envs.unwrapped.sim.data)  # Ensure JIT compiled dynamics

    # Step through the environment
    for _ in range(n_steps):
        tstart = time.perf_counter()
        envs.step(action)
        jax.block_until_ready(envs.unwrapped.sim.data)
        times.append(time.perf_counter() - tstart)

    envs.close()
    if print_summary:
        print("Gym env step performance:")
        analyze_timings(times, n_steps, envs.unwrapped.sim.n_worlds, sim_config.freq)
    return times


def profile_step(
    sim_config: config_dict.ConfigDict, n_steps: int, device: str, print_summary: bool = True
) -> list[float]:
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

    if print_summary:
        print("Sim step performance:")
        analyze_timings(times, n_steps, sim.n_worlds, sim.freq)
    return times


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


def main(device: str = "cpu", n_worlds_exp: int = 6):
    """Main entry point for profiling."""
    sim_config = config_dict.ConfigDict()
    sim_config.n_worlds = 1
    sim_config.n_drones = 1
    sim_config.physics = "analytical"
    sim_config.control = "attitude"
    sim_config.attitude_freq = 500
    sim_config.device = device
    sim_config.freq = 500

    max_seconds_per_run = 60.0

    print("\nRunning benchmarks for increasing number of parallel environments...")

    # Create a CSV file to store results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = Path(__file__).parent / "data" / f"benchmark_results_{timestamp}.csv"
    csv_file.parent.mkdir(exist_ok=True)

    # Create CSV writer and write header
    with open(csv_file, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            [
                "test_type",
                "n_drones",
                "n_worlds",
                "n_steps",
                "total_time_s",
                "avg_step_time_s",
                "fps",
                "real_time_factor",
                "device",
            ]
        )

    # Reopen the file in append mode for each result

    n_steps = 1000
    skip_sim, skip_gym = False, False
    # Test with increasing number of parallel environments (worlds)
    for n_worlds in [10**i for i in range(n_worlds_exp + 1)]:
        sim_config.n_worlds = n_worlds
        print("-" * 80)
        if not skip_sim:
            # Test with a single step first to see if we should continue
            sim_config.freq = 500  # Test sim at 500 hz
            single_step_time = profile_step(sim_config, 2, device, print_summary=False)[1]

            # If single step takes too long, skip this and remaining tests
            if single_step_time > max_seconds_per_run / n_steps:  # threshold for the tests
                print(
                    f"  Skipping benchmark for {n_worlds} and higher - projected time "
                    f"{single_step_time * n_steps:.2f}s (> 1m)"
                )
                skip_sim = True

        if not skip_sim:
            # Configure simulator
            print(f"Running simulator benchmark ({n_worlds} worlds)...")
            # Run simulator benchmark using existing function
            times_sim = profile_step(sim_config, n_steps, device)

            # Calculate metrics for CSV
            total_time = sum(times_sim)
            avg_step_time = np.mean(times_sim)
            n_frames = n_steps * n_worlds
            fps = n_frames / total_time
            real_time_factor = (n_steps / sim_config.freq) * n_worlds / total_time

            # Save simulator results
            # Reopen CSV writer in append mode
            with open(csv_file, "a", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(
                    [
                        "simulator",
                        1,  # n_drones
                        n_worlds,
                        n_steps,
                        total_time,
                        avg_step_time,
                        fps,
                        real_time_factor,
                        sim_config.device,
                    ]
                )
                f.flush()

        if not skip_gym:
            print(f"Running gym environment benchmark ({n_worlds} worlds)...")
            # Run gym environment benchmark using existing function
            sim_config.freq = 50  # Test gym at 50 hz
            try:
                step_times = profile_gym_env_step(sim_config, 2, device, print_summary=False)
                single_step_time = step_times[1]
                # If single step takes too long, skip this test only
                if single_step_time > max_seconds_per_run / n_steps:  # threshold for the tests
                    print(
                        f"  Skipping benchmark for {n_worlds} - projected time "
                        f"{single_step_time * n_steps:.2f}s (> 1m)"
                    )
                    skip_gym = True
            except JaxRuntimeError:
                print(f"  Skipping benchmark for {n_worlds} - resource exhausted")
                skip_gym = True

        if not skip_gym:
            try:
                times_gym = profile_gym_env_step(sim_config, n_steps, device)
            except JaxRuntimeError:
                print(f"  Skipping benchmark for {n_worlds} - resource exhausted")
                skip_gym = True
                continue

            # Calculate metrics for CSV
            total_time = sum(times_gym)
            avg_step_time = np.mean(times_gym)
            n_frames = n_steps * n_worlds
            fps = n_frames / total_time
            real_time_factor = (n_steps / sim_config.freq) * sim_config.n_worlds / total_time

            # Save gym environment results
            with open(csv_file, "a", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(
                    [
                        "gym_env",
                        sim_config.n_drones,
                        sim_config.n_worlds,
                        n_steps,
                        total_time,
                        avg_step_time,
                        fps,
                        real_time_factor,
                        sim_config.device,
                    ]
                )
                f.flush()

    print(f"\nBenchmark results saved to {csv_file}")


if __name__ == "__main__":
    fire.Fire(main)
