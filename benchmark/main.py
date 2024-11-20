import time

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.sim.core import Sim


def profile_step(sim: Sim, n_steps: int, device: str):
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
    if max(times) / min(times) > 5:
        tmin, idx_tmin = np.min(times), np.argmin(times)
        tmax, idx_tmax = np.max(times), np.argmax(times)
        print("Warning: step time varies by more than 5x. Is JIT compiling during the benchmark?")
        print(f"Times: max {tmax:.2e}@{idx_tmax}, min {tmin:.2e}@{idx_tmin}")
    n_frames = n_steps * sim.n_worlds  # Number of frames simulated
    total_time = np.sum(times)
    real_time_factor = (n_steps / sim.freq) * sim.n_worlds / total_time
    print(
        f"Avg step time: {np.mean(times):.2e}s, std: {np.std(times):.2e}"
        f"\nFPS: {n_frames / total_time:.3e}, Real time factor: {real_time_factor:.2e}"
    )


def main():
    device = "cpu"
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        physics="sys_id",
        control="attitude",
        controller="emulatefirmware",
        device=device,
    )
    profile_step(sim, 100, device)


if __name__ == "__main__":
    main()
