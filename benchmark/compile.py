import time

import fire
import jax

from crazyflow import Sim


def main(cache: bool = False):
    """Main entry point for profiling."""
    if cache:
        jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

    # Time initialization
    start = time.perf_counter()
    sim = Sim(n_worlds=1, n_drones=1, physics="sys_id", control="attitude")
    init_time = time.perf_counter() - start

    # Time reset compilation
    start = time.perf_counter()
    sim._reset.lower(sim.data, sim.default_data, None).compile()
    reset_time = time.perf_counter() - start

    # Time step compilation
    start = time.perf_counter()
    sim._step.lower(sim.data, 1).compile()
    step_time = time.perf_counter() - start

    print(f"Simulation startup times | {sim.physics} | {sim.control}")
    print(f"Initialization: {init_time:.2f}s")
    print(f"Reset: {reset_time:.2f}s")
    print(f"Step: {step_time:.2f}s")


if __name__ == "__main__":
    fire.Fire(main)
