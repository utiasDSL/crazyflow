"""Speed up the first simulation step with a persistent JAX compilation cache.

JAX traces and compiles jitted functions to XLA on their first call. This is expensive, but later
calls reuse the compiled version. `Sim` uses many jitted functions internally, and its first `step`
compiles the whole chain of dynamics and control functions, so creating a `Sim` and stepping it
for the first time is slow.

The compiled functions are lost when the Python session ends. `enable_cache` stores them on disk
instead, so later sessions load them rather than recompiling. Compilation then only happens once
per machine, or again after the cache directory is deleted.

Run the script twice to compare. The first run compiles and fills the cache, the second loads it
and deletes it again, so the runs alternate between slow and fast.
"""

import shutil
import time
from pathlib import Path

from crazyflow.sim import Sim
from crazyflow.utils import enable_cache


def main():
    cache_dir = Path("/tmp/jax_cache_test")
    if use_cache := cache_dir.exists():
        print("Cache directory exists. This run will be fast.")
        print("\nTo run without cache, run this script again.")
    else:
        print("Cache directory does not exist. This run will be slow.")
        print("\nTo run with cache, run this script again.")
    enable_cache(cache_path=cache_dir)
    t0 = time.perf_counter()
    sim = Sim()
    t1 = time.perf_counter()
    sim.step()
    t2 = time.perf_counter()
    prefix = "Using cache: " if use_cache else "Not using cache: "
    print(f"{prefix}\n Init: {t1 - t0:.3f}s\n Step: {t2 - t1:.3f}s")
    if use_cache:
        shutil.rmtree(cache_dir)  # Clean up cache so that the next run is slow again
    sim.close()


if __name__ == "__main__":
    main()
