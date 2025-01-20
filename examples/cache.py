"""This example demonstrates persistent JAX function caching with the Sim class.

When JAX functions decorated with @jit are first called, JAX traces and compiles them to XLA. This
compilation is expensive but only needs to happen once. Subsequent calls reuse the compiled version.
However, the cache is not persistent between Python sessions.

The Sim class uses many jitted functions internally, particularly in the step() method which
compiles a chain of physics and control functions. On the first step() call, the entire chain is
compiled.

After the Python session ends, the cached functions get lost. However, we can enable a persistent
cache that is used when the function we are jit-compiling has been compiled before, which
significantly speeds up the jit compile time at the first step.

By enabling caching:
1. The first run compiles and caches all jitted functions persistently
2. The second run loads the cached compiled functions instead of recompiling
3. This gives a significant speedup in initialization and first step times

The cache persists between Python sessions, so compilation only needs to happen once on a machine,
or when the cache directory is deleted.
"""

import time
from pathlib import Path

from crazyflow.sim import Sim
from crazyflow.utils import enable_cache


def main():
    cache_dir = Path("/tmp/jax_cache_test")
    if use_cache := cache_dir.exists():
        print("Cache directory exists. This run will be fast.")
        print(f"\nTo run without cache, delete the directory {cache_dir}.")
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


if __name__ == "__main__":
    main()
