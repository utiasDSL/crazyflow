import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["SCIPY_ARRAY_API"] = "1"
# We need multiple devices for sharding tests
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import pytest

# The cache dir is per-user. A shared dir like /tmp/jax_cache breaks on multi-user machines, since
# jax hard-fails on GPU autotune cache writes when another user owns the directory.
jax.config.update("jax_compilation_cache_dir", f"/tmp/jax_cache-{os.getuid()}")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# Do not enable XLA caches, crashes PyTest
# jax.config.update("jax_persistent_cache_enable_xla_caches", "all")


def available_backends() -> list[str]:
    """Return list of available JAX backends."""
    backends = []
    for backend in ["tpu", "gpu", "cpu"]:
        try:
            jax.devices(backend)
        except RuntimeError:
            pass
        else:
            backends.append(backend)
    return backends


@pytest.fixture
def device() -> str:
    """Return GPU device if available, otherwise CPU."""
    if "gpu" in available_backends():
        return "gpu"
    return "cpu"


# Marker for conditional skip in headless environments
skip_if_headless = pytest.mark.skipif(
    os.environ.get("DISPLAY") is None,
    reason="DISPLAY is not set, skipping test in headless environment",
)
