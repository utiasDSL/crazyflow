import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import pytest

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
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
