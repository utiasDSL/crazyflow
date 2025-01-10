import jax
import pytest

from crazyflow.utils import enable_cache


@pytest.mark.unit
@pytest.mark.parametrize("enable_xla", [True, False])
def test_enable_cache(enable_xla: bool):
    """Test that enable_cache correctly sets JAX cache configuration."""
    # Store original config values
    orig_cache_dir = jax.config.values.get("jax_compilation_cache_dir", None)
    orig_min_size = jax.config.values.get("jax_persistent_cache_min_entry_size_bytes", None)
    orig_min_time = jax.config.values.get("jax_persistent_cache_min_compile_time_secs", None)
    orig_xla = jax.config.values.get("jax_persistent_cache_enable_xla_caches", None)

    try:
        cache_path = "/tmp/jax_cache"
        min_size = 1000
        min_time = 2

        enable_cache(
            cache_path=cache_path,
            min_entry_size_bytes=min_size,
            min_compile_time_secs=min_time,
            enable_xla_caches=enable_xla,
        )

        assert cache_path == jax.config.jax_compilation_cache_dir, "Cache path not set correctly"
        assert (
            min_size == jax.config.jax_persistent_cache_min_entry_size_bytes
        ), "Min size not set correctly"
        assert (
            min_time == jax.config.jax_persistent_cache_min_compile_time_secs
        ), "Min time not set correctly"
        expected_xla = "all" if enable_xla else orig_xla
        assert (
            expected_xla == jax.config.jax_persistent_cache_enable_xla_caches
        ), "XLA caches not set correctly"

    finally:
        if orig_cache_dir is not None:
            jax.config.update("jax_compilation_cache_dir", orig_cache_dir)
        if orig_min_size is not None:
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", orig_min_size)
        if orig_min_time is not None:
            jax.config.update("jax_persistent_cache_min_compile_time_secs", orig_min_time)
        if orig_xla is not None:
            jax.config.update("jax_persistent_cache_enable_xla_caches", orig_xla)
