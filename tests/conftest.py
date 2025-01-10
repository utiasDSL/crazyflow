import jax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# Do not enable XLA caches, crashes PyTest
# jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
