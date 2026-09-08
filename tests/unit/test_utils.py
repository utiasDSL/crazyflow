from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import flax
import jax
import jax.numpy as jnp
import pytest
from flax.struct import field

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.utils import CORE_NDIM_KEY, enable_cache, world_mask

if TYPE_CHECKING:
    from jax import Array


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
        assert min_size == jax.config.jax_persistent_cache_min_entry_size_bytes, (
            "Min size not set correctly"
        )
        assert min_time == jax.config.jax_persistent_cache_min_compile_time_secs, (
            "Min time not set correctly"
        )
        expected_xla = "all" if enable_xla else orig_xla
        assert expected_xla == jax.config.jax_persistent_cache_enable_xla_caches, (
            "XLA caches not set correctly"
        )

    finally:
        if orig_cache_dir is not None:
            jax.config.update("jax_compilation_cache_dir", orig_cache_dir)
        if orig_min_size is not None:
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", orig_min_size)
        if orig_min_time is not None:
            jax.config.update("jax_persistent_cache_min_compile_time_secs", orig_min_time)
        if orig_xla is not None:
            jax.config.update("jax_persistent_cache_enable_xla_caches", orig_xla)


@pytest.mark.unit
def test_world_mask():
    # Check that the mask correctly handles cases that previously failed under shape-based detection
    mask = world_mask(Sim(n_worlds=3, control=Control.attitude).data)
    assert mask.states.pos
    assert mask.states_deriv.acc
    assert mask.core.steps
    assert not mask.params.mass
    assert mask.controls.attitude.cmd
    assert not mask.params.gravity_vec
    assert not mask.params.drag_matrix
    assert not mask.core.drone_mocap_ids
    assert not mask.controls.attitude.params["kR"]


@pytest.mark.unit
def test_world_mask_covers_batched_arrays():
    # Make a shape-based check that we catch all world-batched arrays using a unique world length
    sim = Sim(n_worlds=7, control=Control.attitude)
    paths, leaves = jax.tree.flatten_with_path(sim.data)[0], jax.tree.leaves(world_mask(sim.data))
    for (path, x), batched in zip(paths, leaves):
        world_axis = x.ndim >= 2 and x.shape[0] == 7
        assert batched == world_axis, f"{jax.tree_util.keystr(path)}: {x.shape=} is {batched=}"


@pytest.mark.unit
def test_world_mask_user_dataclass():
    @flax.struct.dataclass
    class PluginState:
        buffer: Array = field(metadata={CORE_NDIM_KEY: 1})
        constant: Array = None

    state = PluginState(buffer=jnp.zeros((3, 2)), constant=jnp.zeros(3))
    mask = world_mask({"state": state, "array": jnp.zeros((3, 2))})
    assert mask["state"].buffer
    assert not mask["state"].constant
    assert not mask["array"]  # Arrays that no struct declares are shared


@pytest.mark.unit
def test_world_mask_matches_structure():
    # The mask has to match the tree leaf for leaf, whatever container holds the arrays
    @flax.struct.dataclass
    class PluginState:  # With world data
        buffer: Array = field(metadata={CORE_NDIM_KEY: 1})

    @dataclass
    class Opaque:  # State that is treated as leaf
        a: int

    array = jnp.zeros((3, 2))
    tree = {
        "containers": (array, [array], {"x": array}),
        "nested": (PluginState(buffer=array),),
        "opaque": Opaque(1),
    }
    mask = world_mask(tree)
    assert jax.tree.structure(mask) == jax.tree.structure(tree)
    assert mask["nested"][0].buffer  # A declaration survives inside a container


@pytest.mark.unit
def test_world_mask_rejects_tagged_struct():
    # A struct declares its own fields, so a core ndim on a field holding a struct is an error.
    @flax.struct.dataclass
    class TaggedStruct:
        nested: Any = field(metadata={CORE_NDIM_KEY: 1})

    with pytest.raises(ValueError, match="declares its own fields"):
        world_mask(TaggedStruct(nested=TaggedStruct(nested=None)))


@pytest.mark.unit
def test_world_mask_infers_randomized_parameter():
    # A parameter that is shared by default declares itself once it gains a world axis
    sim = Sim(n_worlds=3)
    assert not world_mask(sim.data).params.drag_matrix  # (3, 3), shared
    drag = jnp.broadcast_to(sim.data.params.drag_matrix, (3, 1, 3, 3))
    sim.data = sim.data.replace(params=sim.data.params.replace(drag_matrix=drag))
    mask = world_mask(sim.data)
    assert mask.params.drag_matrix  # (3, 1, 3, 3), now world-indexed
    assert not mask.params.gravity_vec  # Siblings are unaffected
