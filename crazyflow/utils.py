from __future__ import annotations

import inspect
import math
import os
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import PyTreeNode
from jax import Array

if TYPE_CHECKING:
    from types import FunctionType, ModuleType

CORE_NDIM_KEY = "core_ndim"
"""Metadata key for the number of dimensions a dataclass field has without any batch axes."""


def grid_2d(n: int, spacing: float = 1.0, center: Array | None = None) -> Array:
    """Generate a 2D grid of points."""
    assert n > 0, "Number of points must be positive"
    center = jnp.zeros(2) if center is None else center
    N = math.isqrt(n - 1) + 1
    points = jnp.linspace(-0.5 * spacing * (N - 1), 0.5 * spacing * (N - 1), N)
    x, y = jnp.meshgrid(points, points)
    grid = jnp.stack((x.flatten(), y.flatten()), axis=-1) + center
    order = jnp.argsort(jnp.linalg.norm(grid, axis=-1))
    return grid[order[:n]]


T = TypeVar("T")
TNode = TypeVar("TNode", bound=PyTreeNode)
P = ParamSpec("P")
R = TypeVar("R")


def world_mask(tree: T) -> T:
    """Flag the arrays of a PyTree that are indexed by world.

    Fields declare their number of dimensions without batch axes with the CORE_NDIM_KEY metadata.
    An array carries a world axis when its ndim exceeds that. Arrays without the metadata are
    treated as not batched.

    Args:
        tree: PyTree of arrays and structs to classify.

    Returns:
        A PyTree of booleans matching the input.
    """
    return _flag(tree)


def _flag(node: Any) -> Any:
    """Flag the leaves of a node, recursing through the declared structure of the data."""
    if not _declares_fields(node):
        # Any other node. Nested structs are handed back to us via is_leaf so that we can recurse
        return jax.tree.map(
            lambda x: _flag(x) if _declares_fields(x) else False, node, is_leaf=_declares_fields
        )
    flags = {}
    for f in (f for f in fields(node) if f.metadata.get("pytree_node", True)):
        value, core_ndim = getattr(node, f.name), f.metadata.get(CORE_NDIM_KEY)
        if core_ndim is None:  # Structs and containers classify their own contents
            flags[f.name] = _flag(value)
        elif _declares_fields(value):
            name = f"{type(node).__name__}.{f.name}"
            raise ValueError(f"{name} holds a struct, which declares its own fields")
        else:
            flags[f.name] = jax.tree.map(lambda x, core=core_ndim: x.ndim > core, value)
    return node.replace(**flags)


def _declares_fields(node: Any) -> bool:
    """Check whether a node is a dataclass that jax traverses."""
    return is_dataclass(node) and not jax.tree_util.all_leaves([node])


def pytree_replace(tree: T, new_tree: T, batched: T, mask: Array | None = None) -> T:
    """Overwrite batched leaves of a PyTree with values from another PyTree filtered by a mask.

    Args:
        tree: PyTree to overwrite.
        new_tree: PyTree to take the new values from.
        batched: PyTree of booleans flagging the leaves that carry the batch axis.
        mask: Boolean array matching the leading axis of the batched leaves.
    """

    def _replace(x: Array, y: Array, batched: bool) -> Array:
        """Replace batched leaves in tree.map."""
        if not batched:
            return x
        return jnp.where(broadcast_mask(mask, x.shape), y, x)

    return jax.tree.map(_replace, tree, new_tree, batched)


def leaf_replace(tree: TNode, mask: Array | None = None, **kwargs: Array) -> TNode:
    """Replace elements of a PyTree with the given keyword arguments.

    If a mask is provided, the replacement is applied only to the elements indicated by the mask.

    Args:
        tree: The PyTree to be modified.
        mask: Boolean array matching the first dimension of all kwargs entries in tree.
        kwargs: Leaf names and their replacement values.
    """
    replace = {
        k: jnp.where(broadcast_mask(mask, v.shape), v, getattr(tree, k)) for k, v in kwargs.items()
    }
    return tree.replace(**replace)


def broadcast_mask(mask: Array | None, shape: tuple[int, ...]) -> Array:
    """Broadcast a mask to match the shape of the data."""
    mask = jnp.ones(shape, dtype=bool) if mask is None else mask
    return mask.reshape(*mask.shape, *[1] * (len(shape) - mask.ndim))


def enable_cache(
    cache_path: Path | None = None,
    min_entry_size_bytes: int = -1,
    min_compile_time_secs: int = 0,
    enable_xla_caches: bool = False,
):
    """Enable JAX cache with the requested settings.

    Cache path is user-dependent to avoid permission issues on multi-user machines.
    """
    if cache_path is None:
        cache_path = Path(f"/tmp/jax_cache-{os.getuid()}")
    jax.config.update("jax_compilation_cache_dir", str(cache_path))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", min_entry_size_bytes)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time_secs)
    if enable_xla_caches:
        jax.config.update("jax_persistent_cache_enable_xla_caches", "all")


def parametrize(
    fn: Callable[P, R],
    drone: str,
    load_params: Callable[..., dict],
    xp: ModuleType | None = None,
    device: str | None = None,
) -> Callable[P, R]:
    """Parametrize a function with the default parameters for a drone.

    Args:
        fn: The function to parametrize.
        drone: The drone to use.
        load_params: The function to load the parameters for the given drone.
        xp: The array API module to use. If not provided, numpy is used.
        device: The device to use. If none, the device is inferred from the xp module.

    Returns:
        The parametrized function with all keyword only arguments filled in.
    """
    params = load_params(fn, drone, xp=xp, device=device)
    xp = np if xp is None else xp
    fn_params = inspect.signature(fn).parameters
    fn_kwargs = {k for k, v in fn_params.items() if v.kind == inspect.Parameter.KEYWORD_ONLY}
    kwargs = {k: xp.asarray(v, device=device) for k, v in params.items() if k in fn_kwargs}
    return partial(fn, **kwargs)


def filter_to_signature(params: dict, fn: FunctionType) -> dict:
    """Keep only the params accepted by ``fn``.

    Asserts that every keyword-only parameter of ``fn`` (the injectable params, as opposed to the
    positional runtime inputs) is present in ``params``.
    """
    sig = inspect.signature(fn).parameters
    filtered = {k: v for k, v in params.items() if k in sig}
    required = {k for k, p in sig.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
    missing = required - filtered.keys()
    assert not missing, f"Missing parameters for {fn.__name__}: {missing}"
    return filtered


def to_xp(*args: Any, xp: ModuleType | None = None, device: Any = None) -> Any:
    """Convert arrays, dicts etc recursively to the ``xp`` namespace and device."""
    xp = np if xp is None else xp
    match args:
        case [Mapping() as m]:
            return {k: to_xp(v, xp=xp, device=device) for k, v in m.items()}
        case [single]:
            return xp.asarray(single, device=device)
        case _:
            return tuple(to_xp(a, xp=xp, device=device) for a in args)
