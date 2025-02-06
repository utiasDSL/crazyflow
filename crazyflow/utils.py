from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TypeVar

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import BaseRender
from jax import Array


def grid_2d(n: int, spacing: float = 1.0, center: Array | None = None) -> Array:
    """Generate a 2D grid of points."""
    center = jnp.zeros(2) if center is None else center
    N = int(jnp.ceil(jnp.sqrt(n)))
    points = jnp.linspace(-0.5 * spacing * (N - 1), 0.5 * spacing * (N - 1), N)
    x, y = jnp.meshgrid(points, points)
    grid = jnp.stack((x.flatten(), y.flatten()), axis=-1) + center
    order = jnp.argsort(jnp.linalg.norm(grid, axis=-1))
    return grid[order[:n]]


T = TypeVar("T")  # PyTree type


def pytree_replace(tree: T, new_tree: T, mask: Array | None = None) -> T:
    """Overwrite elements of a PyTree with values from another PyTree filtered by a mask.

    The mask indicates which elements of the leaf arrays to overwrite with new values, and which
    ones to leave unchanged.
    """

    def _replace(x: Array, y: Array) -> Array:
        """Replace broadcastable leaves in tree.map."""
        # Do not replace leaves that are not broadcastable
        if x.ndim < 1 or (mask is not None and mask.shape[0] != x.shape[0]):
            return x
        return jnp.where(broadcast_mask(mask, x.shape), y, x)

    return jax.tree.map(_replace, tree, new_tree)


def leaf_replace(tree: T, mask: Array | None = None, **kwargs: dict[str, Array]) -> T:
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


@partial(jax.jit, static_argnames="device")
def to_device(data: Array, device: str) -> Array:
    """Turn an array into a jax array on the specified device."""
    return jnp.array(data, device=device)


def patch_viewer():
    """Patch the viewer._add_marker_to_scene method to work with mujoco>=3.0.

    The current gymnasium implementation is broken for mujoco>=3.0. The method is our only hook into
    the rendering pipeline between `mjv_updateScene`, which deletes all previous markers, and
    `mjr_render`, so if we want to add markers to the scene, we need to do it here. The patch
    ensures that markers are added correctly to the scene.

    If you want to add more markers than the default limit, you can increase the `max_visual_geom`
    parameter in the `Sim` class.
    """

    def _add_marker_to_scene(self: BaseRender, marker: dict):
        """Add a marker to the rendering scene.

        Args:
            self: The renderer instance.
            marker: The marker to add to the scene.
        """
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError(f"Ran out of geoms. maxgeom: {self.scn.maxgeom}")
        geom_type = marker.get("type", mujoco.mjtGeom.mjGEOM_SPHERE)
        size = marker.get("size", np.array([0.01, 0.01, 0.01]))
        pos = marker.get("pos", np.array([0, 0, 0]))
        mat = marker.get("mat", np.eye(3).flatten())
        rgba = marker.get("rgba", np.array([1, 1, 1, 1]))
        mujoco.mjv_initGeom(
            self.scn.geoms[self.scn.ngeom], geom_type, size=size, pos=pos, mat=mat, rgba=rgba
        )
        self.scn.ngeom += 1

    BaseRender._add_marker_to_scene = _add_marker_to_scene


def enable_cache(
    cache_path: Path = Path("/tmp/jax_cache"),
    min_entry_size_bytes: int = -1,
    min_compile_time_secs: int = 0,
    enable_xla_caches: bool = False,
):
    """Enable JAX cache."""
    jax.config.update("jax_compilation_cache_dir", str(cache_path))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", min_entry_size_bytes)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time_secs)
    if enable_xla_caches:
        jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
