from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import mujoco.mjx as mjx
from jax import Array

from crazyflow.sim.sim import Sim, requires_mujoco_sync


@requires_mujoco_sync
def render_depth(
    sim: Sim, camera: int = 0, resolution: tuple[int, int] = (100, 100), include_drone: bool = False
) -> Array:
    """Render depth images using raycasting.

    Note:
        Code has been adoped from
        https://github.com/Andrew-Luo1/jax_shac/blob/main/vision/2dof_ball.ipynb
    """
    return _render_depth(
        mjx_data=sim.mjx_data,
        mjx_model=sim.mjx_model,
        camera=camera,
        resolution=resolution,
        include_drone=include_drone,
    )


def build_render_depth_fn(
    mjx_model: mjx.Model,
    camera: int = 0,
    resolution: tuple[int, int] = (100, 100),
    geomgroup: tuple[int, ...] = (1, 1, 0, 0, 1, 1, 1, 1),
) -> Callable[[Sim], Array]:
    """Build a depth renderer function for given camera and resolution.

    Compiles the mjx model and rays directly into the rendering function for higher performance. The
    returned function takes a Sim object as input and returns depth images.
    """
    rays = _camera_rays(resolution=resolution, fov_y=jnp.pi / 4)[None, ...]
    ray_fn = jax.jit(
        partial(_render_rays, mjx_model=mjx_model, camera=camera, geomgroup=geomgroup, rays=rays),
        static_argnames=("mjx_model", "camera", "geomgroup", "rays"),
    )

    @requires_mujoco_sync
    def render_depth_fn(sim: Sim) -> Array:
        return ray_fn(mjx_data=sim.mjx_data)

    return render_depth_fn


@jax.jit(static_argnames=("camera", "resolution", "include_drone"))
def _render_depth(
    mjx_data: mjx.Data,
    mjx_model: mjx.Model,
    camera: int,
    resolution: tuple[int, int],
    include_drone: bool = False,
) -> Array:
    """Accelerates the dynamic rendering of depth images."""
    local_rays = _camera_rays(resolution=resolution, fov_y=jnp.pi / 4)[None, ...]
    geomgroup = (1, 1, 1, 0, 1, 1, 1, 1) if include_drone else (1, 1, 0, 0, 1, 1, 1, 1)
    return _render_rays(
        mjx_data=mjx_data, mjx_model=mjx_model, camera=camera, rays=local_rays, geomgroup=geomgroup
    )


def _render_rays(
    mjx_data: mjx.Data, mjx_model: mjx.Model, camera: int, rays: Array, geomgroup: tuple[int, ...]
) -> Array:
    """Render a given ray array using MuJoCo's raycasting."""
    rays = _to_mjx_frame(rays, mjx_data.cam_xmat[:, camera])
    ray_ax = (None, None, None, 0)
    ray = jax.vmap(
        jax.vmap(jax.vmap(partial(mjx.ray, geomgroup=geomgroup), in_axes=ray_ax), in_axes=ray_ax),
        in_axes=(None, 0, 0, 0),
    )
    return ray(mjx_model, mjx_data, mjx_data.cam_xpos[:, camera], rays)[0]


def _to_mjx_frame(x: Array, xmat: Array) -> Array:
    """Transform points to a different frame given its rotation matrix."""
    return (xmat[:, None, None, ...] @ x[..., None])[..., 0]


def _camera_rays(resolution: tuple[int, int] = (100, 100), fov_y: float = jnp.pi / 4) -> Array:
    """Create an array of rays with a given field of view and resolution.

    Args:
        resolution: Image resolution as (width, height).
        fov_y: Vertical field of view in radians.
    """
    image_height = jnp.tan(fov_y / 2) * 2
    image_width = image_height * (resolution[0] / resolution[1])  # Square pixels.
    delta = image_width / (2 * resolution[0])
    x = jnp.linspace(-image_width / 2 + delta, image_width / 2 - delta, resolution[0])
    y = jnp.flip(jnp.linspace(-image_height / 2 + delta, image_height / 2 - delta, resolution[1]))
    X, Y = jnp.meshgrid(x, y)
    rays = jnp.stack([X, Y, -jnp.ones_like(X)], axis=-1)
    return rays / jnp.linalg.norm(rays, axis=-1, keepdims=True)
