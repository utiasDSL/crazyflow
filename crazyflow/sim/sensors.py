from functools import partial

import jax
import jax.numpy as jnp
import mujoco.mjx as mjx
import numpy as np
from jax import Array

from crazyflow.sim.sim import Sim, requires_mujoco_sync


@requires_mujoco_sync
def render_depth(sim: Sim, camera: int = 0, resolution: tuple[int, int] = (100, 100)) -> Array:
    """Render depth images using raycasting.

    Note:
        Code has been adoped from
        https://github.com/Andrew-Luo1/jax_shac/blob/main/vision/2dof_ball.ipynb
    """
    local_rays = camera_rays(resolution=resolution, fov_y=np.pi / 4)[None, ...]
    rays = to_mjx_frame(local_rays, sim.mjx_data.cam_xmat[:, camera])
    dist, _ = ray_fn(sim.mjx_model, sim.mjx_data, sim.mjx_data.cam_xpos[:, camera], rays)
    return dist


@jax.jit
def to_mjx_frame(x: Array, xmat: Array) -> Array:
    """Transform points to a different frame given its rotation matrix."""
    return (xmat[:, None, None, ...] @ x[..., None])[..., 0]


_geomgroup = (1, 1, 1, 0, 1, 1, 1, 1)  # Exclude collision geoms
ray_fn = jax.jit(
    jax.vmap(
        jax.vmap(
            jax.vmap(partial(mjx.ray, geomgroup=_geomgroup), in_axes=(None, None, None, 0)),
            in_axes=(None, None, None, 0),
        ),
        in_axes=(None, 0, 0, 0),
    )
)


@partial(jax.jit, static_argnames=("resolution", "fov_y"))
def camera_rays(resolution: tuple[int, int] = (100, 100), fov_y: float = jnp.pi / 4) -> Array:
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
