"""Gaussian splat camera sensors built on [splax](https://github.com/learnsyslab/splax).

Renders batched RGB(-D) images of the splats attached via
[attach_splats][crazyflow.sim.splat.attach_splats] from any model camera. splax rasterizes with Warp
kernels only, so this module requires the ``splats`` extra and a simulation constructed with
``device="gpu"``.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import splax
from scipy.spatial.transform import Rotation as R

from crazyflow.sim.sim import sync_sim2mjx
from crazyflow.sim.splat import SPLATS_KEY, requires_gpu, requires_splats

if TYPE_CHECKING:
    from typing import Callable, Sequence

    from jax import Array
    from mujoco.mjx import Data, Model  # ty: ignore[unresolved-import]

    from crazyflow.sim.data import SimData
    from crazyflow.sim.sim import Sim


@requires_splats
@requires_gpu
def render_splat_rgb(
    sim: Sim,
    drones: int | Sequence[int] | None = None,
    resolution: tuple[int, int] = (640, 480),
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    exclude_self: bool = False,
    camera_prefix: str = "fpv_cam",
) -> Array:
    """Render RGB images of the attached splats in all worlds.

    Scene gaussians are rendered at their static pose, drone gaussians follow the drone poses.

    Args:
        sim: The simulation to render.
        drones: Drones whose cameras are rendered. ``None`` renders every drone, an int renders one,
            and a sequence renders that subset in order.
        resolution: Image resolution as (width, height).
        background: RGB background color with values in [0, 1].
        exclude_self: Hide each drone's own splat from its camera so a drone does not see itself.
        camera_prefix: Camera name prefix, resolved to ``{camera_prefix}:{drone}`` for each drone.

    Returns:
        RGB images with values in [0, 1] of shape (n_worlds, n_selected, height, width, 3).
    """
    drone_ids = _resolve_drones(sim, drones)
    camera_ids = tuple(_camera_id(sim.mj_model, camera_prefix, d) for d in drone_ids)
    f, c = camera_intrinsics(sim.mj_model, camera_ids[0], resolution)
    return _render(
        sim.data,
        sim.mjx_data,
        sim.mjx_model,
        camera_ids=camera_ids,
        img_shape=(resolution[1], resolution[0]),
        f=f,
        c=c,
        background=background,
        exclude=drone_ids if exclude_self else None,
    )


@requires_splats
@requires_gpu
def render_splat_rgbd(
    sim: Sim,
    drones: int | Sequence[int] | None = None,
    resolution: tuple[int, int] = (640, 480),
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    alpha_threshold: float = 0.5,
    max_range: float = 10.0,
    exclude_self: bool = False,
    camera_prefix: str = "fpv_cam",
) -> Array:
    """Render RGB-D images of the attached splats in all worlds.

    Adds the expected depth of the gaussians as a fourth channel. Pixels whose coverage stays below
    ``alpha_threshold`` are read as empty space and report ``max_range``.

    Args:
        sim: The simulation to render.
        drones: Drones whose cameras are rendered. ``None`` renders every drone, an int renders one,
            and a sequence renders that subset in order.
        resolution: Image resolution as (width, height).
        background: RGB background color with values in [0, 1].
        alpha_threshold: Accumulated coverage a pixel needs before its depth counts as a hit.
        max_range: Sensor range in meters. Reported where nothing is hit, and depth clips to it.
        exclude_self: Hide each drone's own splat from its camera so a drone does not see itself.
        camera_prefix: Camera name prefix, resolved to ``{camera_prefix}:{drone}`` for each drone.

    Returns:
        RGB in [0, 1] followed by depth in meters along the camera's optical axis, of shape
        (n_worlds, n_selected, height, width, 4).
    """
    drone_ids = _resolve_drones(sim, drones)
    camera_ids = tuple(_camera_id(sim.mj_model, camera_prefix, d) for d in drone_ids)
    f, c = camera_intrinsics(sim.mj_model, camera_ids[0], resolution)
    return _render(
        sim.data,
        sim.mjx_data,
        sim.mjx_model,
        camera_ids=camera_ids,
        img_shape=(resolution[1], resolution[0]),
        f=f,
        c=c,
        background=background,
        exclude=drone_ids if exclude_self else None,
        depth=True,
        alpha_threshold=alpha_threshold,
        max_range=max_range,
    )


def _resolve_drones(sim: Sim, drones: int | Sequence[int] | None) -> tuple[int, ...]:
    """Normalize a drone selection to a tuple of drone indices."""
    if isinstance(drones, (int, np.integer)):
        return (int(drones),)
    ids = range(sim.n_drones) if drones is None else drones
    return tuple(int(d) for d in ids)


def _camera_id(mj_model: mujoco.MjModel, prefix: str, drone: int) -> int:
    """Camera index of a drone for the given camera name prefix."""
    name = f"{prefix}:{drone}"
    camera_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if camera_id < 0:
        raise ValueError(f"Camera '{name}' not found in the model")
    return camera_id


@requires_splats
@requires_gpu
def build_render_splat_fn(
    sim: Sim,
    drones: int | Sequence[int] | None = None,
    resolution: tuple[int, int] = (640, 480),
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    exclude_self: bool = False,
    camera_prefix: str = "fpv_cam",
) -> Callable[[SimData], Array]:
    """Build a splat renderer function for a given drone selection, camera prefix, and resolution.

    We bake all arguments into the function to avoid the overhead of flattening static arguments,
    significantly improving performance.
    """
    drone_ids = _resolve_drones(sim, drones)
    cameras_ids = tuple(_camera_id(sim.mj_model, camera_prefix, d) for d in drone_ids)
    f, c = camera_intrinsics(sim.mj_model, cameras_ids[0], resolution)
    return jax.jit(
        partial(
            _render,
            mjx_data=sim.mjx_data,
            mjx_model=sim.mjx_model,
            camera_ids=cameras_ids,
            img_shape=(resolution[1], resolution[0]),
            f=f,
            c=c,
            background=background,
            exclude=drone_ids if exclude_self else None,
        )
    )


@requires_splats
@requires_gpu
def build_render_splat_rgbd_fn(
    sim: Sim,
    drones: int | Sequence[int] | None = None,
    resolution: tuple[int, int] = (640, 480),
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    alpha_threshold: float = 0.5,
    max_range: float = 10.0,
    exclude_self: bool = False,
    camera_prefix: str = "fpv_cam",
) -> Callable[[SimData], Array]:
    """Build a splat RGB-D renderer for a drone selection, camera prefix, and resolution.

    Mirrors [build_render_splat_fn][crazyflow.sim.sensors.splat.build_render_splat_fn].
    """
    drone_ids = _resolve_drones(sim, drones)
    camera_ids = tuple(_camera_id(sim.mj_model, camera_prefix, d) for d in drone_ids)
    f, c = camera_intrinsics(sim.mj_model, camera_ids[0], resolution)
    # The outer jit turns the MuJoCo model and data into baked-in constants rather than arguments
    # flattened on every call, which otherwise dominates the runtime of a single render.
    return jax.jit(
        partial(
            _render,
            mjx_data=sim.mjx_data,
            mjx_model=sim.mjx_model,
            camera_ids=camera_ids,
            img_shape=(resolution[1], resolution[0]),
            f=f,
            c=c,
            background=background,
            exclude=drone_ids if exclude_self else None,
            depth=True,
            alpha_threshold=alpha_threshold,
            max_range=max_range,
        )
    )


@jax.jit
def viewmats(cam_xpos: Array, cam_xmat: Array) -> Array:
    """World-to-camera matrices in OpenCV convention for MuJoCo cameras.

    MuJoCo cameras look along -z with +y up (OpenGL convention), splax expects OpenCV convention
    cameras (+z forward, +y down).

    Args:
        cam_xpos: Camera positions of shape (..., 3).
        cam_xmat: Camera rotation matrices (camera-to-world) of shape (..., 3, 3).

    Returns:
        World-to-camera matrices of shape (..., 4, 4).
    """
    rot_c2w = cam_xmat * jnp.array([1.0, -1.0, -1.0])  # Flip the y and z camera axes (columns)
    rot = jnp.swapaxes(rot_c2w, -1, -2)
    return _homogeneous(rot, -(rot @ cam_xpos[..., None])[..., 0])


# TODO: RigidTransform.from_components produces NaN gradients. Swap once scipy#25767 is released.
def _homogeneous(rot: Array, trans: Array) -> Array:
    """Stack a rotation matrix and a translation into a (..., 4, 4) homogeneous transform."""
    bottom = jnp.broadcast_to(jnp.array([0.0, 0.0, 0.0, 1.0]), (*rot.shape[:-2], 1, 4))
    return jnp.concatenate([jnp.concatenate([rot, trans[..., None]], -1), bottom], -2)


def camera_intrinsics(
    mj_model: mujoco.MjModel, camera_id: int, resolution: tuple[int, int]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Pinhole intrinsics of a model camera for a given image resolution.

    Args:
        mj_model: MuJoCo model containing the camera.
        camera_id: Camera index.
        resolution: Image resolution as (width, height).

    Returns:
        Focal lengths (fx, fy) and principal point (cx, cy) in pixels.
    """
    width, height = resolution
    fov_y = np.deg2rad(mj_model.cam_fovy[camera_id])
    focal = float(height / (2.0 * np.tan(fov_y / 2.0)))
    return (focal, focal), (width / 2.0, height / 2.0)


def _camera_transforms(
    cam_xpos: Array,
    cam_xmat: Array,
    pos: Array,
    quat: Array,
    slices: tuple[tuple[int, int], ...],
    exclude: tuple[int, ...] | None,
) -> tuple[Array, Array | None, int | None]:
    """View matrices, drone transforms, and the axis the transforms are vmapped over per camera."""
    # viewmats needs a single leading batch axis, the render is then nested-vmapped over both axes.
    vm = viewmats(cam_xpos.reshape(-1, 3), cam_xmat.reshape(-1, 3, 3))
    vm = vm.reshape(*cam_xpos.shape[:2], 4, 4)
    if not slices:
        return vm, None, None
    tfs = _homogeneous(R.from_quat(quat).as_matrix(), pos)  # (n_worlds, n_drones)
    if exclude is None:
        return vm, tfs, None
    # Slices are static, so teleport each camera's own drone far below the scene to cull it.
    far = jnp.eye(4, dtype=tfs.dtype).at[2, 3].set(-1e4)
    n_cams = cam_xpos.shape[1]
    tfs = jnp.broadcast_to(tfs[:, None], (tfs.shape[0], n_cams, *tfs.shape[1:]))
    return vm, tfs.at[:, jnp.arange(n_cams), jnp.asarray(exclude)].set(far), 0


@jax.jit(
    static_argnames=(
        "camera_ids",
        "img_shape",
        "f",
        "c",
        "background",
        "exclude",
        "depth",
        "alpha_threshold",
        "max_range",
    )
)
def _render(
    data: SimData,
    mjx_data: Data,
    mjx_model: Model,
    camera_ids: tuple[int, ...],
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    background: tuple[float, float, float],
    exclude: tuple[int, ...] | None,
    depth: bool = False,
    alpha_threshold: float = 0.5,
    max_range: float = 10.0,
) -> Array:
    """Render the splats from every camera with each drone at its current pose.

    Args:
        data: Simulation data holding the drone states and the attached splats.
        mjx_data: MuJoCo data the camera kinematics are computed from.
        mjx_model: MuJoCo model the camera kinematics are computed from.
        camera_ids: Model camera index rendered for each selected drone.
        img_shape: Image size as (height, width).
        f: Focal lengths in pixels.
        c: Principal point in pixels.
        background: RGB background color.
        exclude: Drone culled from each camera, or None to render every drone everywhere.
        depth: If True, append the depth in meters as a fourth channel.
        alpha_threshold: Coverage a pixel needs before its depth counts as a hit. Unused for RGB.
        max_range: Sensor range in meters. Unused for RGB.

    Returns:
        Images of shape (n_worlds, n_cams, height, width, 3), one channel deeper with ``depth``.
    """
    _, mjx_data = sync_sim2mjx(data, mjx_data, mjx_model)  # We need the current camera poses
    splats = data.plugins[SPLATS_KEY]
    slices = splats.slices
    vm, tfs, cam_axis = _camera_transforms(
        mjx_data.cam_xpos[:, camera_ids],
        mjx_data.cam_xmat[:, camera_ids],
        data.states.pos,
        data.states.quat,
        slices,
        exclude,
    )
    render = partial(
        splax.render,
        *splats.params,
        background=jnp.asarray(background, dtype=splats.means.dtype),
        img_shape=img_shape,
        f=f,
        c=c,
        render_depth=depth,
    )
    if tfs is None:
        img, alpha = jax.vmap(jax.vmap(lambda v: render(viewmat=v)))(vm)
    else:
        render_cam = lambda v, t: render(viewmat=v, gaussian_transforms=t, gaussian_slices=slices)  # noqa: E731
        img, alpha = jax.vmap(jax.vmap(render_cam, in_axes=(0, cam_axis)), in_axes=(0, 0))(vm, tfs)
    img = img.at[..., :3].min(1.0)  # Spherical harmonics colors have no upper bound, so we clip
    if not depth:
        return img
    # splax reads 0 depth where nothing was hit, so thin coverage would otherwise come out as a
    # surface right at the camera rather than as empty space.
    metric = jnp.where(alpha < alpha_threshold, max_range, jnp.minimum(img[..., 3], max_range))
    return img.at[..., 3].set(metric)
