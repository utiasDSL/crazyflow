"""Compare splat-rendered FPV gradients with known z-position and yaw gradients.

Splat rendering is differentiable w.r.t. the camera pose. A reference FPV image is rendered at a
baseline pose, and its photometric error is differentiated over small z and yaw offsets. The
resulting gradients match the gradient of the squared pose error in sign and shape. Requires splax
(`pip install "crazyflow[splats]"`) and a CUDA GPU.

Run with `--plot` to plot the results, which requires matplotlib from the `tests` environment.
"""

from __future__ import annotations

import argparse
import logging
import os
from functools import partial
from typing import TYPE_CHECKING

os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import splax
from scipy.spatial.transform import Rotation as R
from splax.io import fetch

from crazyflow.sim import Sim
from crazyflow.sim.sensors.splat import camera_intrinsics
from crazyflow.sim.splat import SPLATS_KEY, attach_splats

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array
    from numpy.typing import NDArray

ASSETS_URL = "https://huggingface.co/datasets/amacati/splats/resolve/main"

BASELINE_POSITION = np.array([5.0, 0.0, 2.0], dtype=np.float32)
BASELINE_QUATERNION = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # xyzw

RESOLUTION = (360, 240)
# Limit to 100 to prevent OOM during testing when JAX has already claimed significant GPU memory
N_SAMPLES = 100
MAX_Z_OFFSET = 0.025
MAX_YAW_DEGREES = 0.1


def _camera_viewmats(
    z_offsets: Array, yaw_angles: Array, camera_position: Array, camera_rotation: R
) -> Array:
    """Create splax world-to-camera matrices for z/yaw drone poses."""
    n_views = z_offsets.shape[0]
    position = jnp.broadcast_to(jnp.asarray(BASELINE_POSITION), (n_views, 3))
    position = position.at[:, 2].add(z_offsets)

    body_rotation = R.from_quat(jnp.asarray(BASELINE_QUATERNION)) * R.from_euler(
        "z", yaw_angles[:, None]
    )
    camera_world_position = position + body_rotation.apply(camera_position)

    # MuJoCo cameras use OpenGL axes. Rotate them 180 degrees around x to obtain the OpenCV axes
    # expected by splax, then invert the complete camera-to-world rotation.
    camera_to_world = body_rotation * camera_rotation * R.from_euler("x", jnp.pi)
    world_to_camera_rotation = camera_to_world.inv().as_matrix()
    world_to_camera_translation = -(world_to_camera_rotation @ camera_world_position[..., None])[
        ..., 0
    ]

    viewmats = jnp.broadcast_to(jnp.eye(4, dtype=z_offsets.dtype), (n_views, 4, 4))
    viewmats = viewmats.at[:, :3, :3].set(world_to_camera_rotation)
    return viewmats.at[:, :3, 3].set(world_to_camera_translation)


def _build_renderer(sim: Sim) -> Callable[[Array, Array], Array]:
    """Build a differentiable FPV renderer parameterized by z and yaw."""
    camera_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv_cam:0")
    if camera_id < 0:
        raise ValueError("Camera 'fpv_cam:0' not found in the model")

    camera_position = jnp.asarray(sim.mj_model.cam_pos[camera_id], dtype=jnp.float32)
    camera_quaternion = jnp.roll(
        jnp.asarray(sim.mj_model.cam_quat[camera_id], dtype=jnp.float32), -1
    )  # wxyz -> xyzw
    camera_rotation = R.from_quat(camera_quaternion)

    f, c = camera_intrinsics(sim.mj_model, camera_id, RESOLUTION)
    splats = sim.data.plugins[SPLATS_KEY]
    background = jnp.zeros(3, dtype=splats.means.dtype)
    image_shape = (RESOLUTION[1], RESOLUTION[0])
    render = partial(
        splax.render, *splats.params, background=background, img_shape=image_shape, f=f, c=c
    )

    def render_views(z_offsets: Array, yaw_angles: Array) -> Array:
        viewmats = _camera_viewmats(z_offsets, yaw_angles, camera_position, camera_rotation)
        return jax.vmap(lambda viewmat: render(viewmat=viewmat)[0])(viewmats)

    return render_views


def plot_results(experiments: tuple[tuple[NDArray, NDArray, NDArray, str, str], ...]) -> None:
    """Plot normalized gradient comparisons for both pose sweeps."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, (x_values, image_gradient, pose_gradient, title, x_label) in zip(
        axes, experiments, strict=True
    ):
        image_gradient = image_gradient / np.max(np.abs(image_gradient))
        pose_gradient = pose_gradient / np.max(np.abs(pose_gradient))

        axis.plot(x_values, image_gradient, label="FPV photometric gradient", linewidth=2)
        axis.plot(x_values, pose_gradient, "--", label="Known pose gradient", linewidth=2)
        axis.axvline(0.0, color="black", linewidth=1, alpha=0.4)
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Normalized gradient")
        axis.set_ylim(-1.1, 1.1)
        axis.grid(alpha=0.25)
        axis.legend()

    fig.suptitle(
        "Differentiable FPV pose gradients\n"
        "Each gradient curve is normalized by its own maximum absolute value"
    )
    plt.show()


def main(plot: bool = False) -> None:
    """Compare photometric and known pose gradients for independent z and yaw sweeps."""
    sim = Sim(n_worlds=1, n_drones=1, device="gpu")
    scene = fetch(f"{ASSETS_URL}/robot_hall.ply")
    attach_splats(sim, scene=scene)
    render = _build_renderer(sim)

    z_offsets = jnp.linspace(-MAX_Z_OFFSET, MAX_Z_OFFSET, N_SAMPLES, dtype=jnp.float32)
    yaw_angles = jnp.deg2rad(
        jnp.linspace(-MAX_YAW_DEGREES, MAX_YAW_DEGREES, N_SAMPLES, dtype=jnp.float32)
    )
    zeros = jnp.zeros_like(z_offsets)
    reference_image = jax.lax.stop_gradient(render(zeros[:1], zeros[:1])[0])

    def photometric_objective(z_offsets: Array, yaw_angles: Array) -> Array:
        images = render(z_offsets, yaw_angles)
        return jnp.mean((images - reference_image) ** 2, axis=(1, 2, 3)).sum()

    pose_gradient = jax.jit(jax.grad(photometric_objective, argnums=(0, 1)))
    z_gradient, _ = pose_gradient(z_offsets, zeros)
    _, yaw_gradient = pose_gradient(zeros, yaw_angles)
    sim.close()

    z_offsets = np.asarray(z_offsets)
    yaw_angles = np.asarray(yaw_angles)
    experiments = (
        (
            100.0 * z_offsets,
            np.asarray(z_gradient),
            z_offsets,
            "Z translation only",
            "z offset from baseline (cm)",
        ),
        (
            np.rad2deg(yaw_angles),
            np.asarray(yaw_gradient),
            yaw_angles,
            "Yaw rotation only",
            "yaw offset from baseline (degrees)",
        ),
    )
    if plot:
        plot_results(experiments)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("jax").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot the results")
    main(plot=parser.parse_args().plot)
