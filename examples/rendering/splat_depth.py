"""Example where a drone flies a lap through a hall and renders gaussian splat depth images.

`build_render_splat_rgbd_fn` compiles an RGB-D renderer that is a pure function of `sim.data`. The
depth is measured along the optical axis and clipped to `max_range`. Requires splax
(`pip install "crazyflow[splats]"`) and a CUDA GPU.

Run with `--plot` to plot the results, which requires matplotlib from the `tests` environment.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import TYPE_CHECKING

os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
from scipy.spatial.transform import Rotation as R
from splax.io import fetch

from crazyflow.sim import Sim
from crazyflow.sim.sensors.splat import build_render_splat_rgbd_fn
from crazyflow.sim.splat import attach_splats

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

ASSETS_URL = "https://huggingface.co/datasets/amacati/splats/resolve/main"

TRAJ_CENTER = np.array([0.0, 0.0, 1.5])
TRAJ_SIZE = np.array([9.0, 3.5, 0.0])
MAX_RANGE = 12.0
RESOLUTION = (320, 240)
DURATION = 40
FPS = 30


def trajectory(t: float) -> np.ndarray:
    """Return an elliptical lap with the yaw along the tangent."""
    omega = 2.0 * np.pi / DURATION
    phi = omega * t
    cmd = np.zeros((1, 1, 16))
    cmd[..., 0:3] = TRAJ_CENTER + TRAJ_SIZE * np.array([np.cos(phi), np.sin(phi), 0.0])
    yaw = np.arctan2(TRAJ_SIZE[1] * np.cos(phi), -TRAJ_SIZE[0] * np.sin(phi))
    cmd[..., 9:13] = R.from_euler("z", yaw).as_quat()
    return cmd


def main(plot: bool = False):
    """Fly the lap and render the depth channel of the drone's fpv camera."""
    sim = Sim(control="state", device="gpu")
    scene = fetch(f"{ASSETS_URL}/robot_hall.ply")
    drone = fetch(f"{ASSETS_URL}/{sim.drone}.ply")
    attach_splats(sim, scene=scene, drone=drone)
    render = build_render_splat_rgbd_fn(sim, drones=0, resolution=RESOLUTION, max_range=MAX_RANGE)

    # Start on the lap so the controller does not have to fly in from the origin first
    cmd = trajectory(0.0)
    states = sim.data.states.replace(pos=sim.data.states.pos.at[...].set(cmd[..., :3]))
    states = states.replace(quat=states.quat.at[...].set(cmd[..., 9:13]))
    sim.data = sim.data.replace(states=states)

    update_plot = plot_depth() if plot else None
    for i in range(DURATION * FPS):
        sim.state_control(trajectory(i / FPS))
        sim.step(sim.freq // FPS)
        depth = np.asarray(render(sim.data))[0, 0, ..., 3]
        if plot:
            update_plot(depth)
    sim.close()


def plot_depth() -> Callable[[NDArray], None]:
    """Open a window for the depth images and return a function that draws the next one."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(np.zeros(RESOLUTION[::-1]), cmap="turbo_r", vmin=0.0, vmax=MAX_RANGE)
    ax.set_title("Splat depth camera, fpv")
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="depth along the optical axis (m)")
    fig.tight_layout()
    plt.show(block=False)

    def update(depth: NDArray) -> None:
        im.set_data(depth)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    return update


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("jax").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot the results")
    main(plot=parser.parse_args().plot)
