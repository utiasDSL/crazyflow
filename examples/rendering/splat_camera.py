"""Render the FPV cameras of two drones looking at each other with gaussian splats.

`render_splat_rgb` renders the FPV camera of every drone in every world in a single batched call.
Two drones circle the hall center while facing each other. Requires splax
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
from crazyflow.sim.sensors.splat import render_splat_rgb
from crazyflow.sim.splat import attach_splats

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

ASSETS_URL = "https://huggingface.co/datasets/amacati/splats/resolve/main"

RADIUS = 0.5
HEIGHT = 1.0


def trajectory(t: float, n_worlds: int, n_drones: int) -> np.ndarray:
    """Circle both drones around the center, each yawed to look across at its partner."""
    angle = t + (2 * np.pi / n_drones) * np.arange(n_drones)
    cmd = np.zeros((n_worlds, n_drones, 16))
    cmd[..., 0] = RADIUS * np.cos(angle)
    cmd[..., 1] = RADIUS * np.sin(angle)
    cmd[..., 2] = HEIGHT
    yaw = angle + np.pi  # yaw faces the opposite point on the circle, where the partner is
    cmd[..., 9:13] = R.from_euler("z", yaw[:, None]).as_quat()
    return cmd


def main(plot: bool = False):
    """Render the splat camera views of both drones in both worlds."""
    sim = Sim(n_worlds=2, n_drones=2, control="state", device="gpu")
    scene = fetch(f"{ASSETS_URL}/robot_hall.ply")
    drone = fetch(f"{ASSETS_URL}/{sim.drone}.ply")
    attach_splats(sim, scene=scene, drone=drone)

    angle = (2 * np.pi / sim.n_drones) * np.arange(sim.n_drones)
    x, y, z = RADIUS * np.cos(angle), RADIUS * np.sin(angle), np.full_like(angle, HEIGHT)
    pos = np.stack([x, y, z], axis=-1)
    quat = R.from_euler("z", (angle + np.pi)[:, None]).as_quat()
    states = sim.data.states.replace(pos=sim.data.states.pos.at[...].set(pos))
    states = states.replace(quat=sim.data.states.quat.at[...].set(quat))
    sim.data = sim.data.replace(states=states)

    duration = 6
    fps = 30
    n_frames = int(duration * fps)
    resolution = (320, 240)

    update_plot = plot_views(sim.n_worlds, sim.n_drones, resolution) if plot else None
    for _ in range(n_frames):
        t = sim.data.core.steps[0, 0] / sim.freq
        sim.state_control(trajectory(t, sim.n_worlds, sim.n_drones))
        sim.step(sim.freq // fps)
        # Render each drone's fpv across all worlds, hiding each drone from its own view.
        views = np.asarray(render_splat_rgb(sim, resolution=resolution))
        if plot:
            update_plot(views)
    sim.close()


def plot_views(
    n_worlds: int, n_drones: int, resolution: tuple[int, int]
) -> Callable[[NDArray], None]:
    """Open a window for the camera views and return a function that draws the next ones."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_worlds, n_drones, figsize=(10, 7))
    img = np.zeros((resolution[1], resolution[0], 3), dtype=np.float32)
    ims = np.empty((n_worlds, n_drones), dtype=object)
    for w in range(n_worlds):
        for d in range(n_drones):
            ax = axes[w, d]
            ims[w, d] = ax.imshow(img)
            ax.set_title(f"World {w}, drone {d} FPV")
            ax.axis("off")
    fig.tight_layout()
    plt.show(block=False)

    def update(views: NDArray) -> None:
        for w in range(n_worlds):
            for d in range(n_drones):
                ims[w, d].set_data(views[w, d])
        fig.canvas.draw_idle()
        plt.pause(1e-12)

    return update


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("jax").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot the results")
    main(plot=parser.parse_args().plot)
