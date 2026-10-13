"""Render the simulation as gaussian splats in a web-based viewer.

`SplatViewer` serves a web viewer at `http://localhost:8080`. The splats are streamed to the
browser once, after which `update` only pushes the drone poses, so the viewer runs in real time on
any device. Requires splax (`pip install "crazyflow[splats]"`).
"""

from __future__ import annotations

import logging
import os
import time

os.environ["SCIPY_ARRAY_API"] = "1"

import numpy as np
from scipy.spatial.transform import Rotation as R
from splax.io import fetch

from crazyflow.sim import Sim
from crazyflow.sim.splat import SplatViewer, attach_splats

logger = logging.getLogger(__name__)
ASSETS_URL = "https://huggingface.co/datasets/amacati/splats/resolve/main"


def trajectory(t: float) -> np.ndarray:
    cmd = np.zeros((1, 1, 16))
    cmd[..., 9:13] = R.from_euler("z", 0.0).as_quat()
    cmd[..., :3] = [0.5 * (np.cos(t) - 1), 0.5 * np.sin(t), 1.0 + 0.2 * np.sin(0.5 * t)]
    return cmd


def main():
    sim = Sim(control="state")
    scene = fetch(f"{ASSETS_URL}/robot_hall.ply")
    drone = fetch(f"{ASSETS_URL}/{sim.drone}.ply")
    attach_splats(sim, scene=scene, drone=drone)
    viewer = SplatViewer(sim)

    for i in range(500):
        t_start = time.perf_counter()
        sim.state_control(trajectory(i / sim.control_freq))
        sim.step(sim.freq // sim.control_freq)
        viewer.update(sim)
        time.sleep(max(0.0, 1 / sim.control_freq - (time.perf_counter() - t_start)))

    viewer.close()
    sim.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("jax").setLevel(logging.WARNING)
    main()
