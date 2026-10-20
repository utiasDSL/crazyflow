"""Change the viewer camera with a camera configuration.

The script renders the same flight twice, first with the default camera and then with a camera
that looks at the origin from a fixed distance, elevation, and azimuth.
"""

import os

os.environ["SCIPY_ARRAY_API"] = "1"

import numpy as np
from scipy.spatial.transform import Rotation as R

from crazyflow.sim import Sim

cam_config = {"distance": 0.8, "elevation": -45.0, "azimuth": -135.0, "lookat": [0.0, 0.0, 0.0]}


def main(cam_config: dict | None = None):
    sim = Sim(control="state")
    sim.reset()

    duration = 5.0
    fps = 60

    cmd = np.zeros((sim.n_worlds, sim.n_drones, 16))
    cmd[..., 9:13] = R.from_euler("z", 0.0).as_quat()
    cmd[..., :3] = 0.2

    for i in range(int(duration * sim.control_freq)):
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render(cam_config=cam_config)
    sim.close()


if __name__ == "__main__":
    main(cam_config=None)
    main(cam_config=cam_config)
