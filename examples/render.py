from collections import deque

import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from scipy.spatial.transform import Rotation as R

from crazyflow.sim import Physics, Sim


def render_traces(viewer: MujocoRenderer, pos: deque, quat: deque):
    """Render traces of the drone trajectories."""
    if len(pos) < 2 or viewer is None:
        return

    n_trace, n_drones = len(pos) - 1, len(pos[0])
    pos, quat = np.array(pos), np.array(quat)
    sizes = np.zeros((n_trace, n_drones, 3))
    rgbas = np.zeros((n_trace, n_drones, 4))
    sizes[..., 2] = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)
    mats = R.from_quat(quat[1:].reshape(-1, 4)).as_matrix().flatten().reshape(n_trace, -1, 9)
    rgbas = np.zeros((n_trace, n_drones, 4))
    np.random.seed(0)  # Ensure consistent colors
    rgbas[..., :3] = np.random.uniform(0, 1, (1, n_drones, 3))
    rgbas[..., 3] = np.linspace(0, 1, n_trace)[:, None]

    for i in range(n_trace):
        for j in range(n_drones):
            viewer.viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=sizes[i, j],
                pos=pos[i][j],
                mat=mats[i, j],
                rgba=rgbas[i, j],
            )


def main():
    """Spawn 25 drones in one world and render each with a trace behind it."""
    n_worlds, n_drones = 1, 25
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=Physics.sys_id, device="cpu")
    fps = 60
    cmd = np.array([[[0.3, 0, 0, 0] for _ in range(sim.n_drones)]])

    pos = deque(maxlen=15)
    quat = deque(maxlen=15)

    for i in range(int(5 * sim.control_freq)):
        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if i % 20 == 0:
            pos.append(sim.data.states.pos[0, :])
            quat.append(sim.data.states.quat[0, :])
        if ((i * fps) % sim.control_freq) < fps:
            render_traces(sim.viewer, pos, quat)
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
