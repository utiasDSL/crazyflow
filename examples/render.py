from collections import deque

import einops
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from crazyflow.constants import GRAVITY, MASS
from crazyflow.sim import Physics, Sim


def main():
    """Spawn 25 drones in one world and render each with a trace behind it."""
    n_worlds, n_drones = 1, 25
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=Physics.sys_id, device="cpu")
    fps = 60
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))
    cmd[..., 0] = MASS * GRAVITY * 1.2

    pos = deque(maxlen=16)
    rot = deque(maxlen=15)

    for i in range(int(5 * sim.control_freq)):
        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if i % 20 == 0:
            pos.append(sim.data.states.pos[0, :])
            if len(pos) > 1:
                rot.append(rotation_matrix_from_points(pos[-2], pos[-1]))
        if ((i * fps) % sim.control_freq) < fps:
            render_traces(sim.viewer, pos, rot)
            sim.render()
    sim.close()


def render_traces(viewer: MujocoRenderer, pos: deque[NDArray], rot: deque[R]):
    """Render traces of the drone trajectories."""
    if len(pos) < 2 or viewer is None:
        return

    n_trace, n_drones = len(rot), len(pos[0])
    pos = np.array(pos)
    sizes = np.zeros((n_trace, n_drones, 3))
    rgbas = np.zeros((n_trace, n_drones, 4))
    sizes[..., 2] = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)
    sizes[..., :2] = np.linspace(1.0, 5.0, n_trace)[:, None, None]
    mats = np.zeros((n_trace, n_drones, 9))
    for i in range(n_trace):
        mats[i, :] = einops.rearrange(rot[i].as_matrix(), "d n m -> d (n m)")
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


def rotation_matrix_from_points(p1: NDArray, p2: NDArray) -> R:
    z_axis = (v := p2 - p1) / np.linalg.norm(v, axis=-1, keepdims=True)
    random_vector = np.random.rand(*z_axis.shape)
    x_axis = (v := np.cross(random_vector, z_axis)) / np.linalg.norm(v, axis=-1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    return R.from_matrix(np.stack((x_axis, y_axis, z_axis), axis=-1))


if __name__ == "__main__":
    main()
