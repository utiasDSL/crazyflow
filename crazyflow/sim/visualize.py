import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from crazyflow.sim import Sim


def draw_line(
    sim: Sim,
    points: NDArray,
    rgba: NDArray | None = None,
    start_size: float = 3.0,
    end_size: float = 3.0,
):
    """Draw a line into the simulation.

    Args:
        sim: The simulation.
        points: An array of [N, 3] points that make up the line.
        rgba: The color of the line.
        start_size: The size of the start of the line.
        end_size: The size of the end of the line.

    Note:
        This function has to be called every time before the sim.render() step.
    """
    assert points.ndim == 2, f"Expected array of [N, 3] points, got Array of shape {points.shape}"
    assert points.shape[-1] == 3, f"Points must be 3D, are {points.shape[-1]}"
    if sim.viewer is None:  # Do not attempt to add markers if viewer is still None
        return
    if sim.max_visual_geom < points.shape[0]:
        raise RuntimeError("Attempted to draw too many lines. Try to increase Sim.max_visual_geom")
    viewer = sim.viewer.viewer
    sizes = np.zeros_like(points)[:-1, :]
    sizes[:, 2] = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    sizes[:, :2] = np.linspace(start_size, end_size, len(sizes))[..., None]
    if rgba is None:
        rgba = np.array([1.0, 0, 0, 1])
    mats = _rotation_matrix_from_points(points[:-1], points[1:]).as_matrix().reshape(-1, 9)
    for i in range(len(points) - 1):
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_LINE, size=sizes[i], pos=points[i], mat=mats[i], rgba=rgba
        )


def draw_points(sim: Sim, points: NDArray, rgba: NDArray | None = None, size: float = 0.01):
    """Draw points into the simulation.

    Args:
        sim: The simulation.
        points: An array of [N, 3] points to draw.
        rgba: The color of the points.
        size: The size of the points.
    """
    if sim.viewer is None:  # Do not attempt to add markers if viewer is still None
        return
    if sim.max_visual_geom < points.shape[0]:
        raise RuntimeError("Attempted to draw too many points. Try to increase Sim.max_visual_geom")
    viewer = sim.viewer.viewer
    if rgba is None:
        rgba = np.array([1.0, 0, 0, 1])
    for i in range(len(points)):
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([size, size, size]),
            pos=points[i],
            rgba=rgba,
        )


def change_material(
    sim: Sim,
    mat_name: str,
    drone_ids: NDArray,
    rgba: NDArray | None = None,
    emission: NDArray | None = None,
):
    """Change the material of specified drones.

    Args:
        sim: The simulation.
        mat_name: The name of the material to change.
        drone_ids: Array of drone indices to modify, shape (n,), dtype=int.
        rgba: The RGBA color to set, should be of shape (n, 4) or (4,) to be auto-broadcasted.
        emission: The emission value of material, should be of shape (n,) or scalar.
    """
    if drone_ids.ndim != 1:
        raise ValueError(f"drone_ids must be 1D array, got shape {drone_ids.shape}")
    if np.any(drone_ids < 0) or np.any(drone_ids >= sim.n_drones):
        raise ValueError(f"drone_ids must be in range [0, {sim.n_drones - 1}], got {drone_ids}")

    if rgba is not None:
        rgba = np.broadcast_to(rgba, (len(drone_ids), 4))

    if emission is not None:
        emission = np.broadcast_to(emission, (len(drone_ids),))

    mj_model = sim.mj_model
    mat_ids = []
    for i, drone_id in enumerate(drone_ids):
        full_mat_name = f"{mat_name}:{drone_id}"
        mat_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_MATERIAL, full_mat_name)
        if mat_id < 0:
            raise ValueError(f"Material '{full_mat_name}' not found in MuJoCo model.")
        mat_ids.append(mat_id)

    if rgba is not None:
        mj_model.mat_rgba[mat_ids] = rgba

    if emission is not None:
        mj_model.mat_emission[mat_ids] = emission


def _rotation_matrix_from_points(p1: NDArray, p2: NDArray) -> R:
    """Generate rotation matrices that align their z-axis to p2-p1."""
    p1, p2 = p1.copy(), p2.copy()  # Make sure we don't modify the original arrays
    p2[np.linalg.norm(p2 - p1, axis=-1) < 1e-6] += 1e-6
    z_axis = (v := p2 - p1) / np.linalg.norm(v, axis=-1, keepdims=True)
    random_vector = np.random.rand(*z_axis.shape)
    x_axis = (v := np.cross(random_vector, z_axis)) / np.linalg.norm(v, axis=-1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    return R.from_matrix(np.stack((x_axis, y_axis, z_axis), axis=-1))
