"""Gaussian splat support built on [splax](https://github.com/learnsyslab/splax).

This module owns the splat plugin state and its visualization.
[attach_splats][crazyflow.sim.splat.attach_splats] loads 3D gaussian splatting ``.ply`` files and
stores them in the simulation's plugin data, and [SplatViewer][crazyflow.sim.splat.SplatViewer]
streams the splats to a web-based viewer.

The batched RGB camera sensor that renders splats lives in [crazyflow.sim.sensors.splat][].

Splat files must be aligned to the simulation frames. The viewer renders in the browser and works on
any device. Camera sensors rasterize on the GPU only.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable, Concatenate, ParamSpec, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import splax
from flax.struct import PyTreeNode, field
from splax.viewer import Viewer

from crazyflow.sim.sim import requires_mujoco_sync
from crazyflow.utils import CORE_NDIM_KEY

if TYPE_CHECKING:
    from pathlib import Path

    from jax import Array
    from numpy.typing import NDArray

    from crazyflow.sim.sim import Sim

Params = ParamSpec("Params")
Return = TypeVar("Return")

SPLATS_KEY = "splats"
"""Key of the [SplatData][crazyflow.sim.splat.SplatData] in ``sim.data.plugins``."""


class SplatData(PyTreeNode):
    """Gaussian splat plugin data.

    Gaussians are stored in a single array as ``[scene | drone:0 | ... | drone:n-1]``. Scene
    gaussians are not part of any slice and remain static. Drone slices follow the drone's pose.
    """

    means: Array = field(metadata={CORE_NDIM_KEY: 2})
    """Gaussian centers, shape (N, 3)."""
    log_scales: Array = field(metadata={CORE_NDIM_KEY: 2})
    """Log of the per-axis scales, shape (N, 3)."""
    quats: Array = field(metadata={CORE_NDIM_KEY: 2})
    """Rotations as wxyz quaternions, shape (N, 4)."""
    sh_colors: Array = field(metadata={CORE_NDIM_KEY: 3})
    """Spherical harmonics color coefficients, shape (N, K, 3)."""
    logit_opacities: Array = field(metadata={CORE_NDIM_KEY: 1})
    """Opacity logits, shape (N,)."""
    slices: tuple[tuple[int, int], ...] = field(pytree_node=False)
    """Start and stop index of each drone's gaussians."""

    @property
    def params(self) -> tuple[Array, Array, Array, Array, Array]:
        """The gaussian parameter arrays in splax's argument order."""
        return self.means, self.log_scales, self.quats, self.sh_colors, self.logit_opacities


def requires_splats(
    fn: Callable[Concatenate[Sim, Params], Return],
) -> Callable[Concatenate[Sim, Params], Return]:
    """Decorator to ensure that the simulation has splats attached."""

    @wraps(fn)
    def wrapper(sim: Sim, *args: Params.args, **kwargs: Params.kwargs) -> Return:
        if SPLATS_KEY not in sim.data.plugins:
            raise RuntimeError("No splats attached to this simulation, call attach_splats first")
        return fn(sim, *args, **kwargs)

    return wrapper


def requires_gpu(
    fn: Callable[Concatenate[Sim, Params], Return],
) -> Callable[Concatenate[Sim, Params], Return]:
    """Decorator to ensure that the simulation is running on the GPU."""

    @wraps(fn)
    def wrapper(sim: Sim, *args: Params.args, **kwargs: Params.kwargs) -> Return:
        if sim.device.platform != "gpu":
            raise RuntimeError("Gaussian splatting requires running on the GPU.")
        return fn(sim, *args, **kwargs)

    return wrapper


def attach_splats(sim: Sim, scene: Path | None = None, drone: Path | None = None):
    """Load gaussian splat ``.ply`` files and attach them to the simulation.

    The drone splat is replicated once per drone so that each drone has its own slice of gaussians.
    The splat data is inserted into ``sim.data.plugins`` and the default data is rebuilt so it
    survives resets.

    Args:
        sim: The simulation to attach the splats to.
        scene: Path to a static scene splat, aligned to the MuJoCo world frame.
        drone: Path to a drone splat in the drone body frame, replicated for each drone.
    """
    if scene is None and drone is None:
        raise ValueError("At least one of scene or drone must be provided")
    parts, n_splats, slices = [], 0, ()
    if scene is not None:
        scene_arrays = splax.io.load_ply(scene)
        n_splats = scene_arrays[0].shape[0]
        parts.append(scene_arrays)
    if drone is not None:
        drone_arrays = splax.io.load_ply(drone)
        n_drone_splats = drone_arrays[0].shape[0]
        parts.extend([drone_arrays] * sim.n_drones)
        starts = [n_splats + i * n_drone_splats for i in range(sim.n_drones)]
        slices = tuple((start, start + n_drone_splats) for start in starts)
    coefficients = {part[3].shape[1] for part in parts}
    assert len(coefficients) == 1, f"Splats must share their SH degree, got {coefficients=}"
    arrays = [jnp.concatenate(x, axis=0) for x in zip(*parts)]
    splats = jax.device_put(SplatData(*arrays, slices=slices), sim.device)
    sim.data = sim.data.replace(plugins=sim.data.plugins | {SPLATS_KEY: splats})
    sim.build_default_data()


class SplatViewer:
    """Web-based gaussian splat viewer.

    Starts a ``splax.viewer.Viewer`` (viser web server) and uploads all attached splats once.
    [update][crazyflow.sim.splat.SplatViewer.update] then only pushes the current drone poses, so
    the viewer runs at real-time rates on any device. The viewer is owned by its creator and is
    independent of ``sim.render()``.

    Args:
        sim: The simulation to visualize. Requires
            [attach_splats][crazyflow.sim.splat.attach_splats] to have been called.
        port: Port of the web server.
    """

    def __init__(self, sim: Sim, port: int = 8080):
        if SPLATS_KEY not in sim.data.plugins:
            raise RuntimeError("No splats attached to this simulation, call attach_splats first")
        splats = sim.data.plugins[SPLATS_KEY]
        self.viewer = Viewer(port=port)
        n_splats = splats.slices[0][0] if splats.slices else splats.means.shape[0]
        if n_splats > 0:
            scene = jax.tree.map(lambda x: x[:n_splats], splats)
            self.viewer.add_splats("scene", *scene.params)
        for i, (start, stop) in enumerate(splats.slices):
            drone = jax.tree.map(lambda x, a=start, b=stop: x[a:b], splats)
            self.viewer.add_splats(f"drone:{i}", *drone.params)
        self._n_drones = len(splats.slices)

    def update(self, sim: Sim, world: int = 0):
        """Push the current drone poses of one world to the viewer.

        Args:
            sim: The simulation to visualize.
            world: Index of the world whose drone poses are shown.
        """
        if self._n_drones == 0:
            return
        pos, quat = self._drone_mocap_poses(sim, world)
        for i in range(self._n_drones):
            self.viewer.update_pose(f"drone:{i}", pos[i], quat[i])

    def close(self):
        """Shut down the web server."""
        self.viewer.close()

    @staticmethod
    @requires_mujoco_sync
    def _drone_mocap_poses(sim: Sim, world: int) -> tuple[NDArray, NDArray]:
        """Drone mocap positions and wxyz quaternions of one world, synced with the MuJoCo data."""
        ids = sim.data.core.drone_mocap_ids
        pos = np.asarray(sim.mjx_data.mocap_pos[world, ids])
        quat = np.asarray(sim.mjx_data.mocap_quat[world, ids])  # MuJoCo quats are already wxyz
        return pos, quat
