"""Distribution of crazyflow over multiple devices.

Sharding arrays along the world axis allows efficient parallelization of crazyflow across multiple
devices. The data itself specifies which arrays carry the world axis. See
[world_mask][crazyflow.utils.world_mask].
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from crazyflow.utils import world_mask

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jax import Array, Device
    from jax.sharding import Mesh
    from mujoco.mjx import Data  # ty: ignore[unresolved-import]

    from crazyflow.sim.data import SimData

WORLD_AXIS = "worlds"
"""Name of the mesh axis that the worlds are distributed over."""


def world_mesh(devices: Sequence[Device]) -> Mesh:
    """Create a mesh that distributes the worlds over the devices.

    The mesh uses automatic axis types. Explicit axis types cause problems in SciPy.

    Args:
        devices: Devices to distribute the worlds over.

    Returns:
        A one-dimensional mesh over the world axis.
    """
    axes, names = (len(devices),), (WORLD_AXIS,)
    return jax.make_mesh(axes, names, axis_types=(AxisType.Auto,), devices=devices)


def placement(data: SimData, mesh: Mesh) -> SimData:
    """Build the placement that distributes the worlds of the simulation data over a mesh.

    Note:
        Pass the placement to `jax.device_put`, or modify it first to place the data by hand.

    Args:
        data: Simulation data to place.
        mesh: Mesh to distribute the worlds over, as built by
            [world_mesh][crazyflow.sim.sharding.world_mesh].

    Returns:
        A pytree of shardings matching the simulation data.
    """
    world = NamedSharding(mesh, PartitionSpec(WORLD_AXIS))
    replicated = NamedSharding(mesh, PartitionSpec())
    return jax.tree.map(lambda indexed: world if indexed else replicated, world_mask(data))


def shard(data: SimData, mesh: Mesh) -> SimData:
    """Distribute the worlds of the simulation data over a mesh.

    Args:
        data: Simulation data to place.
        mesh: Mesh to distribute the worlds over.

    Returns:
        The placed simulation data.
    """
    return jax.device_put(data, placement(data, mesh))


def build_sharded_data(
    create: Callable[[int | Array], SimData], rng_key: int | Array, mesh: Mesh
) -> SimData:
    """Build simulation data distributed over a mesh.

    Tracing the construction lets us put the result directly on the mesh and prevents data from
    being materialised on a single device.

    Args:
        create: Callable that builds the data from an rng key.
        rng_key: Random number generator key for the simulation, or a seed to derive one from.
        mesh: Mesh to distribute the worlds over.

    Returns:
        The placed simulation data.
    """
    if isinstance(rng_key, int):  # Tracing turns a seed into an array that is not a key
        rng_key = jax.random.key(rng_key)
    return jax.jit(create, out_shardings=placement(jax.eval_shape(create, rng_key), mesh))(rng_key)


def build_sharded_mjx_data(data: Data, n_worlds: int, mesh: Mesh) -> Data:
    """Distribute the per-world MJX data over a mesh without materialising on a single device.

    Args:
        data: MJX data of a single world.
        n_worlds: Number of worlds to copy the data into.
        mesh: Mesh to distribute the worlds over.

    Returns:
        The placed MJX data.
    """
    data = jax.device_put(data, NamedSharding(mesh, PartitionSpec()))  # Trace on the mesh devices
    broadcast = partial(jax.tree.map, lambda x: jnp.broadcast_to(x, (n_worlds, *x.shape)))
    return jax.jit(broadcast, out_shardings=NamedSharding(mesh, PartitionSpec(WORLD_AXIS)))(data)
