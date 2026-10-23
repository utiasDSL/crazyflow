"""Unit tests for distributing the simulation across devices."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import available_backends
from jax.sharding import NamedSharding, PartitionSpec

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.functional import attitude_control
from crazyflow.sim.sharding import placement, shard, world_mesh

# Sharding tests require at least 2 devices
multi_device = [
    pytest.param(
        platform,
        marks=pytest.mark.skipif(
            len(jax.devices(platform)) < 2, reason=f"requires at least 2 {platform} devices"
        ),
    )
    for platform in available_backends()
]


def assert_sharding(x: Any, sharding: Any):
    assert x.sharding.is_equivalent_to(sharding, x.ndim), f"{x.sharding} != {sharding}"


@pytest.mark.unit
@pytest.mark.parametrize("platform", multi_device)
@pytest.mark.parametrize("at_construction", [True, False])
@pytest.mark.parametrize("n_drones", [1, 2])
def test_placement(platform: str, at_construction: bool, n_drones: int):
    devices = jax.devices(platform)
    mesh = world_mesh(devices)
    n_worlds = 2 * len(devices)
    if at_construction:
        sim = Sim(n_worlds=n_worlds, n_drones=n_drones, device=platform, mesh=mesh)
    else:
        sim = Sim(n_worlds=n_worlds, n_drones=n_drones, device=platform)
        sim.shard(mesh)
    assert sim.data.states.pos.sharding.spec == PartitionSpec("worlds")
    assert sim.data.params.mass.sharding.spec == PartitionSpec()
    assert sim.data.params.gravity_vec.sharding.spec == PartitionSpec()
    assert sim.data.params.rotor_dyn_coef.sharding.spec == PartitionSpec()
    expected = placement(sim.data, mesh)
    jax.tree.map(assert_sharding, sim.data, expected)  # Sanity-check the rest
    jax.tree.map(assert_sharding, sim.default_data, expected)
    # MJX data is also per-world, so should be sharded as well
    world = NamedSharding(mesh, PartitionSpec("worlds"))
    jax.tree.map(lambda leaf: assert_sharding(leaf, world), sim.mjx_data)


@pytest.mark.unit
@pytest.mark.parametrize("platform", multi_device)
def test_sharded_oop_api(platform: str):
    devices = jax.devices(platform)
    sim = Sim(n_worlds=2 * len(devices), device=platform, control=Control.attitude)
    sim.shard(world_mesh(devices))
    expected = placement(sim.data, world_mesh(jax.devices(platform)))
    jax.tree.map(assert_sharding, sim.default_data, expected)
    sim.attitude_control(jnp.zeros((sim.n_worlds, 1, 4)))
    sim.step()
    jax.tree.map(assert_sharding, sim.data, expected)
    sim.reset()
    jax.tree.map(assert_sharding, sim.data, expected)


@pytest.mark.unit
@pytest.mark.parametrize("platform", multi_device)
def test_sharded_functional_api(platform: str):
    devices = jax.devices(platform)
    sim = Sim(n_worlds=2 * len(devices), control=Control.attitude, device=platform)
    step_fn, reset_fn = sim.build_step_fn(), sim.build_reset_fn()
    mesh = world_mesh(devices)
    data, default_data = shard(sim.data, mesh), shard(sim.default_data, mesh)
    expected = placement(sim.data, mesh)
    data = attitude_control(data, jnp.zeros((sim.n_worlds, 1, 4)))
    data = step_fn(data)
    jax.tree.map(assert_sharding, data, expected)
    data = reset_fn(data, default_data)
    jax.tree.map(assert_sharding, data, expected)


@pytest.mark.unit
@pytest.mark.parametrize("platform", multi_device)
def test_sharded_step_values(platform: str):
    # Sharding must not change the simulation results
    devices = jax.devices(platform)
    sim = Sim(n_worlds=2 * len(devices), device=platform)
    sim.step(10)
    pos = np.asarray(sim.data.states.pos)  # Copy to np for comparison across shardings
    sim.reset()
    sim.shard(world_mesh(devices))
    sim.step(10)
    assert np.allclose(np.asarray(sim.data.states.pos), pos, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("platform", multi_device)
def test_sharded_contacts(platform: str):
    # Put drones in collision every other world, check that contacts can be computed and are correct
    devices = jax.devices(platform)
    n_worlds = 2 * len(devices)
    grounded = jnp.arange(n_worlds) % 2 == 0
    sim = Sim(n_worlds=n_worlds, device=platform, mesh=world_mesh(devices))
    pos = sim.data.states.pos.at[:, 0, 2].set(jnp.where(grounded, -0.05, 1.0))
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=pos), core=sim.data.core.replace(mjx_synced=False)
    )
    assert jnp.array_equal(sim.contacts()[:, 0], grounded), "wrong worlds collide"
    sim.close()
