"""Unit tests for the sensors module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.sim import Sim
from crazyflow.sim.sensors import camera_rays, render_depth


@pytest.mark.unit
def test_camera_rays():
    """Test that camera_rays produces arrays with correct shape and device."""
    resolution = (64, 48)
    rays = camera_rays(resolution=resolution)
    # Check shape: should be (height, width, 3)
    expected_shape = (resolution[1], resolution[0], 3)
    assert rays.shape == expected_shape, f"Expected shape {expected_shape}, got {rays.shape}"
    # Check that rays are normalized
    norm = jnp.linalg.norm(rays, axis=-1)
    assert jnp.allclose(norm, 1.0, atol=1e-6), "Rays should be normalized"
    # Check that rays respect the FOV
    rays_narrow = camera_rays(fov_y=np.pi / 6)
    rays_wide = camera_rays(fov_y=np.pi / 3)
    # Corner rays should have different angles for different FOV
    # Check the top corner ray y-component (wider FOV should have larger y-component)
    corner_y_narrow = abs(rays_narrow[0, 0, 1])  # Top-left corner
    corner_y_wide = abs(rays_wide[0, 0, 1])
    assert corner_y_wide > corner_y_narrow, "Wider FOV should produce rays with larger y-components"


@pytest.mark.unit
def test_render_depth(device: str):
    """Test render_depth with different resolutions."""
    sim = Sim(n_worlds=2)
    dist = render_depth(sim, camera=0, resolution=(10, 10))
    assert dist.shape == (2, 10, 10), f"Expected shape (2, 10, 10), got {dist.shape}"
    assert dist.device == jax.devices(device)[0], f"Expected device {device}, got {dist.device}"
