import jax
import pytest

from crazyflow.sim.core import Sim
from crazyflow.sim.physics import Physics


@pytest.mark.parametrize("physics", Physics)
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_sim_creation(physics: Physics, device: str):
    sim = Sim(n_worlds=2, n_drones=3, physics=physics, device=device)
    assert sim._n_worlds == 2
    assert sim._n_drones == 3
    assert sim.device == jax.devices(device)[0]
    assert sim.physics == physics

    # Test state buffer shapes
    assert sim._states["pos"].shape == (2, 3, 3)
    assert sim._states["pos"].device == jax.devices(device)[0]
    assert sim._states["quat"].shape == (2, 3, 4)
    assert sim._states["vel"].shape == (2, 3, 3)
    assert sim._states["ang_vel"].shape == (2, 3, 3)

    # Test control buffer shapes
    assert sim._controls["state"].shape == (2, 3, 13)
    assert sim._controls["attitude"].shape == (2, 3, 4)
    assert sim._controls["thrust"].shape == (2, 3, 4)


@pytest.mark.parametrize("physics", Physics)
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_sim_step(physics: Physics, device: str):
    sim = Sim(n_worlds=2, n_drones=3, physics=physics, device=device)
    try:
        sim.step()
    except NotImplementedError:
        pytest.skip("Physics not implemented")
