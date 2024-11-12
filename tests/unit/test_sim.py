import jax
import pytest

from crazyflow.control.controller import Control
from crazyflow.exception import ConfigError
from crazyflow.sim.core import Sim
from crazyflow.sim.physics import Physics


@pytest.mark.unit
@pytest.mark.parametrize("physics", Physics)
@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("control", Control)
def test_sim_creation(physics: Physics, device: str, control: Control):
    if physics == Physics.sys_id and control != Control.attitude:
        with pytest.raises(ConfigError):
            Sim(n_worlds=2, n_drones=3, physics=physics, device=device, control=control)
        return
    sim = Sim(n_worlds=2, n_drones=3, physics=physics, device=device, control=control)
    assert sim.n_worlds == 2
    assert sim.n_drones == 3
    assert sim.device == jax.devices(device)[0]
    assert sim.physics == physics

    # Test state buffer shapes
    assert sim.states["pos"].shape == (2, 3, 3)
    assert sim.states["pos"].device == jax.devices(device)[0]
    assert sim.states["quat"].shape == (2, 3, 4)
    assert sim.states["vel"].shape == (2, 3, 3)
    assert sim.states["ang_vel"].shape == (2, 3, 3)

    # Test control buffer shapes
    assert sim._controls["attitude"].shape == (2, 3, 4)
    assert sim._controls["thrust"].shape == (2, 3, 4)
    assert sim._controls["state"].shape == (2, 3, 13)
    assert sim._controls["state"].device == jax.devices(device)[0]


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_setup(device: str):
    sim = Sim(n_worlds=2, n_drones=3, device=device)
    sim.setup()


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("physics", Physics)
def test_reset(device: str, physics: Physics):
    sim = Sim(n_worlds=2, n_drones=3, physics=physics, device=device)
    sim.reset()


@pytest.mark.unit
@pytest.mark.parametrize("physics", Physics)
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_sim_step(physics: Physics, device: str):
    sim = Sim(n_worlds=2, n_drones=3, physics=physics, device=device)
    try:
        for _ in range(3):
            sim.step()
    except NotImplementedError:
        pytest.skip("Physics not implemented")


@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.render
def test_render(device: str):
    sim = Sim(device=device)
    sim.render()
    sim.viewer.close()
