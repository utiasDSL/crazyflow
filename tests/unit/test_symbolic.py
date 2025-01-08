import pytest

from crazyflow.constants import MASS, J
from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.symbolic import symbolic_attitude, symbolic_from_sim, symbolic_thrust


@pytest.mark.unit
def test_symbolic_attitude_model_creation():
    """Test creating symbolic attitude model directly."""
    dt = 0.01
    model = symbolic_attitude(dt)

    assert model.nx == 12  # State dimension
    assert model.nu == 4  # Input dimension
    assert model.ny == 12  # Output dimension
    assert model.dt == dt


@pytest.mark.unit
def test_symbolic_thrust_model_creation():
    """Test creating symbolic thrust model directly."""
    dt = 0.01
    model = symbolic_thrust(mass=MASS, J=J, dt=dt)

    assert model.nx == 12  # State dimension
    assert model.nu == 4  # Input dimension
    assert model.ny == 12  # Output dimension
    assert model.dt == dt


@pytest.mark.unit
@pytest.mark.parametrize("n_worlds", [1, 2])
@pytest.mark.parametrize("control", [Control.attitude, Control.thrust])
def test_symbolic_from_sim(n_worlds: int, control: Control):
    """Test creating symbolic model from sim instance."""
    sim = Sim(n_worlds=n_worlds, n_drones=1, control=control)
    model = symbolic_from_sim(sim)

    assert model.nx == 12
    assert model.nu == 4
    assert model.ny == 12
    assert model.dt == 1 / sim.freq
    assert model.x_sym.shape == (12, 1)
    assert model.u_sym.shape == (4, 1)
