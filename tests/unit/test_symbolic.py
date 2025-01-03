import pytest

from crazyflow.constants import MASS, J
from crazyflow.sim import Sim
from crazyflow.sim.symbolic import symbolic, symbolic_from_sim


@pytest.mark.unit
def test_symbolic_model_creation():
    """Test creating symbolic model directly."""
    dt = 0.01
    model = symbolic(mass=MASS, J=J, dt=dt)

    assert model.nx == 12  # State dimension
    assert model.nu == 4  # Input dimension
    assert model.ny == 12  # Output dimension
    assert model.dt == dt


@pytest.mark.unit
@pytest.mark.parametrize("n_worlds", [1, 2])
def test_symbolic_from_sim(n_worlds: int):
    """Test creating symbolic model from sim instance."""
    sim = Sim(n_worlds=n_worlds, n_drones=1)
    model = symbolic_from_sim(sim)

    assert model.nx == 12
    assert model.nu == 4
    assert model.ny == 12
    assert model.dt == 1 / sim.freq
    assert model.x_sym.shape == (12, 1)
    assert model.u_sym.shape == (4, 1)
