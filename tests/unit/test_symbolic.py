import casadi as cs
import pytest

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.symbolic import symbolic_from_sim


@pytest.mark.unit
@pytest.mark.parametrize("n_worlds", [1, 2])
def test_symbolic_from_sim(n_worlds: int):
    """Test creating symbolic model from sim instance."""
    sim = Sim(n_worlds=n_worlds, n_drones=1, control=Control.attitude)
    X_dot, X, U, Y = symbolic_from_sim(sim)

    assert isinstance(X_dot, cs.MX)
    assert isinstance(X, cs.MX)
    assert isinstance(U, cs.MX)
    assert isinstance(Y, cs.MX)
    assert X_dot.shape == (13, 1)
    assert X.shape == (13, 1)
    assert U.shape == (4, 1)
    assert Y.shape == (7, 1)


@pytest.mark.unit
@pytest.mark.parametrize("control", [Control.state, Control.force_torque])
def test_symbolic_from_sim_errors(control: Control):
    """Test creating symbolic model from sim instance."""
    sim = Sim(control=control)
    with pytest.raises(ValueError, match="Symbolic model dynamics only support"):
        symbolic_from_sim(sim)
