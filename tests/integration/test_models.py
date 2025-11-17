import pytest

from drone_models.drones import available_drones

from crazyflow.sim import Sim
from crazyflow.sim.physics import Physics


@pytest.mark.integration
@pytest.mark.parametrize("physics", Physics)
@pytest.mark.parametrize("model", available_drones)
def test_attitude_symbolic(physics: Physics, model: "str"):
    """Tests if xml files contain syntax errors."""
    Sim(physics=physics, drone_model=model)
