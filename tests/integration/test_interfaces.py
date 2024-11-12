import jax.numpy as jnp
import pytest

from crazyflow.control.controller import Control, Controller
from crazyflow.exception import ConfigError
from crazyflow.sim.core import Physics, Sim


@pytest.mark.integration
@pytest.mark.parametrize("physics", Physics)
@pytest.mark.parametrize("controller", Controller)
def test_state_interface(physics: Physics, controller: Controller):
    if physics == Physics.sys_id:
        with pytest.raises(ConfigError):
            Sim(physics=physics, control=Control.state, controller=controller)
        return
    # Create environment with 1 world and 1 drone
    sim = Sim(physics=physics, control=Control.state, controller=controller)

    # Run simulation for 2 seconds
    for _ in range(int(2 * sim.freq)):
        # Simple P controller for attitude to reach target height
        cmd = jnp.zeros((1, 1, 13), dtype=jnp.float32)
        cmd = cmd.at[0, 0, 2].set(1.0)  # Set z position target to 1.0
        sim.state_control(cmd)
        sim.step()

    # Check if drone reached target position
    distance = jnp.linalg.norm(sim.states["pos"][0, 0] - jnp.array([0.0, 0.0, 1.0]))
    assert distance < 0.1, f"Failed to reach target height with {physics} physics"


@pytest.mark.integration
@pytest.mark.parametrize("physics", Physics)
@pytest.mark.parametrize("controller", Controller)
def test_attitude_interface(physics: Physics, controller: Controller):
    # Create environment with 1 world and 1 drone
    sim = Sim(physics=physics, control=Control.attitude, controller=controller)
    target_pos = jnp.array([0.0, 0.0, 1.0])

    # Run simulation for 2 seconds
    for _ in range(int(2 * sim.freq)):
        # Simple hover command - thrust only
        cmd = jnp.zeros((1, 1, 4), dtype=jnp.float32)
        cmd = cmd.at[0, 0, 0].set(1.0)  # Set thrust to 1.0
        sim.attitude_control(cmd)
        sim.step()

    # Check if drone maintained hover position
    dpos = sim.states["pos"][0, 0] - target_pos
    distance = jnp.linalg.norm(dpos)
    assert distance < 0.1, f"Failed to maintain hover with {physics} and {controller} ({dpos})"
