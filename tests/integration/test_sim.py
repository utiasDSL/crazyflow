import jax.numpy as jnp
import pytest

from crazyflow.control.controller import Control, Controller
from crazyflow.sim.core import Physics, Sim


@pytest.mark.integration
def test_hover():
    # Create environment with 1 world and 1 drone
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        physics=Physics.sys_id,
        control=Control.state,
        controller=Controller.emulatefirmware,
    )

    # Run simulation for 5 seconds
    for _ in range(int(5 * sim.freq)):
        # Simple P controller for attitude to reach target height
        cmd = jnp.zeros((1, 1, 13), dtype=jnp.float32)
        cmd.at[0, 0, 2].set(1.0)
        sim.state_control(cmd)
        sim.step()

    drone_pos = sim._states["pos"][0, 0]
    assert jnp.abs(drone_pos[2] - 1.0) < 0.05
