import jax.numpy as jnp
import pytest

from crazyflow.control.controller import Control, state2attitude
from crazyflow.sim.core import Physics, Sim


@pytest.mark.integration
@pytest.mark.parametrize("physics", Physics)
def test_state_interface(physics: Physics):
    # Create environment with 1 world and 1 drone
    if physics == Physics.sys_id:
        pytest.skip("sys_id physics does not support state control")
    sim = Sim(physics=physics, control=Control.state)

    # Run simulation for 2 seconds
    for _ in range(int(2 * sim.freq)):
        # Simple P controller for attitude to reach target height
        cmd = jnp.zeros((1, 1, 13), dtype=jnp.float32)
        cmd = cmd.at[0, 0, 2].set(1.0)  # Set z position target to 1.0
        sim.state_control(cmd)
        sim.step()
        if jnp.linalg.norm(sim.states.pos[0, 0] - jnp.array([0.0, 0.0, 1.0])) < 0.1:
            break

    # Check if drone reached target position
    distance = jnp.linalg.norm(sim.states.pos[0, 0] - jnp.array([0.0, 0.0, 1.0]))
    assert distance < 0.1, f"Failed to reach target height with {physics} physics"


@pytest.mark.integration
@pytest.mark.parametrize("physics", Physics)
def test_attitude_interface(physics: Physics):
    # Create environment with 1 world and 1 drone
    sim = Sim(physics=physics, control=Control.attitude)
    target_pos = jnp.array([0.0, 0.0, 1.0])

    i_error = jnp.zeros((1, 1, 3))

    for _ in range(int(2 * sim.freq)):  # Run simulation for 2 seconds
        pos, vel, quat = sim.states.pos, sim.states.vel, sim.states.quat
        des_pos = jnp.array([[[0, 0, 1.0]]])
        cmd, i_error = state2attitude(
            pos, vel, quat, des_pos, jnp.zeros((1, 1, 3)), jnp.zeros((1, 1, 1)), i_error, sim.dt
        )
        sim.attitude_control(cmd)
        sim.step()
        if jnp.linalg.norm(sim.states.pos[0, 0] - target_pos) < 0.1:
            break

    # Check if drone maintained hover position
    dpos = sim.states.pos[0, 0] - target_pos
    distance = jnp.linalg.norm(dpos)
    assert distance < 0.1, f"Failed to maintain hover with {physics} ({dpos})"


@pytest.mark.integration
@pytest.mark.parametrize("physics", Physics)
def test_swarm_control(physics: Physics):
    n_worlds, n_drones = 2, 3
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=physics, control=Control.state)
    target_pos = sim.states.pos + jnp.array([0.2, 0.2, 0.2])

    for _ in range(int(5 * sim.freq)):  # Run simulation for 2 seconds
        cmd = jnp.zeros((n_worlds, n_drones, 13))
        cmd = cmd.at[..., :3].set(target_pos)
        sim.state_control(cmd)
        sim.step()
        if jnp.linalg.norm(sim.states.pos[0, 0] - target_pos) < 0.1:
            break

    # Check if drone maintained hover position
    max_dist = jnp.max(jnp.linalg.norm(sim.states.pos - target_pos, axis=-1))
    assert max_dist < 0.05, f"Failed to reach target, max dist: {max_dist}"
