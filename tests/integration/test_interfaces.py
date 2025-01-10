import jax
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from crazyflow.control.control import Control, state2attitude
from crazyflow.sim import Physics, Sim


@pytest.mark.integration
@pytest.mark.parametrize("physics", Physics)
def test_state_interface(physics: Physics):
    sim = Sim(physics=physics, control=Control.state)

    # Simple P controller for attitude to reach target height
    cmd = np.zeros((1, 1, 13), dtype=np.float32)
    cmd[0, 0, 2] = 1.0  # Set z position target to 1.0

    for _ in range(int(2 * sim.control_freq)):  # Run simulation for 2 seconds
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if np.linalg.norm(sim.data.states.pos[0, 0] - np.array([0.0, 0.0, 1.0])) < 0.1:
            break

    # Check if drone reached target position
    distance = np.linalg.norm(sim.data.states.pos[0, 0] - np.array([0.0, 0.0, 1.0]))
    assert distance < 0.1, f"Failed to reach target height with {physics} physics"


@pytest.mark.integration
@pytest.mark.parametrize("physics", Physics)
def test_attitude_interface(physics: Physics):
    sim = Sim(physics=physics, control=Control.attitude)
    target_pos = np.array([0.0, 0.0, 1.0])
    jit_state2attitude = jax.jit(state2attitude)

    i_error = np.zeros((1, 1, 3))

    for _ in range(int(2 * sim.control_freq)):  # Run simulation for 2 seconds
        pos, vel, quat = sim.data.states.pos, sim.data.states.vel, sim.data.states.quat
        des_pos = np.array([[[0, 0, 1.0]]])
        dt = 1 / sim.data.controls.attitude_freq
        cmd, i_error = jit_state2attitude(
            pos, vel, quat, des_pos, np.zeros((1, 1, 3)), np.zeros((1, 1, 1)), i_error, dt
        )
        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if np.linalg.norm(sim.data.states.pos[0, 0] - target_pos) < 0.1:
            break

    # Check if drone maintained hover position
    dpos = sim.data.states.pos[0, 0] - target_pos
    distance = np.linalg.norm(dpos)
    assert distance < 0.1, f"Failed to maintain hover with {physics} ({dpos})"


@pytest.mark.integration
@pytest.mark.parametrize("physics", Physics)
def test_swarm_control(physics: Physics):
    n_worlds, n_drones = 2, 3
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=physics, control=Control.state)
    target_pos = sim.data.states.pos + np.array([0.2, 0.2, 0.2])

    cmd = np.zeros((n_worlds, n_drones, 13))
    cmd[..., :3] = target_pos

    for _ in range(int(3 * sim.control_freq)):  # Run simulation for 2 seconds
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if np.linalg.norm(sim.data.states.pos[0, 0] - target_pos) < 0.1:
            break

    # Check if drone maintained hover position
    max_dist = np.max(np.linalg.norm(sim.data.states.pos - target_pos, axis=-1))
    assert max_dist < 0.05, f"Failed to reach target, max dist: {max_dist}"


@pytest.mark.integration
@pytest.mark.parametrize("physics", Physics)
def test_yaw_rotation(physics: Physics):
    if physics == Physics.sys_id:  # TODO: Remove once yaw is supported for sys_id
        pytest.skip("Yaw != 0 currently not supported for sys_id")

    sim = Sim(physics=physics, control=Control.state)
    sim.reset()

    cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
    cmd[..., :3] = 0.2
    cmd[..., 9] = np.pi / 2  # Test if the drone can rotate in yaw

    sim.state_control(cmd)
    sim.step(sim.freq * 2)
    pos = sim.data.states.pos[0, 0]
    rot = R.from_quat(sim.data.states.quat[0, 0])
    distance = np.linalg.norm(pos - np.array([0.2, 0.2, 0.2]))
    assert distance < 0.1, f"Failed to reach target, distance: {distance}"
    angle = rot.as_euler("xyz")[2]
    assert np.abs(angle - np.pi / 2) < 0.1, f"Failed to rotate in yaw, angle: {angle}"
