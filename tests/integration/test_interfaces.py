import jax
import numpy as np
import pytest
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude
from drone_models.core import load_params
from drone_models.transform import motor_force2rotor_vel
from scipy.spatial.transform import Rotation as R

from crazyflow.control.control import Control
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
    jit_state2attitude = jax.jit(parametrize(state2attitude, drone_model="cf2x_L250"))

    i_error = np.zeros((1, 1, 3))
    cmd = np.zeros((1, 1, 13))
    cmd[0, 0, 2] = 1.0  # Set z position target to 1.0

    for _ in range(int(2 * sim.control_freq)):  # Run simulation for 2 seconds
        pos, vel, quat = sim.data.states.pos, sim.data.states.vel, sim.data.states.quat
        rpyt, i_error = jit_state2attitude(pos, quat, vel, None, cmd, (i_error,), ctrl_freq=100)
        sim.attitude_control(rpyt)
        sim.step(sim.freq // sim.control_freq)

    # Check if drone maintained hover position
    dpos = sim.data.states.pos[0, 0] - target_pos
    distance = np.linalg.norm(dpos)
    assert distance < 0.05, f"Failed to maintain hover with {physics} ({dpos})"


@pytest.mark.integration
def test_rotor_vel_interface():
    sim = Sim(physics=Physics.first_principles, control=Control.rotor_vel)
    params = load_params("first_principles", "cf2x_L250")
    max_rpm = motor_force2rotor_vel(np.array([params["thrust_max"]]), params["rpm2thrust"])[0]

    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=sim.data.states.pos.at[..., 2].set(0.5))
    )
    cmd = np.full((1, 1, 4), max_rpm, dtype=np.float32)  # More RPMs than required for hover
    sim.rotor_vel_control(cmd)
    sim.step(sim.freq * 2)  # Run simulation for 2 seconds

    # Check if drone is not tilted
    assert R.from_quat(sim.data.states.quat[0, 0]).magnitude() < 0.1, "Drone is tilted"
    assert sim.data.states.pos[0, 0, 2] > 0.5, "Failed to accelerate with rotor velocity control"


@pytest.mark.integration
@pytest.mark.parametrize("physics", Physics)
def test_swarm_control(physics: Physics):
    n_worlds, n_drones = 2, 3
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=physics, control=Control.state)
    target_pos = sim.data.states.pos + np.array([0.3, 0.3, 0.3])

    cmd = np.zeros((n_worlds, n_drones, 13))
    cmd[..., :3] = target_pos
    sim.state_control(cmd)
    sim.step(3 * sim.freq)
    # Check if drone maintained hover position
    max_dist = np.max(np.linalg.norm(sim.data.states.pos - target_pos, axis=-1))
    assert max_dist < 0.05, f"Failed to reach target, max dist: {max_dist}"


@pytest.mark.integration
@pytest.mark.parametrize("physics", Physics)
def test_yaw_rotation(physics: Physics):
    # TODO: Enable yaw rotations once the models are better calibrated
    if physics != Physics.first_principles:
        pytest.skip(f"Physics mode {physics} currently does not support yaw rotation")

    sim = Sim(physics=physics, control=Control.state, state_freq=100)
    sim.reset()

    cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
    cmd[..., :3] = 0.2
    cmd[..., 9] = np.pi / 2  # Test if the drone can rotate in yaw

    sim.state_control(cmd)
    sim.step(200 * sim.freq // sim.control_freq)  # Run simulation for 2 seconds
    pos = sim.data.states.pos[0, 0]
    rot = R.from_quat(sim.data.states.quat[0, 0])
    distance = np.linalg.norm(pos - np.array([0.2, 0.2, 0.2]))
    assert distance < 0.1, f"Failed to reach target, distance: {distance}"
    angle = rot.as_euler("xyz")[2]
    assert np.abs(angle - np.pi / 2) < 0.1, f"Failed to rotate in yaw, angle: {angle}"
