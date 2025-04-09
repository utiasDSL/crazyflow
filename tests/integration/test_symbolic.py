import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from crazyflow.control.control import MAX_THRUST, MIN_THRUST
from crazyflow.sim import Sim
from crazyflow.sim.physics import ang_vel2rpy_rates
from crazyflow.sim.structs import SimState
from crazyflow.sim.symbolic import symbolic_from_sim


def sim_state2symbolic_state(state: SimState) -> NDArray[np.float32]:
    """Convert the simulation state to the symbolic state vector."""
    pos = state.pos.squeeze()  # shape: (3,)
    vel = state.vel.squeeze()  # shape: (3,)
    euler = R.from_quat(state.quat.squeeze()).as_euler("xyz")  # shape: (3,), Euler angles
    rpy_rates = ang_vel2rpy_rates(state.ang_vel.squeeze(), state.quat.squeeze())  # shape: (3,)
    return np.concatenate([pos, euler, vel, rpy_rates])


@pytest.mark.integration
@pytest.mark.parametrize("freq", [500, 1000])
def test_attitude_symbolic(freq: int):
    sim = Sim(physics="sys_id", freq=freq)
    sym = symbolic_from_sim(sim)

    x0 = np.zeros(12)

    # Simulate with both models for 0.5 seconds
    t_end = 0.5
    dt = 1 / sim.freq
    steps = int(t_end / dt)

    # Track states over time
    x_sym_log = []
    x_sim_log = []

    # Initialize logs with initial state
    x_sym = x0.copy()
    x_sim = x0.copy()
    x_sym_log.append(x_sym)
    x_sim_log.append(x_sim)

    u_low = np.array([4 * MIN_THRUST, -np.pi, -np.pi, -np.pi])
    u_high = np.array([4 * MAX_THRUST, np.pi, np.pi, np.pi])
    rng = np.random.default_rng(seed=42)

    # Run simulation
    for _ in range(steps):
        u_rand = (rng.random(4) * (u_high - u_low) + u_low).astype(np.float32)
        # Simulate with symbolic model
        res = sym.fd_func(x0=x_sym, p=u_rand)
        x_sym = res["xf"].full().flatten()
        x_sym_log.append(x_sym)
        # Simulate with attitude controller
        sim.attitude_control(u_rand.reshape(1, 1, 4))
        sim.step(sim.freq // sim.control_freq)
        x_sim_log.append(sim_state2symbolic_state(sim.data.states))

    # Convert logs to arrays. Do not record the rpy rates (deviate easily).
    x_sym_log = np.array(x_sym_log)[..., :-3]
    x_sim_log = np.array(x_sim_log)[..., :-3]

    # Check if states match throughout simulation
    err_msg = "Symbolic and simulation prediction do not match approximately"
    assert np.allclose(x_sym_log, x_sim_log, rtol=1e-2, atol=1e-2), err_msg
    sim.close()


@pytest.mark.integration
@pytest.mark.parametrize("freq", [500, 1000])
def test_thrust_symbolic(freq: int):
    sim = Sim(physics="analytical", control="thrust", freq=freq)
    sym = symbolic_from_sim(sim)

    x0 = np.zeros(12)

    # Simulate with both models for 0.5 seconds
    t_end = 0.5
    dt = 1 / sim.freq
    steps = int(t_end / dt)

    # Track states over time
    x_sym_log = []
    x_sim_log = []

    # Initialize logs with initial state
    x_sym = x0.copy()
    x_sim = x0.copy()
    x_sym_log.append(x_sym)
    x_sim_log.append(x_sim)

    rng = np.random.default_rng(seed=42)

    # Run simulation
    for _ in range(steps):
        u_rand = rng.uniform(MIN_THRUST, MIN_THRUST, 4)
        # Simulate with symbolic model
        res = sym.fd_func(x0=x_sym, p=u_rand)
        x_sym = res["xf"].full().flatten()
        x_sym_log.append(x_sym)
        # Simulate with attitude controller
        sim.thrust_control(u_rand.reshape(1, 1, 4))
        sim.step(sim.freq // sim.control_freq)
        x_sim_log.append(sim_state2symbolic_state(sim.data.states))

    x_sym_log = np.array(x_sym_log)
    x_sim_log = np.array(x_sim_log)

    # Check if states match throughout simulation
    err_msg = "Symbolic and simulation prediction do not match approximately"
    assert np.allclose(x_sym_log, x_sim_log, rtol=1e-2, atol=1e-3), err_msg
    sim.close()
