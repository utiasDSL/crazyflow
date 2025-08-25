import casadi as cs
import numpy as np
import pytest
from numpy.typing import NDArray

from crazyflow.sim import Sim
from crazyflow.sim.physics import Physics
from crazyflow.sim.structs import SimState
from crazyflow.sim.symbolic import symbolic_from_sim


def sim_state2symbolic_state(state: SimState) -> NDArray[np.float32]:
    """Convert the simulation state to the symbolic state vector."""
    return np.concat([state.pos, state.quat, state.vel, state.ang_vel], axis=-1)[0, 0][..., None]


@pytest.mark.integration
@pytest.mark.parametrize("physics", Physics)
@pytest.mark.parametrize("freq", [500, 1000])
def test_attitude_symbolic(physics: Physics, freq: int):
    if physics in (Physics.so_rpy_rotor, Physics.so_rpy_rotor_drag):
        pytest.skip(f"Physics mode {physics} not yet implemented")

    sim = Sim(physics=physics, freq=freq)
    sim.step_pipeline = sim.step_pipeline[:-1]  # Remove clip floor from step pipeline
    X_dot, X, U, Y = symbolic_from_sim(sim)
    fd = cs.integrator("fd", "cvodes", {"x": X, "p": U, "ode": X_dot}, 0, 1 / freq)

    x0 = sim_state2symbolic_state(sim.data.states)

    # Simulate with both models for 0.5 seconds
    t_end = 0.5
    dt = 1 / sim.freq
    steps = int(t_end / dt)

    # Track states over time
    x_sym_log = []
    x_sim_log = []

    # Initialize logs with initial state
    x_sym = x0.copy()
    x_sym_log.append(x_sym)
    x_sim = x0.copy()
    x_sim_log.append(x_sim)

    u_low = np.array([-np.pi, -np.pi, -np.pi, 0.3]).reshape(4, 1)
    u_high = np.array([np.pi, np.pi, np.pi, 0.5]).reshape(4, 1)
    rng = np.random.default_rng(seed=42)

    # Run simulation
    for _ in range(steps):
        u_rand = (rng.random(4)[..., None] * (u_high - u_low) + u_low).astype(np.float32)
        assert x_sym.shape == (13, 1)
        assert u_rand.shape == (4, 1)
        # Simulate with symbolic model
        x_sym = fd(x0=x_sym, p=u_rand)["xf"].full()
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
