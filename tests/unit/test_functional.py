from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import crazyflow.sim.functional as F
from crazyflow.control import Control
from crazyflow.sim import Sim


@pytest.mark.unit
def test_functional_resets():
    """Test that the functional API works as expected for resets."""
    sim = Sim()
    data, default_data = sim.build_data(), sim.build_default_data()
    reset_fn = sim.build_reset_fn()
    data = data.replace(states=data.states.replace(pos=jnp.ones_like(data.states.pos)))
    sim.data = data
    # Test types
    assert callable(reset_fn), "reset_fn must be a pure function"
    assert not hasattr(reset_fn, "__self__"), "reset_fn must not be a bound method"
    # Test the reset runs as expected
    data_reset = reset_fn(data, default_data, None)
    assert jnp.all(data_reset.states.pos == 0), "reset_fn did not reset positions to zero"
    data_reset = reset_fn(data, default_data, jnp.array([True] * sim.n_worlds))
    assert jnp.all(data_reset.states.pos == 0), "reset_fn did not reset positions to zero with mask"


@pytest.mark.unit
def test_functional_steps():
    """Test that the functional API works as expected for steps."""
    sim = Sim()
    data = sim.build_data()
    data = data.replace(states=data.states.replace(pos=jnp.ones_like(data.states.pos)))
    step_fn = sim.build_step_fn()
    # Test types
    assert callable(step_fn), "step_fn must be a pure function"
    assert not hasattr(step_fn, "__self__"), "step_fn must not be a bound method"
    # Test the step function runs as expected
    data_step = step_fn(data, 5)
    assert jnp.all(data_step.states.pos[..., 2] < 1), "step_fn did not step correctly"


@pytest.mark.unit
@pytest.mark.parametrize("attitude_freq", [33, 50, 100, 200])
def test_functional_attitude_control(attitude_freq: int):
    """Test that functional attitude control respects frequency and applies commands correctly.

    Ported from test_attitude_control in test_sim.py.
    """
    sim = Sim(n_worlds=2, n_drones=3, control="attitude", freq=100, attitude_freq=attitude_freq)

    data = sim.build_data()
    default_data = sim.build_default_data()
    reset_fn = sim.build_reset_fn()
    step_fn = sim.build_step_fn()

    can_control_1 = np.arange(6) * attitude_freq % sim.freq < attitude_freq
    can_control_2 = np.array([0, 0, 1, 2, 3, 4]) * attitude_freq % sim.freq < attitude_freq
    for i in range(6):
        cmd = np.random.rand(sim.n_worlds, sim.n_drones, 4)
        # Check controllable status
        controllable = F.controllable(data)
        assert jnp.all(controllable[0] == can_control_1[i]), f"Controllable 1 mismatch at t={i}"
        assert jnp.all(controllable[1] == can_control_2[i]), f"Controllable 2 mismatch at t={i}"
        # Apply control
        data = F.attitude_control(data, cmd)
        data = step_fn(data, 1)
        sim_cmd = data.controls.attitude.cmd[0]
        if can_control_1[i]:
            assert jnp.all(sim_cmd == cmd[0]), f"Controls do not match at t={i}"
        else:
            assert not jnp.all(sim_cmd == cmd[0]), f"Controls shouldn't match at t={i}"
        sim_cmd = data.controls.attitude.cmd[1]
        if can_control_2[i]:
            assert jnp.all(sim_cmd == cmd[1]), f"Controls do not match at t={i}"
        else:
            assert not jnp.all(sim_cmd == cmd[1]), f"Controls shouldn't match at t={i}"
        if i == 0:  # Make world 2 asynchronous
            data = reset_fn(data, default_data, np.array([False, True]))


@pytest.mark.unit
def test_functional_attitude_control_device(device: str):
    """Test that functional attitude control maintains JAX arrays on correct device."""
    sim = Sim(n_worlds=2, n_drones=3, control=Control.attitude, device=device)
    data = sim.build_data()
    cmd = np.random.rand(sim.n_worlds, sim.n_drones, 4)
    data = F.attitude_control(data, cmd)
    controls = data.controls.attitude
    assert isinstance(controls.staged_cmd, jnp.ndarray), "Buffers must remain JAX arrays"
    assert jnp.all(controls.staged_cmd == cmd), "Buffers must match command"


@pytest.mark.unit
@pytest.mark.parametrize("state_freq", [33, 50, 100, 200])
def test_functional_state_control(state_freq: int):
    """Test that functional state control respects frequency and applies commands correctly."""
    sim = Sim(n_worlds=2, n_drones=3, control=Control.state, freq=100, state_freq=state_freq)

    data = sim.build_data()
    default_data = sim.build_default_data()
    reset_fn = sim.build_reset_fn()
    step_fn = sim.build_step_fn()

    can_control_1 = np.arange(6) * state_freq % sim.freq < state_freq
    can_control_2 = np.array([0, 0, 1, 2, 3, 4]) * state_freq % sim.freq < state_freq

    for i in range(6):
        cmd = np.random.rand(sim.n_worlds, sim.n_drones, 13)
        # Check controllable status
        controllable = F.controllable(data)
        assert jnp.all(controllable[0] == can_control_1[i]), f"Controllable 1 mismatch at t={i}"
        assert jnp.all(controllable[1] == can_control_2[i]), f"Controllable 2 mismatch at t={i}"
        # Apply control
        data = F.state_control(data, cmd)
        last_attitude = data.controls.attitude.staged_cmd
        data = step_fn(data, 1)
        attitude = data.controls.attitude.staged_cmd

        last_att, att = last_attitude[0], attitude[0]
        if can_control_1[i]:
            assert not jnp.all(att == last_att), f"Controls haven't been applied at t={i}"
        else:
            assert jnp.all(att == last_att), f"Controls should be unchanged at t={i}"

        last_att, att = last_attitude[1], attitude[1]
        if can_control_2[i]:
            assert not jnp.all(att == last_att), f"Controls haven't been applied at t={i}"
        else:
            assert jnp.all(att == last_att), f"Controls should be unchanged at t={i}"
        if i == 0:  # Make world 2 asynchronous
            data = reset_fn(data, default_data, np.array([False, True]))


@pytest.mark.unit
def test_functional_state_control_device(device: str):
    """Test that functional state control maintains JAX arrays on correct device."""
    sim = Sim(n_worlds=2, n_drones=3, control=Control.state, device=device)
    data = sim.build_data()
    cmd = np.random.rand(sim.n_worlds, sim.n_drones, 13)
    data = F.state_control(data, cmd)
    controls = data.controls.state
    assert isinstance(controls.cmd, jnp.ndarray), "Buffers must remain JAX arrays"
    assert isinstance(controls.staged_cmd, jnp.ndarray), "Buffers must remain JAX arrays"
    assert jnp.all(controls.staged_cmd == cmd), "Buffers must match command"


@pytest.mark.unit
@pytest.mark.parametrize("control", Control)
def test_functional_controllable(control: Control):
    """Test that functional controllable function works correctly."""
    sim = Sim(n_worlds=2, n_drones=3, control=control)
    data = sim.build_data()
    controllable = F.controllable(data)
    assert isinstance(controllable, jnp.ndarray), "Controllable must be a JAX array"
    shape = controllable.shape
    des_shape = (sim.n_worlds, 1)
    assert shape == des_shape, f"Controllable shape must be {des_shape}, got {shape}"
