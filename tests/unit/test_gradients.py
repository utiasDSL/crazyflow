from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from crazyflow.sim import Sim
from crazyflow.sim.data import Control, SimData
from crazyflow.sim.physics import Physics


@pytest.mark.skip(reason="State needs SVD in from_matrix, which is not differentiable.")
@pytest.mark.unit
@pytest.mark.parametrize("physics", Physics)
def test_state_cmd_gradients(physics: Physics):
    sim = Sim(physics=physics, control=Control.state, freq=500)
    sim_step = sim._step

    def step(cmd: Array, data: SimData) -> Array:
        data = data.replace(
            controls=data.controls.replace(state=data.controls.state.replace(staged_cmd=cmd))
        )
        data = sim_step(data, sim.freq // sim.control_freq)
        return (data.states.pos[0, 0, 2] - 1.0) ** 2  # Quadratic cost to reach 1m height

    step_grad = jax.jit(jax.grad(step))

    cmd = jnp.zeros((1, 1, 13), dtype=jnp.float32)
    cmd = cmd.at[..., 3].set(0.3)

    grad = step_grad(cmd, sim.data)
    assert not jnp.any(jnp.isnan(grad))


@pytest.mark.unit
@pytest.mark.parametrize("physics", Physics)
def test_attitude_cmd_gradients(physics: Physics):
    sim = Sim(physics=physics, control=Control.attitude, freq=500)

    def step(cmd: Array, data: SimData) -> Array:
        data = data.replace(
            controls=data.controls.replace(attitude=data.controls.attitude.replace(staged_cmd=cmd))
        )
        data = sim._step(data, 10)
        return (data.states.pos[0, 0, 2] - 1.0) ** 2  # Quadratic cost to reach 1m height

    step_grad = jax.jit(jax.grad(step))

    cmd = jnp.zeros((1, 1, 4), dtype=jnp.float32)
    cmd = cmd.at[..., 3].set(0.3)

    grad = step_grad(cmd, sim.data)
    assert not jnp.any(jnp.isnan(grad))


@pytest.mark.unit
def test_force_torque_cmd_gradients():
    sim = Sim(physics=Physics.first_principles, control=Control.force_torque, freq=500)

    def step(cmd: Array, data: SimData) -> Array:
        data = data.replace(
            controls=data.controls.replace(
                force_torque=data.controls.force_torque.replace(staged_cmd=cmd)
            )
        )
        data = sim._step(data, 10)
        return (data.states.pos[0, 0, 2] - 1.0) ** 2  # Quadratic cost to reach 1m height

    step_grad = jax.jit(jax.grad(step))

    cmd = jnp.zeros((1, 1, 4), dtype=jnp.float32)
    cmd = cmd.at[..., 0].set(0.3)

    grad = step_grad(cmd, sim.data)
    assert not jnp.any(jnp.isnan(grad))
