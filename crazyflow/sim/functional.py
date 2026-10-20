from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from crazyflow.control import Control
from crazyflow.control.core import controllable as _controllable

if TYPE_CHECKING:
    from jax import Array

    from crazyflow.sim.data import SimData


def state_control(data: SimData, controls: Array) -> SimData:
    """State control function."""
    assert data.controls.mode == Control.state, f"control type {data.controls.mode} not enabled"
    assert controls.shape == (data.core.n_worlds, data.core.n_drones, 16), "controls shape mismatch"
    assert data.controls.state is not None, "state control data not initialized"
    controls = jnp.asarray(controls)
    data = data.replace(
        controls=data.controls.replace(state=data.controls.state.replace(staged_cmd=controls))
    )
    return data


def attitude_control(data: SimData, controls: Array) -> SimData:
    """Attitude control function.

    We need to stage the attitude controls because the so_rpy dynamics mode operates directly on
    the attitude controls. If we were to directly update the controls, this would effectively
    bypass the control frequency and run the attitude controller at the dynamics update rate. By
    staging the controls, we ensure that the dynamics module sees the old controls until the
    controller updates at its correct frequency.
    """
    assert data.controls.mode == Control.attitude, f"control type {data.controls.mode} not enabled"
    assert controls.shape == (data.core.n_worlds, data.core.n_drones, 4), "controls shape mismatch"
    assert data.controls.attitude is not None, "attitude control data not initialized"
    controls = jnp.asarray(controls)
    data = data.replace(
        controls=data.controls.replace(attitude=data.controls.attitude.replace(staged_cmd=controls))
    )
    return data


def body_rate_control(data: SimData, controls: Array) -> SimData:
    """Body rate control function."""
    assert data.controls.mode == Control.body_rate, f"control type {data.controls.mode} not enabled"
    assert controls.shape == (data.core.n_worlds, data.core.n_drones, 4), "controls shape mismatch"
    assert data.controls.body_rate is not None, "body_rate control data not initialized"
    controls = jnp.asarray(controls)
    return data.replace(
        controls=data.controls.replace(
            body_rate=data.controls.body_rate.replace(staged_cmd=controls)
        )
    )


def force_torque_control(data: SimData, controls: Array) -> SimData:
    """Force-torque control function."""
    assert data.controls.mode == Control.force_torque, (
        f"control type {data.controls.mode} not enabled"
    )
    assert controls.shape == (data.core.n_worlds, data.core.n_drones, 4), "controls shape mismatch"
    assert data.controls.force_torque is not None, "force_torque control data not initialized"
    controls = jnp.asarray(controls)
    data = data.replace(
        controls=data.controls.replace(
            force_torque=data.controls.force_torque.replace(staged_cmd=controls)
        )
    )
    return data


def rotor_vel_control(data: SimData, controls: Array) -> SimData:
    """Rotor velocity control.

    Directly set the desired rotor velocities of the drone.
    """
    assert data.controls.mode == Control.rotor_vel, f"control type {data.controls.mode} not enabled"
    assert controls.shape == (data.core.n_worlds, data.core.n_drones, 4), "controls shape mismatch"
    controls = jnp.asarray(controls)
    return data.replace(controls=data.controls.replace(rotor_vel=controls))


def controllable(data: SimData) -> Array:
    """Check which worlds can currently update their controllers."""
    controls = data.controls
    match data.controls.mode:
        case Control.state:
            control = controls.state
        case Control.attitude:
            control = controls.attitude
        case Control.body_rate:
            control = controls.body_rate
        case Control.force_torque:
            control = controls.force_torque
        case Control.rotor_vel:
            return jnp.ones_like(data.core.steps, dtype=bool)
        case _:
            raise NotImplementedError(f"Control mode {data.controls.mode} not implemented")
    assert control is not None, f"Control mode {data.controls.mode} has no initialized data"
    return _controllable(data.core.steps, data.core.freq, control.steps, control.freq)
