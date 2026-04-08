from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from crazyflow.control import Control
from crazyflow.control.control import controllable as _controllable
from crazyflow.utils import to_device

if TYPE_CHECKING:
    from jax import Array

    from crazyflow.sim.data import SimData


def state_control(data: SimData, controls: Array) -> SimData:
    """State control function."""
    assert data.controls.mode == Control.state, f"control type {data.controls.mode} not enabled"
    assert controls.shape == (data.core.n_worlds, data.core.n_drones, 13), "controls shape mismatch"
    controls = to_device(controls, data.core.device)
    data = data.replace(
        controls=data.controls.replace(state=data.controls.state.replace(staged_cmd=controls))
    )
    return data


def attitude_control(data: SimData, controls: Array) -> SimData:
    """Attitude control function.

    We need to stage the attitude controls because the sys_id physics mode operates directly on
    the attitude controls. If we were to directly update the controls, this would effectively
    bypass the control frequency and run the attitude controller at the physics update rate. By
    staging the controls, we ensure that the physics module sees the old controls until the
    controller updates at its correct frequency.
    """
    assert data.controls.mode == Control.attitude, f"control type {data.controls.mode} not enabled"
    assert controls.shape == (data.core.n_worlds, data.core.n_drones, 4), "controls shape mismatch"
    controls = to_device(controls, data.core.device)
    data = data.replace(
        controls=data.controls.replace(attitude=data.controls.attitude.replace(staged_cmd=controls))
    )
    return data


def force_torque_control(data: SimData, controls: Array) -> SimData:
    """Force-torque control function."""
    assert data.controls.mode == Control.force_torque, (
        f"control type {data.controls.mode} not enabled"
    )
    assert controls.shape == (data.core.n_worlds, data.core.n_drones, 4), "controls shape mismatch"
    controls = to_device(controls, data.core.device)
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
    controls = to_device(controls, data.core.device)
    return data.replace(controls=data.controls.replace(rotor_vel=controls))


def controllable(data: SimData) -> Array:
    """Check which worlds can currently update their controllers."""
    controls = data.controls
    match data.controls.mode:
        case Control.state:
            control_steps, control_freq = controls.state.steps, controls.state.freq
        case Control.attitude:
            control_steps, control_freq = controls.attitude.steps, controls.attitude.freq
        case Control.force_torque:
            control_steps = controls.force_torque.steps
            control_freq = controls.force_torque.freq
        case Control.rotor_vel:
            return jnp.ones((data.core.n_worlds, 1), dtype=bool, device=data.core.device)
        case _:
            raise NotImplementedError(f"Control mode {data.controls.mode} not implemented")
    return _controllable(data.core.steps, data.core.freq, control_steps, control_freq)
