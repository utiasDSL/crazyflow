"""Wrappers around the drone dynamics for `SimData` compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from crazyflow.dynamics import first_principles, so_rpy, so_rpy_rotor, so_rpy_rotor_drag
from crazyflow.sim.data import SimStateDeriv

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData


def first_principles_dynamics(data: SimData) -> SimStateDeriv:
    """Wrap the first principles dynamics."""
    params = data.params
    assert isinstance(params, first_principles.Params), "Params do not match the dynamics mode"
    vel, _, acc, ang_acc, rotor_acc = first_principles.dynamics(
        pos=data.states.pos,
        quat=data.states.quat,
        vel=data.states.vel,
        ang_vel=data.states.ang_vel,
        cmd=data.controls.rotor_vel,
        rotor_vel=data.states.rotor_vel,
        dist_f=data.states.force,
        dist_t=data.states.torque,
        **params.__dict__,
    )
    return SimStateDeriv(
        vel=vel, ang_vel=data.states.ang_vel, acc=acc, ang_acc=ang_acc, rotor_acc=rotor_acc
    )


def so_rpy_dynamics(data: SimData) -> SimStateDeriv:
    """Wrap the so_rpy dynamics."""
    params = data.params
    assert isinstance(params, so_rpy.Params), "Params do not match the dynamics mode"
    assert data.controls.attitude is not None, "attitude control data not initialized"
    vel, _, acc, ang_acc = so_rpy.dynamics(
        pos=data.states.pos,
        quat=data.states.quat,
        vel=data.states.vel,
        ang_vel=data.states.ang_vel,
        cmd=data.controls.attitude.cmd,
        dist_f=data.states.force,
        dist_t=data.states.torque,
        **params.__dict__,
    )
    rotor_acc = jnp.zeros_like(data.states.rotor_vel)
    return SimStateDeriv(
        vel=vel, ang_vel=data.states.ang_vel, acc=acc, ang_acc=ang_acc, rotor_acc=rotor_acc
    )


def so_rpy_rotor_dynamics(data: SimData) -> SimStateDeriv:
    """Wrap the so_rpy_rotor dynamics."""
    params = data.params
    assert isinstance(params, so_rpy_rotor.Params), "Params do not match the dynamics mode"
    assert data.controls.attitude is not None, "attitude control data not initialized"
    vel, _, acc, ang_acc, rotor_acc = so_rpy_rotor.dynamics(
        pos=data.states.pos,
        quat=data.states.quat,
        vel=data.states.vel,
        ang_vel=data.states.ang_vel,
        rotor_vel=data.states.rotor_vel,
        cmd=data.controls.attitude.cmd,
        dist_f=data.states.force,
        dist_t=data.states.torque,
        **params.__dict__,
    )
    return SimStateDeriv(
        vel=vel, ang_vel=data.states.ang_vel, acc=acc, ang_acc=ang_acc, rotor_acc=rotor_acc
    )


def so_rpy_rotor_drag_dynamics(data: SimData) -> SimStateDeriv:
    """Wrap the so_rpy_rotor_drag dynamics."""
    params = data.params
    assert isinstance(params, so_rpy_rotor_drag.Params), "Params do not match the dynamics mode"
    assert data.controls.attitude is not None, "attitude control data not initialized"
    vel, _, acc, ang_acc, rotor_acc = so_rpy_rotor_drag.dynamics(
        pos=data.states.pos,
        quat=data.states.quat,
        vel=data.states.vel,
        ang_vel=data.states.ang_vel,
        cmd=data.controls.attitude.cmd,
        rotor_vel=data.states.rotor_vel,
        dist_f=data.states.force,
        dist_t=data.states.torque,
        **params.__dict__,
    )
    return SimStateDeriv(
        vel=vel, ang_vel=data.states.ang_vel, acc=acc, ang_acc=ang_acc, rotor_acc=rotor_acc
    )
