"""Fused versions of the dynamics and controller functions.

Writing fused versions that directly operate on the state and control structs of the simulation
allows us to avoid indexing into the structs and to update the structs within jit-compiled code.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.control.controller import attitude2rpm, state2attitude
from crazyflow.sim.physics import analytical_dynamics, identified_dynamics, rpms2collective_wrench
from crazyflow.sim.structs import SimControls, SimParams, SimState


@jax.jit
@partial(jax.vmap, in_axes=(0, 0, None))
@partial(jax.vmap, in_axes=(0, 0, None))
def fused_identified_dynamics(state: SimState, cmd: SimControls, dt: float) -> SimState:
    """Dynamics model identified from data collected on the real drone.

    Note:
        Fused version of `crazyflow.sim.physics.identified_dynamics`. See `crazyflow.sim.fused` for
        more details.

    Args:
        state: The current simulation state.
        cmd: The current simulation controls.
        dt: The simulation time step.
    """
    pos, quat, vel, rpy_rates = identified_dynamics(
        cmd.attitude, state.pos, state.quat, state.vel, state.rpy_rates, dt
    )
    return state.replace(pos=pos, quat=quat, vel=vel, rpy_rates=rpy_rates)


@jax.jit
@partial(jax.vmap, in_axes=(0, 0, 0, 0, None))
@partial(jax.vmap, in_axes=(0, 0, 0, 0, None))
def fused_analytical_dynamics(
    forces: Array, torques: Array, state: SimState, params: SimParams, dt: float
) -> SimState:
    """Dynamics model based on the physical parameters of the drone.

    Note:
        Fused version of `crazyflow.sim.physics.analytical_dynamics`. See `crazyflow.sim.fused` for
        more details.

    Args:
        forces: The forces applied to the center of mass of the drone in the global frame.
        torques: The torques applied to the center of mass of the drone in the global frame.
        state: The current simulation state.
        params: The current simulation parameters.
        dt: The simulation time step.
    """
    pos, quat, vel, rpy_rates = state.pos, state.quat, state.vel, state.rpy_rates
    pos, quat, vel, rpy_rates = analytical_dynamics(
        forces, torques, pos, quat, vel, rpy_rates, params.mass, params.J_INV, dt
    )
    return state.replace(pos=pos, quat=quat, vel=vel, rpy_rates=rpy_rates)


@jax.jit
@partial(jax.vmap, in_axes=(0, 0, 0, None))
@partial(jax.vmap, in_axes=(None, 0, 0, None))
def fused_masked_state2attitude(
    mask: Array, state: SimState, cmd: SimControls, dt: float
) -> SimControls:
    """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

    Note:
        Fused version of `crazyflow.control.controller.state2attitude`. See `crazyflow.sim.fused`
        for more details.

    Args:
        mask: A boolean array of shape (n_worlds, ). Internally, we broadcast over all drones.
        state: The current simulation state.
        cmd: The current simulation controls.
        dt: The simulation time step.
    """
    des_pos, des_vel, des_yaw = cmd.state[:3], cmd.state[3:6], cmd.state[9].reshape((1,))
    attitude, pos_err_i = state2attitude(
        state.pos, state.vel, state.quat, des_pos, des_vel, des_yaw, cmd.pos_err_i, dt
    )
    # Non-branching selection depending on the mask. XLA should be able to optimize a short path
    # that avoids computing the above expressions when the mask is false.
    attitude = jnp.where(mask, attitude, cmd.attitude)
    pos_err_i = jnp.where(mask, pos_err_i, cmd.pos_err_i)
    return cmd.replace(attitude=attitude, pos_err_i=pos_err_i)


@jax.jit
@partial(jax.vmap, in_axes=(0, 0, 0, None))
@partial(jax.vmap, in_axes=(None, 0, 0, None))
def fused_masked_attitude2rpm(
    mask: Array, state: SimState, cmd: SimControls, dt: float
) -> SimControls:
    """Compute the next desired RPMs of the drone.

    Note:
        Fused version of `crazyflow.control.controller.attitude2rpm`. See `crazyflow.sim.fused` for
        more details.

    Args:
        mask: A boolean array of shape (n_worlds, ). Internally, we broadcast over all drones.
        state: The current simulation state.
        cmd: The current simulation controls.
        dt: The simulation time step.
    """
    rpms, rpy_err_i = attitude2rpm(cmd.attitude, state.quat, cmd.last_rpy, cmd.rpy_err_i, dt)
    # Non-branching selection depending on the mask. See fused_masked_state2attitude for more info.
    rpms = jnp.where(mask, rpms, cmd.rpms)
    rpy_err_i = jnp.where(mask, rpy_err_i, cmd.rpy_err_i)
    last_rpy = jnp.where(mask, R.from_quat(state.quat).as_euler("xyz"), cmd.last_rpy)
    return cmd.replace(rpms=rpms, rpy_err_i=rpy_err_i, last_rpy=last_rpy)


@jax.jit
@jax.vmap
@jax.vmap
def fused_rpms2collective_wrench(
    states: SimState, cmd: SimControls, params: SimParams
) -> tuple[Array, Array]:
    """Convert RPMs to forces and torques in the global frame.

    Note:
        Fused version of `crazyflow.sim.physics.rpms2collective_wrench`. See `crazyflow.sim.fused`
        for more details.

    Args:
        states: The current simulation states.
        cmd: The current simulation controls.
        params: The current simulation parameters.
    """
    forces, torques = rpms2collective_wrench(cmd.rpms, states.quat, states.rpy_rates, params.J)
    return forces, torques
