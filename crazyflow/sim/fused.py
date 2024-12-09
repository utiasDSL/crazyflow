"""Fused versions of the dynamics and controller functions.

Writing fused versions that directly operate on the state and control structs of the simulation
allows us to avoid indexing into the structs and to update the structs within jit-compiled code.
"""

from jax import Array

from crazyflow.control.controller import state2attitude as state2attitude_ctrl
from crazyflow.sim.physics import analytical_dynamics, identified_dynamics, rpms2collective_wrench
from crazyflow.sim.structs import SimControls, SimData, SimParams, SimState


def fused_identified_dynamics(data: SimData) -> SimData:
    """Dynamics model identified from data collected on the real drone.

    Note:
        Fused version of `crazyflow.sim.physics.identified_dynamics`. See `crazyflow.sim.fused` for
        more details.

    Args:
        state: The current simulation state.
        cmd: The current simulation controls.
        dt: The simulation time step.
    """
    states, controls = data.states, data.controls
    pos, quat, vel, rpy_rates = states.pos, states.quat, states.vel, states.rpy_rates
    pos, quat, vel, rpy_rates = identified_dynamics(
        controls.attitude, pos, quat, vel, rpy_rates, 1 / data.sim.freq
    )
    return data.replace(states=states.replace(pos=pos, quat=quat, vel=vel, rpy_rates=rpy_rates))


def fused_analytical_dynamics(data: SimData) -> SimData:
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
    states, controls, params = data.states, data.controls, data.params
    forces, torques = rpms2collective_wrench(controls.rpms, states.quat, states.rpy_rates, params.J)
    # The dynamics model only considers the force and torque at the center of mass, not the motors.
    pos, quat, vel, rpy_rates = states.pos, states.quat, states.vel, states.rpy_rates
    pos, quat, vel, rpy_rates = analytical_dynamics(
        forces, torques, pos, quat, vel, rpy_rates, params.mass, params.J_INV, 1 / data.sim.freq
    )
    return data.replace(states=states.replace(pos=pos, quat=quat, vel=vel, rpy_rates=rpy_rates))


def state2attitude(data: SimData) -> SimData:
    """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

    Note:
        Fused version of `crazyflow.control.controller.state2attitude`. See `crazyflow.sim.fused`
        for more details.

    Warning:
        We write the results to `cmd.staged_attitude` instead of `cmd.attitude`. If you want to
        apply the results, you need to update `cmd.attitude` with `cmd.staged_attitude`.

    Args:
        mask: A boolean array of shape (n_worlds, ). Internally, we broadcast over all drones.
        state: The current simulation state.
        cmd: The current simulation controls.
        dt: The simulation time step.
    """
    states, controls = data.states, data.controls
    des_pos, des_vel = controls.state[..., :3], controls.state[..., 3:6]
    des_yaw = controls.state[..., 9]
    dt = 1 / data.sim.freq
    attitude, pos_err_i = state2attitude_ctrl(
        states.pos, states.vel, states.quat, des_pos, des_vel, des_yaw, controls.pos_err_i, dt
    )
    return data.replace(controls=controls.replace(staged_attitude=attitude, pos_err_i=pos_err_i))


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
