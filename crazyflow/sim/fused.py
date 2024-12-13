"""Fused versions of the dynamics and controller functions.

Writing fused versions that directly operate on the state and control structs of the simulation
allows us to avoid indexing into the structs and to update the structs within jit-compiled code.
"""

from crazyflow.sim.physics import analytical_dynamics, identified_dynamics, rpms2collective_wrench
from crazyflow.sim.structs import SimData


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
        controls.attitude, pos, quat, vel, rpy_rates, 1 / data.core.freq
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
        forces, torques, pos, quat, vel, rpy_rates, params.mass, params.J_INV, 1 / data.core.freq
    )
    return data.replace(states=states.replace(pos=pos, quat=quat, vel=vel, rpy_rates=rpy_rates))
