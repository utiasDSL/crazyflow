"""Second-order fitted RPY dynamics with first-order thrust dynamics.

This module extends the ``so_rpy`` dynamics by adding a thrust state that models motor spin-up and
spin-down with a first-order lag. Rotational dynamics are still modelled as a fitted second-order
linear system driven by RPY commands.

The command interface is ``[roll_rad, pitch_rad, yaw_rad, thrust_N]``. The ``rotor_vel`` state holds
the thrust in Newtons rather than motor RPMs. It keeps the four entries of the shared state layout,
and only the first entry enters the dynamics.

Both a numeric implementation ([dynamics][crazyflow.dynamics.so_rpy_rotor.dynamics]) and symbolic
CasADi implementations ([symbolic_dynamics][crazyflow.dynamics.so_rpy_rotor.symbolic_dynamics],
[symbolic_dynamics_euler][crazyflow.dynamics.so_rpy_rotor.symbolic_dynamics_euler]) are provided.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import casadi as cs
import jax.numpy as jnp
from array_api_compat import array_namespace
from array_api_compat import device as xp_device
from flax.struct import dataclass, field
from scipy.spatial.transform import Rotation as R

import crazyflow.dynamics.symbols as symbols
from crazyflow.dynamics.core import load_params, supports
from crazyflow.dynamics.utils import rotation
from crazyflow.utils import CORE_NDIM_KEY, to_xp

if TYPE_CHECKING:
    from jax import Device

    from crazyflow._typing import Array  # To be changed to array_api_typing later
    from crazyflow.sim.data import SimData


@supports(rotor_dynamics=True)
def dynamics(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    cmd: Array,
    rotor_vel: Array | None = None,
    dist_f: Array | None = None,
    dist_t: Array | None = None,
    *,
    mass: float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array,
    thrust_time_coef: Array,
    acc_coef: Array,
    cmd_f_coef: Array,
    rpy_coef: Array,
    rpy_rates_coef: Array,
    cmd_rpy_coef: Array,
) -> tuple[Array, Array, Array, Array, Array | None]:
    """Fitted linear, second order rpy dynamics with thrust dynamics.

    Converts the state to Euler angles, evaluates ``dynamics_euler``, maps the derivatives back, and
    adds the force/torque disturbances.

    Args:
        pos: Position of the drone (m).
        quat: Quaternion of the drone (xyzw).
        vel: Velocity of the drone (m/s).
        ang_vel: Angular velocity of the drone (rad/s).
        cmd: Roll pitch yaw (rad) and collective thrust (N) command.
        rotor_vel: Thrust state (N) of shape ``(4,)``, of which only the first entry is used. If
            None, the commanded thrust is directly applied (not recommended). If a value is given,
            thrust dynamics are calculated.
        dist_f: Disturbance force (N) in the world frame acting on the CoM.
        dist_t: Disturbance torque (Nm) in the world frame acting on the CoM.

        mass: Mass of the drone (kg).
        gravity_vec: Gravity vector (m/s^2). We assume the gravity vector points downwards, e.g.
            [0, 0, -9.81].
        J: Inertia matrix (kg m^2).
        J_inv: Inverse inertia matrix (1/kg m^2).
        thrust_time_coef: Coefficient for the rotor dynamics (1/s).
        acc_coef: Coefficient for the acceleration (1/s^2).
        cmd_f_coef: Coefficient for the collective thrust (N/rad^2).
        rpy_coef: Coefficient for the roll pitch yaw dynamics (1/s).
        rpy_rates_coef: Coefficient for the roll pitch yaw rates dynamics (1/s^2).
        cmd_rpy_coef: Coefficient for the roll pitch yaw command dynamics (1/s).

    Returns:
        The derivatives (pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot).
    """
    xp = array_namespace(pos)
    # Convert parameters to correct xp framework
    device = xp_device(pos)
    mass, J, J_inv = to_xp(mass, J, J_inv, xp=xp, device=device)

    # Convert to the native Euler-angle state, evaluate the core dynamics, then map back
    rot = R.from_quat(quat)
    rpy = rot.as_euler("xyz")
    rpy_rates = rotation.ang_vel2rpy_rates(quat, ang_vel)
    pos_dot, _, vel_dot, rpy_rates_dot, rotor_vel_dot = dynamics_euler(
        pos,
        rpy,
        vel,
        rpy_rates,
        cmd,
        rotor_vel,
        mass=mass,
        gravity_vec=gravity_vec,
        thrust_time_coef=thrust_time_coef,
        acc_coef=acc_coef,
        cmd_f_coef=cmd_f_coef,
        rpy_coef=rpy_coef,
        rpy_rates_coef=rpy_rates_coef,
        cmd_rpy_coef=cmd_rpy_coef,
    )

    if dist_f is not None:
        vel_dot = vel_dot + dist_f / mass  # Adding force disturbances to the state
    quat_dot = rotation.ang_vel2quat_dot(quat, ang_vel)
    ang_vel_dot = rotation.rpy_rates_deriv2ang_vel_deriv(quat, rpy_rates, rpy_rates_dot)
    if dist_t is not None:
        # adding torque disturbances to the state
        # angular acceleration can be converted to total torque given the inertia matrix
        torque = (J @ ang_vel_dot[..., None])[..., 0]
        torque = torque + xp.linalg.cross(ang_vel, (J @ ang_vel[..., None])[..., 0])
        # adding torque. TODO: This should be a linear transformation. Can't we just transform the
        # disturbance torque to an ang_vel_dot summand directly?
        torque = torque + rot.apply(dist_t, inverse=True)
        # back to angular acceleration
        torque = torque - xp.linalg.cross(ang_vel, (J @ ang_vel[..., None])[..., 0])
        ang_vel_dot = (J_inv @ torque[..., None])[..., 0]

    return pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot


def dynamics_euler(
    pos: Array,
    rpy: Array,
    vel: Array,
    rpy_rates: Array,
    cmd: Array,
    rotor_vel: Array | None = None,
    *,
    mass: float,
    gravity_vec: Array,
    thrust_time_coef: Array,
    acc_coef: Array,
    cmd_f_coef: Array,
    rpy_coef: Array,
    rpy_rates_coef: Array,
    cmd_rpy_coef: Array,
) -> tuple[Array, Array, Array, Array, Array | None]:
    """Core fitted second-order rpy and thrust dynamics in Euler-angle coordinates."""
    xp = array_namespace(pos)
    device = xp_device(pos)
    mass, gravity_vec = to_xp(mass, gravity_vec, xp=xp, device=device)
    thrust_time_coef, acc_coef = to_xp(thrust_time_coef, acc_coef, xp=xp, device=device)
    cmd_f_coef, rpy_coef = to_xp(cmd_f_coef, rpy_coef, xp=xp, device=device)
    rpy_rates_coef, cmd_rpy_coef = to_xp(rpy_rates_coef, cmd_rpy_coef, xp=xp, device=device)
    cmd_f = cmd[..., -1]
    cmd_rpy = cmd[..., 0:3]
    # Note that we are abusing the rotor_vel state as the thrust
    if rotor_vel is None:
        warnings.warn("Rotor velocity not provided, using commanded rotor velocity.")
        rotor_vel, rotor_vel_dot = cmd_f[..., None], None
    else:
        rotor_vel_dot = (cmd_f[..., None] - rotor_vel) / thrust_time_coef
    forces_motor = rotor_vel[..., 0:1]  # (..., 1)
    thrust = acc_coef + cmd_f_coef * forces_motor
    drone_z_axis = R.from_euler("xyz", rpy).as_matrix()[..., -1]
    pos_dot = vel
    vel_dot = 1.0 / mass * thrust * drone_z_axis + gravity_vec
    rpy_rates_dot = rpy_coef * rpy + rpy_rates_coef * rpy_rates + cmd_rpy_coef * cmd_rpy
    return pos_dot, rpy_rates, vel_dot, rpy_rates_dot, rotor_vel_dot


def symbolic_dynamics(
    model_rotor_vel: bool = True,
    model_dist_f: bool = False,
    model_dist_t: bool = False,
    *,
    mass: float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array,
    thrust_time_coef: Array,
    acc_coef: Array,
    cmd_f_coef: Array,
    rpy_coef: Array,
    rpy_rates_coef: Array,
    cmd_rpy_coef: Array,
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """Return CasADi symbolic expressions for the so_rpy_rotor dynamics in quaternion form.

    Internally delegates to
    [symbolic_dynamics_euler][crazyflow.dynamics.so_rpy_rotor.symbolic_dynamics_euler] and converts
    the Euler-angle state to quaternion + angular-velocity state so that the interface matches that
    of [symbolic_dynamics][crazyflow.dynamics.first_principles.symbolic_dynamics].

    Args:
        model_rotor_vel: If ``True``, the thrust state is included in ``X`` and first-order thrust
            dynamics are modelled. Defaults to ``True``.
        model_dist_f: If ``True``, a 3-D force disturbance is appended to ``X``.
        model_dist_t: If ``True``, a 3-D torque disturbance is appended to ``X``.
        mass: Drone mass in kg.
        gravity_vec: Gravity vector, shape ``(3,)``.
        J: Inertia matrix, shape ``(3, 3)``.
        J_inv: Inverse inertia matrix, shape ``(3, 3)``.
        thrust_time_coef: First-order thrust lag time constant coefficient (1/s).
        acc_coef: Scalar acceleration offset coefficient.
        cmd_f_coef: Collective-thrust-to-acceleration coefficient.
        rpy_coef: RPY state feedback coefficient, shape ``(3,)``.
        rpy_rates_coef: RPY-rate feedback coefficient, shape ``(3,)``.
        cmd_rpy_coef: RPY command feedforward coefficient, shape ``(3,)``.

    Returns:
        Tuple ``(X_dot, X, U, Y)`` of CasADi ``MX`` expressions:

        * ``X_dot``: State derivative, length 17 when ``model_rotor_vel=True`` (13 otherwise), plus
            3 per enabled disturbance.
        * ``X``: State vector ``[pos(3), quat(4), vel(3), ang_vel(3)]``, with ``rotor_vel(4)``
            appended if ``model_rotor_vel=True``. Note that ``rotor_vel`` here represents the thrust
            state in Newtons, and that only its first entry enters the dynamics.
        * ``U``: Input vector ``[roll_rad, pitch_rad, yaw_rad, thrust_N]``.
        * ``Y``: Output ``[pos(3), quat(4)]``.
    """
    ## We need to set the rpy and drpy symbols before building the euler dynamics
    _saved_rpy = symbols.rpy
    _saved_drpy = symbols.drpy
    _rpy_quat = rotation.cs_quat2euler(symbols.quat)
    _drpy_quat = rotation.cs_ang_vel2rpy_rates(symbols.quat, symbols.ang_vel)
    symbols.rpy = _rpy_quat
    symbols.drpy = _drpy_quat
    X_dot_euler, X_euler, U_euler, Y_euler = symbolic_dynamics_euler(
        model_rotor_vel=model_rotor_vel,
        mass=mass,
        gravity_vec=gravity_vec,
        J=J,
        J_inv=J_inv,
        thrust_time_coef=thrust_time_coef,
        acc_coef=acc_coef,
        cmd_f_coef=cmd_f_coef,
        rpy_coef=rpy_coef,
        rpy_rates_coef=rpy_rates_coef,
        cmd_rpy_coef=cmd_rpy_coef,
    )
    symbols.rpy = _saved_rpy
    symbols.drpy = _saved_drpy

    # States and Inputs
    X = cs.vertcat(symbols.pos, symbols.quat, symbols.vel, symbols.ang_vel)
    if model_rotor_vel:
        X = cs.vertcat(X, symbols.rotor_vel)
    if model_dist_f:
        X = cs.vertcat(X, symbols.dist_f)
    if model_dist_t:
        X = cs.vertcat(X, symbols.dist_t)
    U = U_euler

    # Linear equation of motion
    pos_dot = X_dot_euler[0:3]
    vel_dot = X_dot_euler[6:9]
    if model_dist_f:
        # Adding force disturbances to the state
        vel_dot = vel_dot + symbols.dist_f / mass

    # Rotational equation of motion
    xi = cs.vertcat(
        cs.horzcat(0, -symbols.ang_vel.T), cs.horzcat(symbols.ang_vel, -cs.skew(symbols.ang_vel))
    )
    quat_dot = 0.5 * (xi @ symbols.quat)
    ang_vel_dot = rotation.cs_rpy_rates_deriv2ang_vel_deriv(
        symbols.quat, _drpy_quat, X_dot_euler[9:12]
    )
    if model_dist_t:
        # adding torque disturbances to the state
        # angular acceleration can be converted to total torque
        torque = J @ ang_vel_dot + cs.cross(symbols.ang_vel, J @ symbols.ang_vel)
        # adding torque
        torque = torque + symbols.rot.T @ symbols.dist_t
        # back to angular acceleration
        ang_vel_dot = J_inv @ (torque - cs.cross(symbols.ang_vel, J @ symbols.ang_vel))

    if model_rotor_vel:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot, X_dot_euler[-4:])
    else:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot)
    Y = cs.vertcat(symbols.pos, symbols.quat)

    return X_dot, X, U, Y


def symbolic_dynamics_euler(
    model_rotor_vel: bool = True,
    *,
    mass: float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array,
    thrust_time_coef: Array,
    acc_coef: Array,
    cmd_f_coef: Array,
    rpy_coef: Array,
    rpy_rates_coef: Array,
    cmd_rpy_coef: Array,
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """Return CasADi symbolic expressions for the so_rpy_rotor dynamics in Euler-angle form.

    This is the native representation of the ``so_rpy_rotor`` dynamics.  The state uses
    roll/pitch/yaw and their rates rather than quaternion + angular velocity, which avoids
    trigonometric overhead inside CasADi-based solvers.

    Args:
        model_rotor_vel: If ``True``, the thrust state is included in ``X`` and first-order thrust
            dynamics are modelled. Defaults to ``True``.
        mass: Drone mass in kg.
        gravity_vec: Gravity vector, shape ``(3,)``.
        J: Inertia matrix, shape ``(3, 3)``.
        J_inv: Inverse inertia matrix, shape ``(3, 3)``.
        thrust_time_coef: First-order thrust lag time constant coefficient (1/s).
        acc_coef: Scalar acceleration offset coefficient.
        cmd_f_coef: Collective-thrust-to-acceleration coefficient.
        rpy_coef: RPY state feedback coefficient, shape ``(3,)``.
        rpy_rates_coef: RPY-rate feedback coefficient, shape ``(3,)``.
        cmd_rpy_coef: RPY command feedforward coefficient, shape ``(3,)``.

    Returns:
        Tuple ``(X_dot, X, U, Y)`` of CasADi ``MX`` expressions:

        * ``X_dot``: State derivative, length 16 when ``model_rotor_vel=True`` (12 otherwise).
        * ``X``: State vector ``[pos(3), rpy(3), vel(3), drpy(3)]``, with ``rotor_vel(4)`` appended
            if ``model_rotor_vel=True``. Note that ``rotor_vel`` here represents the thrust state in
            Newtons, and that only its first entry enters the dynamics.
        * ``U``: Input vector ``[roll_rad, pitch_rad, yaw_rad, thrust_N]``.
        * ``Y``: Output ``[pos(3), rpy(3)]``.
    """
    # States and Inputs
    X = cs.vertcat(symbols.pos, symbols.rpy, symbols.vel, symbols.drpy)
    if model_rotor_vel:
        X = cs.vertcat(X, symbols.rotor_vel)
    U = symbols.cmd_rpyt
    cmd_rpy = U[:3]
    cmd_thrust = U[-1]
    rot = rotation.cs_rpy2matrix(symbols.rpy)

    # Defining the dynamics function
    # Note that we are abusing the rotor_vel state as the thrust
    if model_rotor_vel:
        rotor_vel_dot = 1 / thrust_time_coef * (cmd_thrust - symbols.rotor_vel)
        forces_motor = symbols.rotor_vel[0]  # We are only using the first element
    else:
        forces_motor = cmd_thrust

    # Creating force vector
    forces_motor_vec = cs.vertcat(0, 0, acc_coef + cmd_f_coef * forces_motor)

    # Linear equation of motion
    pos_dot = symbols.vel
    vel_dot = rot @ forces_motor_vec / mass + gravity_vec

    ddrpy = rpy_coef * symbols.rpy + rpy_rates_coef * symbols.drpy + cmd_rpy_coef * cmd_rpy

    if model_rotor_vel:
        X_dot = cs.vertcat(pos_dot, symbols.drpy, vel_dot, ddrpy, rotor_vel_dot)
    else:
        X_dot = cs.vertcat(pos_dot, symbols.drpy, vel_dot, ddrpy)
    Y = cs.vertcat(symbols.pos, symbols.rpy)

    return X_dot, X, U, Y


@dataclass
class Params:
    mass: Array = field(metadata={CORE_NDIM_KEY: 1})  # (1,)
    """Mass of the drone."""
    gravity_vec: Array = field(metadata={CORE_NDIM_KEY: 1})  # (3,)
    """Gravity vector of the drone."""
    J: Array = field(metadata={CORE_NDIM_KEY: 2})  # (3, 3)
    """Inertia matrix of the drone."""
    J_inv: Array = field(metadata={CORE_NDIM_KEY: 2})  # (3, 3)
    """Inverse of the inertia matrix of the drone."""
    thrust_time_coef: Array = field(metadata={CORE_NDIM_KEY: 1})  # (1,)
    """Rotor coefficient of the drone."""
    acc_coef: Array = field(metadata={CORE_NDIM_KEY: 1})  # (1,)
    """Acceleration coefficient of the drone."""
    cmd_f_coef: Array = field(metadata={CORE_NDIM_KEY: 1})  # (1,)
    """Collective thrust coefficient of the drone."""
    rpy_coef: Array = field(metadata={CORE_NDIM_KEY: 1})  # (3,)
    """Roll pitch yaw coefficient of the drone."""
    rpy_rates_coef: Array = field(metadata={CORE_NDIM_KEY: 1})  # (3,)
    """Roll pitch yaw rates coefficient of the drone."""
    cmd_rpy_coef: Array = field(metadata={CORE_NDIM_KEY: 1})  # (3,)
    """Roll pitch yaw command coefficient of the drone."""

    @staticmethod
    def create(drone: str, device: Device) -> Params:
        """Create the default parameters for the simulation."""
        p = load_params(dynamics, drone)
        J = jnp.asarray(p["J"], device=device)
        return Params(
            mass=jnp.asarray([p["mass"]], device=device),
            gravity_vec=jnp.asarray(p["gravity_vec"], device=device),
            J=J,
            J_inv=jnp.linalg.inv(J),
            thrust_time_coef=jnp.asarray([p["thrust_time_coef"]], device=device),
            acc_coef=jnp.asarray([p["acc_coef"]], device=device),
            cmd_f_coef=jnp.asarray([p["cmd_f_coef"]], device=device),
            rpy_coef=jnp.asarray(p["rpy_coef"], device=device),
            rpy_rates_coef=jnp.asarray(p["rpy_rates_coef"], device=device),
            cmd_rpy_coef=jnp.asarray(p["cmd_rpy_coef"], device=device),
        )


def sim_dynamics(data: SimData) -> SimData:
    """Compute the forces and torques from the so_rpy_rotor dynamics."""
    params: Params = data.params
    vel, _, acc, ang_acc, rotor_acc = dynamics(
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
    states_deriv = data.states_deriv.replace(
        vel=vel, ang_vel=data.states.ang_vel, acc=acc, ang_acc=ang_acc, rotor_acc=rotor_acc
    )
    return data.replace(states_deriv=states_deriv)
