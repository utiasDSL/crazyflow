"""Mellinger controller reimplementation based on the Crazyflie firmware.

The controller is split into three pure functions that form a pipeline:
``state2attitude`` → ``attitude2force_torque`` → ``force_torque2rotor_vel``.
Each stage can be used independently or chained together to produce per-motor
RPM commands from a full-state setpoint. ``body_rate2force_torque`` replaces
the second stage for body rate setpoints.

Reference: D. Mellinger and V. Kumar, "Minimum snap trajectory generation and
control for quadrotors", ICRA 2011.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import array_api_extra as xpx
import jax.numpy as jnp
from array_api_compat import array_namespace
from flax.struct import PyTreeNode, field
from scipy.spatial.transform import Rotation as R

from crazyflow.control.core import controllable, load_fn_params
from crazyflow.control.transform import force2pwm, motor_force2rotor_vel, pwm2force
from crazyflow.utils import CORE_NDIM_KEY, leaf_replace

if TYPE_CHECKING:
    from jax import Device

    from crazyflow._typing import Array  # To be changed to array_api_typing later
    from crazyflow.sim.data import SimData


def state2attitude(
    pos: Array,
    quat: Array,
    vel: Array,
    cmd: Array,
    pos_err_i: Array | None = None,
    ctrl_freq: float = 100,
    *,
    mass: float,
    kp: Array,
    kd: Array,
    ki: Array,
    gravity_vec: Array,
    mass_thrust: float,
    int_err_max: Array,
    thrust_max: float,
    pwm_max: float,
) -> tuple[Array, Array]:
    """Compute the positional part of the mellinger controller.

    All controllers are implemented as pure functions. Therefore, integral errors have to be passed
    as an argument and returned as well.

    Args:
        pos: Drone position with shape (..., 3).
        quat: Drone orientation as xyzw quaternion with shape (..., 4).
        vel: Drone velocity with shape (..., 3).
        cmd: Full state command in SI units with shape (..., 16). The entries are
            [x, y, z, vx, vy, vz, ax, ay, az, qx, qy, qz, qw, wx, wy, wz]. Only the yaw of the
            attitude quaternion is used. The body rates are forwarded to the attitude controller.
        pos_err_i: Position integral error (..., 3) from the previous call. If None, it is
            initialised to zero.
        ctrl_freq: Control frequency in Hz
        mass: Drone mass used for calculations in the controller in kg.
        kp: Proportional gain for the position controller with shape (3,).
        kd: Derivative gain for the position controller with shape (3,).
        ki: Integral gain for the position controller with shape (3,).
        gravity_vec: Gravity vector with shape (3,). We assume gravity to be in the negative z
            direction. E.g., [0, 0, -9.81].
        mass_thrust: Conversion factor from thrust to PWM.
        int_err_max: Range of the integral error with shape (3,). i_range in the firmware.
        thrust_max: Maximum thrust in N.
        pwm_max: Maximum PWM value.

    Returns:
        The RPY collective thrust command [rad, rad, rad, N], and the integral error of the position
        controller.
    """
    xp = array_namespace(pos)

    setpoint_pos = cmd[..., 0:3]
    setpoint_vel = cmd[..., 3:6]
    setpoint_acc = cmd[..., 6:9]
    setpoint_quat = cmd[..., 9:13]
    dt = 1 / ctrl_freq
    # From firmware controller_mellinger
    pos_err = setpoint_pos - pos  # l. 145 Position Error (ep)
    vel_err = setpoint_vel - vel  # l. 148 Velocity Error (ev)
    # l.151 ff Integral Error
    int_pos_err = xp.zeros_like(pos) if pos_err_i is None else pos_err_i
    int_pos_err = xp.clip(int_pos_err + pos_err * dt, -int_err_max, int_err_max)
    # l. 161 Desired thrust [F_des]
    # => only one case here, since setpoint is always in absolute mode
    # Note: since we've defined the gravity in z direction, a "-" needs to be added
    target_thrust = (
        mass * (setpoint_acc - gravity_vec) + kp * pos_err + kd * vel_err + ki * int_pos_err
    )
    # l. 178 Rate-controlled YAW is moving YAW angle setpoint
    # => only one case here, since the setpoint is always in absolute mode
    desired_yaw = R.from_quat(setpoint_quat).as_euler("xyz")[..., 2]
    # l. 189 Z-Axis [zB]
    rot = R.from_quat(quat).as_matrix()
    z_axis = rot[..., -1]  # 3rd column or roation matrix is z axis
    # l. 194 yaw correction (only if position control is not used)
    # => skipped since we always use position control here

    # l. 204 Current thrust [F]
    # Taking the dot product of the last axis:
    current_thrust = xp.vecdot(target_thrust, z_axis, axis=-1)
    # l. 207 Calculate axis [zB_des]
    z_axis_desired = target_thrust / xp.linalg.vector_norm(target_thrust, axis=-1, keepdims=True)
    # l. 210 [xC_des]
    # x_axis_desired = z_axis_desired x [sin(yaw), cos(yaw), 0]^T
    x_c_des_x = xp.cos(desired_yaw)
    x_c_des_y = xp.sin(desired_yaw)
    x_c_des_z = xp.zeros_like(x_c_des_x)
    x_c_des = xp.stack((x_c_des_x, x_c_des_y, x_c_des_z), axis=-1)
    # [yB_des]
    y_axis_desired = xp.linalg.cross(z_axis_desired, x_c_des)
    y_axis_desired = y_axis_desired / xp.linalg.vector_norm(y_axis_desired, axis=-1, keepdims=True)
    # [xB_des]
    x_axis_desired = xp.linalg.cross(y_axis_desired, z_axis_desired)
    # converting desired axis to rotation matrix and then to RPY.
    matrix = xp.stack((x_axis_desired, y_axis_desired, z_axis_desired), axis=-1)
    # l. 220 [eR] The mellinger controller now continues with the attitude controller. However, we
    # decouple the attitude controller from the state controller. We therefore stop here and
    # continue the computation in the attitude2force_torque controller. The conversion to RPY is
    # necessary to pass the command to the attitude2force_torque controller in the correct format.
    #
    # safety: assume_valid is okay here because we just constructed the rotation matrix from
    # orthonormal vectors
    command_RPY = R.from_matrix(matrix, assume_valid=True).as_euler("xyz", degrees=False)
    # l. 283 [control_thrust]
    # The firmware returns thrust in PWM, but we want to stay in SI units. The conversion from
    # thrust to PWM uses a mass_thrust parameter, which is a constant converting thrust values to
    # PWMs. This transformation changes the thrust value, because it is fixed to a specific value
    # instead of dynamically scaling with the mass parameter of the controller! Hence, we include
    # this conversion here and thus effectively rescale the thrust slightly. The conversion below
    # maps thrust -> PWM -> rescaled thrust.
    thrust = pwm2force(mass_thrust * current_thrust, thrust_max * 4, pwm_max)
    command_rpyt = xp.concat((command_RPY, thrust[..., None]), axis=-1)
    return command_rpyt, int_pos_err


def attitude2force_torque(
    quat: Array,
    ang_vel: Array,
    cmd: Array,
    prev_ang_vel: Array | None = None,
    r_int_error: Array | None = None,
    ctrl_freq: int = 500,
    *,
    kR: Array,
    kw: Array,
    ki_m: Array,
    kd_omega: Array,
    int_err_max: Array,
    torque_pwm_max: Array,
    thrust_max: float,
    pwm_min: float,
    pwm_max: float,
    L: float,
    thrust2torque: float,
    mixing_matrix: Array,
) -> tuple[Array, Array, Array]:
    """Compute the attitude to desired force-torque part of the Mellinger controller.

    Note:
        We omit the axis flip in the firmware as it has only been introduced to make the controller
        compatible with the new frame of the Crazyflie 2.1.

    Args:
        quat: Drone orientation as xyzw quaternion with shape (..., 4).
        ang_vel: Drone angular drone velocity in rad/s with shape (..., 3).
        cmd: Commanded attitude (roll, pitch, yaw) and total thrust [rad, rad, rad, N].
        prev_ang_vel: Angular velocity in rad/s from the previous call. If None, it is initialised
            to zero.
        r_int_error: Angular velocity integral error (..., 3) from the previous call. If None, it
            is initialised to zero.
        ctrl_freq: Control frequency in Hz
        kR: Proportional gain for the rotation error with shape (3,).
        kw: Proportional gain for the angular velocity error with shape (3,).
        ki_m: Integral gain for the rotation error with shape (3,).
        kd_omega: Derivative gain for the angular velocity error with shape (3,).
        int_err_max: Range of the integral error with shape (3,). i_range in the firmware.
        torque_pwm_max: Maximum torque in PWM.
        thrust_max: Maximum thrust in N.
        pwm_min: Minimum PWM value.
        pwm_max: Maximum PWM value.
        L: Distance from the center of the quadrotor to the center of the rotor in m.
        thrust2torque: Conversion factor (m).
        mixing_matrix: Mixing matrix for the motor forces with shape (4, 3).

    Returns:
        Desired force (1,), torques (3,) and i_error_m
    """
    xp = array_namespace(quat)
    ang_vel_des = xp.zeros_like(ang_vel)  # Attitude control assumes a zero body rate setpoint
    return _attitude2force_torque(
        quat,
        ang_vel,
        cmd[..., :3],
        ang_vel_des,
        cmd[..., 3],
        prev_ang_vel,
        ang_vel_des,
        r_int_error,
        ctrl_freq,
        kR=kR,
        kw=kw,
        ki_m=ki_m,
        kd_omega=kd_omega,
        int_err_max=int_err_max,
        torque_pwm_max=torque_pwm_max,
        thrust_max=thrust_max,
        pwm_min=pwm_min,
        pwm_max=pwm_max,
        L=L,
        thrust2torque=thrust2torque,
        mixing_matrix=mixing_matrix,
    )


def body_rate2force_torque(
    quat: Array,
    ang_vel: Array,
    cmd: Array,
    prev_ang_vel: Array | None = None,
    prev_cmd: Array | None = None,
    r_int_error: Array | None = None,
    ctrl_freq: int = 500,
    *,
    kR: Array,
    kw: Array,
    ki_m: Array,
    kd_omega: Array,
    int_err_max: Array,
    torque_pwm_max: Array,
    thrust_max: float,
    pwm_min: float,
    pwm_max: float,
    L: float,
    thrust2torque: float,
    mixing_matrix: Array,
) -> tuple[Array, Array, Array]:
    """Compute the body rate to desired force-torque part of the Mellinger controller.

    The firmware Mellinger controller has no dedicated body rate mode. A body rate setpoint enters
    the angular velocity error and its derivative, while the attitude terms level the drone at its
    current yaw. This function reproduces this behavior with the gains of the attitude controller.
    Set ``kR`` and ``ki_m`` to zero to track body rates without the attitude terms.

    Note:
        We omit the axis flip in the firmware as it has only been introduced to make the controller
        compatible with the new frame of the Crazyflie 2.1.

    Args:
        quat: Drone orientation as xyzw quaternion with shape (..., 4).
        ang_vel: Drone angular velocity in the body frame in rad/s with shape (..., 3).
        cmd: Commanded body rates (wx, wy, wz) and total thrust [rad/s, rad/s, rad/s, N].
        prev_ang_vel: Angular velocity in rad/s from the previous call. If None, it is initialised
            to zero.
        prev_cmd: Command from the previous call with shape (..., 4). The firmware includes the
            derivative of the body rate setpoint in the derivative term. If None, the setpoint is
            assumed to be constant.
        r_int_error: Angular velocity integral error (..., 3) from the previous call. If None, it
            is initialised to zero.
        ctrl_freq: Control frequency in Hz
        kR: Proportional gain for the rotation error with shape (3,).
        kw: Proportional gain for the angular velocity error with shape (3,).
        ki_m: Integral gain for the rotation error with shape (3,).
        kd_omega: Derivative gain for the angular velocity error with shape (3,).
        int_err_max: Range of the integral error with shape (3,). i_range in the firmware.
        torque_pwm_max: Maximum torque in PWM.
        thrust_max: Maximum thrust in N.
        pwm_min: Minimum PWM value.
        pwm_max: Maximum PWM value.
        L: Distance from the center of the quadrotor to the center of the rotor in m.
        thrust2torque: Conversion factor (m).
        mixing_matrix: Mixing matrix for the motor forces with shape (4, 3).

    Returns:
        Desired force (1,), torques (3,) and i_error_m
    """
    xp = array_namespace(quat)
    # l. 215 ff Without a position or attitude setpoint, the firmware levels the drone at the
    # current yaw
    yaw = R.from_quat(quat).as_euler("xyz", degrees=False)[..., 2]
    rpy_des = xp.stack((xp.zeros_like(yaw), xp.zeros_like(yaw), yaw), axis=-1)
    ang_vel_des = cmd[..., :3]
    prev_ang_vel_des = ang_vel_des if prev_cmd is None else prev_cmd[..., :3]
    return _attitude2force_torque(
        quat,
        ang_vel,
        rpy_des,
        ang_vel_des,
        cmd[..., 3],
        prev_ang_vel,
        prev_ang_vel_des,
        r_int_error,
        ctrl_freq,
        kR=kR,
        kw=kw,
        ki_m=ki_m,
        kd_omega=kd_omega,
        int_err_max=int_err_max,
        torque_pwm_max=torque_pwm_max,
        thrust_max=thrust_max,
        pwm_min=pwm_min,
        pwm_max=pwm_max,
        L=L,
        thrust2torque=thrust2torque,
        mixing_matrix=mixing_matrix,
    )


def _attitude2force_torque(
    quat: Array,
    ang_vel: Array,
    rpy_des: Array,
    ang_vel_des: Array,
    force_des: Array,
    prev_ang_vel: Array | None,
    prev_ang_vel_des: Array,
    r_int_error: Array | None,
    ctrl_freq: int,
    *,
    kR: Array,
    kw: Array,
    ki_m: Array,
    kd_omega: Array,
    int_err_max: Array,
    torque_pwm_max: Array,
    thrust_max: float,
    pwm_min: float,
    pwm_max: float,
    L: float,
    thrust2torque: float,
    mixing_matrix: Array,
) -> tuple[Array, Array, Array]:
    """Attitude and body rate controller of the Mellinger controller.

    This function follows the structure of the firmware implementation. The firmware setpoint
    carries both an attitude and a body rate. The attitude and body rate controllers route their
    commands into the respective setpoint.

    Args:
        quat: Drone orientation as xyzw quaternion with shape (..., 4).
        ang_vel: Drone angular velocity in the body frame in rad/s with shape (..., 3).
        rpy_des: Desired attitude as roll, pitch, yaw in rad with shape (..., 3).
        ang_vel_des: Desired angular velocity in the body frame in rad/s with shape (..., 3).
        force_des: Desired total thrust in N with shape (...,).
        prev_ang_vel: Angular velocity from the previous call. If None, it is initialised to zero.
        prev_ang_vel_des: Desired angular velocity from the previous call with shape (..., 3).
        r_int_error: Rotation integral error (..., 3) from the previous call. If None, it is
            initialised to zero.
        ctrl_freq: Control frequency in Hz
        kR: Proportional gain for the rotation error with shape (3,).
        kw: Proportional gain for the angular velocity error with shape (3,).
        ki_m: Integral gain for the rotation error with shape (3,).
        kd_omega: Derivative gain for the angular velocity error with shape (3,).
        int_err_max: Range of the integral error with shape (3,). i_range in the firmware.
        torque_pwm_max: Maximum torque in PWM.
        thrust_max: Maximum thrust in N.
        pwm_min: Minimum PWM value.
        pwm_max: Maximum PWM value.
        L: Distance from the center of the quadrotor to the center of the rotor in m.
        thrust2torque: Conversion factor (m).
        mixing_matrix: Mixing matrix for the motor forces with shape (4, 3).

    Returns:
        Desired force (..., 1), torques (..., 3) and i_error_m
    """
    xp = array_namespace(quat)
    dt = 1 / ctrl_freq
    # l. 220 ff [eR]. We're using the "inefficient" code path from the firmware
    rot = R.from_quat(quat)
    rot_des = R.from_euler("xyz", rpy_des, degrees=False)
    # Equivalent to eRM = R_des.T @ R_act - R_act.T @ R_des
    # Firmware does not multiply by 0.5 here, but the original paper does. We replicate the firmware
    # exactly to avoid sim2real issues with the original controller parameters.
    R_delta = (rot_des.inv() * rot).as_matrix()
    eRM = R_delta - R_delta.mT
    # Vee operator (SO3 to R3)
    eR = xp.stack((eRM[..., 2, 1], eRM[..., 0, 2], eRM[..., 1, 0]), axis=-1)
    # l.248 ff [ew]
    # The firmware negates the pitch components of the gyro and the rate setpoint to convert them
    # to the legacy Crazyflie frame, matching the sign flip of eR.y. We omit both flips and keep all
    # terms in the standard body frame, so the setpoint enters without a sign change.
    ew = ang_vel_des - ang_vel

    # l.259 ff [err_d_rpy]
    prev_ang_vel = xp.zeros_like(ang_vel) if prev_ang_vel is None else prev_ang_vel
    ang_vel_d_err = ((ang_vel_des - prev_ang_vel_des) - (ang_vel - prev_ang_vel)) / dt
    # l.281: No err_d_yaw
    ang_vel_d_err = xpx.at(ang_vel_d_err)[..., 2].set(0)

    # l. 268 ff Integral Error
    r_int_error = xp.zeros_like(ang_vel) if r_int_error is None else r_int_error
    r_int_error = r_int_error - eR * dt
    r_int_error = xp.clip(r_int_error, -int_err_max, int_err_max)
    # l. 278 ff Moment:
    torque_pwm = -kR * eR + kw * ew + ki_m * r_int_error + kd_omega * ang_vel_d_err
    # l. 297 ff
    torque_pwm = xp.clip(torque_pwm, -torque_pwm_max, torque_pwm_max)
    torque_pwm = xp.where((force_des > 0)[..., None], torque_pwm, 0.0)
    force_des_pwm = force2pwm(force_des / 4, thrust_max, pwm_max)
    pwms = force_torque_pwms2pwms(force_des_pwm, torque_pwm, mixing_matrix)
    idle = xp.all(pwms == 0, axis=-1, keepdims=True)
    pwms = xp.where(idle, 0.0, xp.clip(pwms, pwm_min, pwm_max))

    # Info: The Mellinger controller in the firmware ends here. However, we enforce a standardized
    # interface in the simulation from states -> attitude -> force_torque. We therefore need this
    # function to convert from PWMs to forces and torques.
    # In the firmware, this is done implicitly with the motor mixing. We therefore do the motor
    # mixing here, calculate the resulting force and torque, and return them.
    # This process is then reversed in the next step, where we recover the desired motor forces from
    # the force and torque.
    motor_forces = pwm2force(pwms, thrust_max, pwm_max)
    # TODO: Long-term, the Mellinger controller should use the new power distribution which
    # calculates motor forces in Newtons. However, for now the firmware uses the legacy power
    # distribution, so we keep it here for compatibility. To have a single consistent interface for
    # controllers, we still want to return SI forces and torques. We thus need to convert the legacy
    # output to SI units.
    # l. 310 ff
    torque_des = (mixing_matrix @ motor_forces[..., None])[..., 0] * xp.stack([L, L, thrust2torque])
    force_des = xp.sum(motor_forces, axis=-1)[..., None]
    return force_des, torque_des, r_int_error


def force_torque_pwms2pwms(force_pwm: Array, torque_pwm: Array, mixing_matrix: Array) -> Array:
    """Convert desired collective thrust and torques to rotor speeds using legacy behavior."""
    xp = array_namespace(force_pwm)
    torque_pwm = xp.concatenate((torque_pwm[..., :2] / 2, torque_pwm[..., 2:]), axis=-1)
    return force_pwm[..., None] + (torque_pwm @ mixing_matrix)


def force_torque2rotor_vel(
    force: Array,
    torque: Array,
    *,
    thrust_min: float,
    thrust_max: float,
    L: float,
    rpm2thrust: Array,
    thrust2torque: float,
    mixing_matrix: Array,
) -> Array:
    """Convert desired collective thrust and torques to rotor speeds.

    The firmware calculates PWMs for each motor, compensates for the battery voltage, and then
    applies the modified PWMs to the motors. We assume perfect battery compensation here, skip the
    PWM interface except for clipping, and instead return desired motor velocities.

    Note:
        The equivalent function in the crazyflie firmware is power_distribution from
        power_distribution_quadrotor.c.

    Warning:
        This function assumes an X rotor configuration.

    Args:
        force: Desired thrust in SI units with shape (...,).
        torque: Desired torque in SI units with shape (..., 3).
        thrust_min: Minimum thrust in N.
        thrust_max: Maximum thrust in N.
        L: Distance from the center of the quadrotor to the center of the rotor in m.
        rpm2thrust: Force constants (N/RPM, N/RPM**2).
        thrust2torque: Conversion factor (m).
        mixing_matrix: Mixing matrix for the motor forces with shape (4, 3).

    Returns:
        The desired rotor velocities in RPMs with shape (..., 4).
    """
    xp = array_namespace(torque)
    assert torque.shape[-1] == 3, f"Torque must have shape (..., 3), but has {torque.shape}"
    assert force.shape[-1] == 1, f"Force must have shape (..., 1), but has {force.shape}"
    torque_forces = (torque * xp.asarray([1 / L, 1 / L, 1 / thrust2torque])) @ mixing_matrix
    motor_forces = (torque_forces + force) / 4
    # Clip motor forces on the thrust instead of PWM level.
    idle = xp.all(force == 0, axis=-1, keepdims=True)
    motor_forces = xp.where(idle, 0.0, xp.clip(motor_forces, thrust_min, thrust_max))
    # Assume perfect battery compensation and calculate the desired motor speeds directly
    return motor_force2rotor_vel(motor_forces, rpm2thrust)


class MellingerStateData(PyTreeNode):
    cmd: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 16)
    """Full state control command for the drone.

    A command consists of [x, y, z, vx, vy, vz, ax, ay, az, qx, qy, qz, qw, wx, wy, wz].
    """
    staged_cmd: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 16)
    """Staging buffer to store the most recent command until the next controller tick."""
    steps: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, 1)
    """Last simulation steps that the state control command was applied."""
    freq: int = field(pytree_node=False)
    """Frequency of the state control command."""
    pos_err_i: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Integral errors of the state control command."""
    # Parameters for the state controller
    params: dict[str, Array]

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, freq: int, drone: str, device: Device
    ) -> MellingerStateData:
        """Create a default set of state data for the simulation."""
        zeros_3d = jnp.zeros((n_worlds, n_drones, 3), device=device)
        cmd = jnp.zeros((n_worlds, n_drones, 16), device=device).at[..., 12].set(1.0)
        steps = -jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device)
        params = load_fn_params(state2attitude, drone, xp=jnp, device=device)
        return MellingerStateData(
            cmd=cmd,
            staged_cmd=cmd.copy(),
            steps=steps,
            freq=freq,
            pos_err_i=zeros_3d.copy(),
            params=params,
        )


class MellingerAttitudeData(PyTreeNode):
    cmd: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 4)
    """Attitude control setpoint consisting of [roll, pitch, yaw, collective thrust]."""
    staged_cmd: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 4)
    """Staging buffer to store the most recent command until the next controller tick."""
    ang_vel_des: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Body rate setpoint [wx, wy, wz] of the attitude controller."""
    staged_ang_vel_des: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Staging buffer to store the most recent body rate setpoint until the next controller tick."""
    steps: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, 1)
    """Last simulation steps that the attitude control command was applied."""
    freq: int = field(pytree_node=False)
    """Frequency of the attitude control command."""
    r_int_error: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Integral errors of the attitude control command."""
    prev_ang_vel: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Previous angular velocity of the drone."""
    # Parameters for the attitude controller
    params: dict[str, Array]

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, freq: int, drone: str, device: Device
    ) -> MellingerAttitudeData:
        """Create a default set of attitude data for the simulation."""
        zeros_3d = jnp.zeros((n_worlds, n_drones, 3), device=device)
        zeros_4d = jnp.zeros((n_worlds, n_drones, 4), device=device)
        steps = -jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device)
        params = load_fn_params(attitude2force_torque, drone, xp=jnp, device=device)
        return MellingerAttitudeData(
            cmd=zeros_4d.copy(),
            staged_cmd=zeros_4d.copy(),
            ang_vel_des=zeros_3d.copy(),
            staged_ang_vel_des=zeros_3d.copy(),
            steps=steps,
            freq=freq,
            r_int_error=zeros_3d.copy(),
            prev_ang_vel=zeros_3d.copy(),
            params=params,
        )


class MellingerBodyRateData(PyTreeNode):
    cmd: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 4)
    """Body rate control command for the drone.

    A command consists of [wx, wy, wz, collective thrust].
    """
    staged_cmd: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 4)
    """Staging buffer to store the most recent command until the next controller tick."""
    steps: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, 1)
    """Last simulation steps that the body rate control command was applied."""
    freq: int = field(pytree_node=False)
    """Frequency of the body rate control command."""
    r_int_error: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Integral errors of the body rate control command."""
    prev_ang_vel: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Previous angular velocity of the drone."""
    # Parameters for the body rate controller
    params: dict[str, Array]

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, freq: int, drone: str, device: Device
    ) -> MellingerBodyRateData:
        """Create a default set of body rate data for the simulation."""
        zeros_3d = jnp.zeros((n_worlds, n_drones, 3), device=device)
        zeros_4d = jnp.zeros((n_worlds, n_drones, 4), device=device)
        steps = -jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device)
        params = load_fn_params(body_rate2force_torque, drone, xp=jnp, device=device)
        return MellingerBodyRateData(
            cmd=zeros_4d.copy(),
            staged_cmd=zeros_4d.copy(),
            steps=steps,
            freq=freq,
            r_int_error=zeros_3d.copy(),
            prev_ang_vel=zeros_3d.copy(),
            params=params,
        )


class MellingerForceTorqueData(PyTreeNode):
    cmd: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 4)
    """Force-torque command for the drone.

    A command consists of [fz, tx, ty, tz].
    """
    staged_cmd: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 4)
    """Staging buffer to store the most recent command until the next controller tick."""
    steps: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, 1)
    """Last simulation steps that the force and torque control command was applied."""
    freq: int = field(pytree_node=False)
    """Frequency of the force and torque control command."""
    # Parameters for the force and torque controller
    params: dict[str, Array]

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, freq: int, drone: str, device: Device
    ) -> MellingerForceTorqueData:
        zeros_4d = jnp.zeros((n_worlds, n_drones, 4), device=device)
        steps = -jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device)
        params = load_fn_params(force_torque2rotor_vel, drone, xp=jnp, device=device)
        return MellingerForceTorqueData(
            cmd=zeros_4d.copy(), staged_cmd=zeros_4d.copy(), steps=steps, freq=freq, params=params
        )


def control_state2attitude(data: SimData) -> SimData:
    """Compute the updated controls for the state controller."""
    states = data.states
    state_ctrl = data.controls.state
    assert state_ctrl is not None, "Using state controller without initialized data"
    mask = controllable(data.core.steps, data.core.freq, state_ctrl.steps, state_ctrl.freq)
    state_ctrl = leaf_replace(state_ctrl, mask, cmd=state_ctrl.staged_cmd)
    rpyt, pos_err_i = state2attitude(
        states.pos,
        states.quat,
        states.vel,
        state_ctrl.cmd,
        pos_err_i=state_ctrl.pos_err_i,
        ctrl_freq=state_ctrl.freq,
        **state_ctrl.params,
    )
    state_ctrl = leaf_replace(state_ctrl, mask, steps=data.core.steps, pos_err_i=pos_err_i)
    attitude_ctrl = data.controls.attitude
    assert attitude_ctrl is not None, "Using attitude controller without initialized data"
    attitude_ctrl = leaf_replace(
        attitude_ctrl, mask, staged_cmd=rpyt, staged_ang_vel_des=state_ctrl.cmd[..., 13:16]
    )
    return data.replace(controls=data.controls.replace(state=state_ctrl, attitude=attitude_ctrl))


def control_attitude2force_torque(data: SimData) -> SimData:
    """Compute the updated controls for the attitude controller."""
    states = data.states
    attitude_ctrl = data.controls.attitude
    assert attitude_ctrl is not None, "Using attitude controller without initialized data"
    mask = controllable(data.core.steps, data.core.freq, attitude_ctrl.steps, attitude_ctrl.freq)
    prev_ang_vel_des = attitude_ctrl.ang_vel_des
    attitude_ctrl = leaf_replace(
        attitude_ctrl,
        mask,
        cmd=attitude_ctrl.staged_cmd,
        ang_vel_des=attitude_ctrl.staged_ang_vel_des,
    )
    force, torque, r_int_error = _attitude2force_torque(
        states.quat,
        states.ang_vel,
        attitude_ctrl.cmd[..., :3],
        attitude_ctrl.ang_vel_des,
        attitude_ctrl.cmd[..., 3],
        attitude_ctrl.prev_ang_vel,
        prev_ang_vel_des,
        attitude_ctrl.r_int_error,
        attitude_ctrl.freq,
        **attitude_ctrl.params,
    )
    attitude_ctrl = leaf_replace(
        attitude_ctrl,
        mask,
        r_int_error=r_int_error,
        prev_ang_vel=states.ang_vel,
        steps=data.core.steps,
    )
    ft_ctrl = data.controls.force_torque
    assert ft_ctrl is not None, "Using force torque controller without initialized data"
    ft_ctrl = leaf_replace(ft_ctrl, mask, staged_cmd=jnp.concat([force, torque], axis=-1))
    return data.replace(
        states=states, controls=data.controls.replace(attitude=attitude_ctrl, force_torque=ft_ctrl)
    )


def control_body_rate2force_torque(data: SimData) -> SimData:
    """Compute the updated controls for the body rate controller."""
    states = data.states
    body_rate_ctrl = data.controls.body_rate
    assert body_rate_ctrl is not None, "Using body rate controller without initialized data"
    mask = controllable(data.core.steps, data.core.freq, body_rate_ctrl.steps, body_rate_ctrl.freq)
    prev_cmd = body_rate_ctrl.cmd
    body_rate_ctrl = leaf_replace(body_rate_ctrl, mask, cmd=body_rate_ctrl.staged_cmd)
    force, torque, r_int_error = body_rate2force_torque(
        states.quat,
        states.ang_vel,
        body_rate_ctrl.cmd,
        prev_ang_vel=body_rate_ctrl.prev_ang_vel,
        prev_cmd=prev_cmd,
        r_int_error=body_rate_ctrl.r_int_error,
        ctrl_freq=body_rate_ctrl.freq,
        **body_rate_ctrl.params,
    )
    body_rate_ctrl = leaf_replace(
        body_rate_ctrl,
        mask,
        r_int_error=r_int_error,
        prev_ang_vel=states.ang_vel,
        steps=data.core.steps,
    )
    ft_ctrl = data.controls.force_torque
    assert ft_ctrl is not None, "Using force torque controller without initialized data"
    ft_ctrl = leaf_replace(ft_ctrl, mask, staged_cmd=jnp.concat([force, torque], axis=-1))
    return data.replace(
        controls=data.controls.replace(body_rate=body_rate_ctrl, force_torque=ft_ctrl)
    )


def control_commit_attitude(data: SimData) -> SimData:
    """Commit the staged attitude command to the controller setpoint."""
    attitude_ctrl = data.controls.attitude
    assert attitude_ctrl is not None, "Using attitude controller without initialized data"
    mask = controllable(data.core.steps, data.core.freq, attitude_ctrl.steps, attitude_ctrl.freq)
    attitude_ctrl = leaf_replace(attitude_ctrl, mask, cmd=attitude_ctrl.staged_cmd)
    return data.replace(controls=data.controls.replace(attitude=attitude_ctrl))


def control_force_torque2rotor_vel(data: SimData) -> SimData:
    """Compute the updated controls for the thrust controller."""
    ft_ctrl = data.controls.force_torque
    assert ft_ctrl is not None, "Using force torque controller without initialized data"
    mask = controllable(data.core.steps, data.core.freq, ft_ctrl.steps, ft_ctrl.freq)
    ft_ctrl = leaf_replace(ft_ctrl, mask, cmd=ft_ctrl.staged_cmd)
    rotor_vel = force_torque2rotor_vel(
        ft_ctrl.cmd[..., [0]], ft_ctrl.cmd[..., 1:], **ft_ctrl.params
    )
    ft_ctrl = leaf_replace(ft_ctrl, mask, steps=data.core.steps)
    return data.replace(controls=data.controls.replace(rotor_vel=rotor_vel, force_torque=ft_ctrl))
