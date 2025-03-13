from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field
from jax import Array, Device
from lsy_models.utils.constants import Constants

if TYPE_CHECKING:
    from mujoco.mjx import Data, Model


@dataclass
class SimState:
    pos: Array  # (N, M, 3)
    """Position of the drone's center of mass."""
    quat: Array  # (N, M, 4)
    """Quaternion of the drone's orientation."""
    vel: Array  # (N, M, 3)
    """Velocity of the drone's center of mass in the world frame."""
    ang_vel: Array  # (N, M, 3)
    """Angular velocity of the drone's center of mass in the world frame."""
    motor_forces: Array  # (N, M, 4)  # Motor forces along body frame z axis
    """Motor forces along body frame z axis."""
    force: Array  # (N, M, 3)  # CoM force
    """Force applied to the drone's center of mass in the world frame."""
    torque: Array  # (N, M, 3)  # CoM torque
    """Torque applied to the drone's center of mass in the world frame."""

    @staticmethod
    def create(n_worlds: int, n_drones: int, device: Device) -> SimState:
        """Create a default set of states for the simulation."""
        pos = jnp.zeros((n_worlds, n_drones, 3), device=device) + jnp.array([0, 0, 0.05])
        quat = jnp.zeros((n_worlds, n_drones, 4), device=device)
        quat = quat.at[..., -1].set(1.0)
        vel = jnp.zeros((n_worlds, n_drones, 3), device=device)
        ang_vel = jnp.zeros((n_worlds, n_drones, 3), device=device)
        force = jnp.zeros((n_worlds, n_drones, 3), device=device)
        torque = jnp.zeros((n_worlds, n_drones, 3), device=device)
        # TODO remove 0.08 factor and rather make floor solid!
        motor_forces = jnp.ones((n_worlds, n_drones, 4), device=device) * 0.08
        return SimState(
            pos=pos,
            quat=quat,
            vel=vel,
            ang_vel=ang_vel,
            force=force,
            torque=torque,
            motor_forces=motor_forces,
        )


@dataclass
class SimStateDeriv:
    dpos: Array  # (N, M, 3)
    """Derivative of the position of the drone's center of mass."""
    drot: Array  # (N, M, 3)
    """Derivative of the quaternion of the drone's orientation as angular velocity."""
    dvel: Array  # (N, M, 3)
    """Derivative of the velocity of the drone's center of mass."""
    dang_vel: Array  # (N, M, 3)
    """Derivative of the angular velocity of the drone's center of mass."""
    dmotor_forces: Array  # (N, M, 4)
    """Derivative of the motor forces along body frame z axis."""

    @staticmethod
    def create(n_worlds: int, n_drones: int, device: Device) -> SimStateDeriv:
        """Create a default set of state derivatives for the simulation."""
        dpos = jnp.zeros((n_worlds, n_drones, 3), device=device)
        drot = jnp.zeros((n_worlds, n_drones, 3), device=device)
        dvel = jnp.zeros((n_worlds, n_drones, 3), device=device)
        dang_vel = jnp.zeros((n_worlds, n_drones, 3), device=device)
        dmotor_forces = jnp.zeros((n_worlds, n_drones, 4), device=device)
        return SimStateDeriv(
            dpos=dpos, drot=drot, dvel=dvel, dang_vel=dang_vel, dmotor_forces=dmotor_forces
        )


@dataclass
class SimControls:
    state: Array  # (N, M, 13)
    """Full state control command for the drone.

    A command consists of [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate].
    We currently do not use the acceleration and angle rate components. This is subject to change.
    """
    state_steps: Array  # (N, 1)
    """Last simulation steps that the state control command was applied."""
    state_freq: int = field(pytree_node=False)
    """Frequency of the state control command."""
    attitude: Array  # (N, M, 4)
    """Attitude control command for the drone.

    A command consists of [collective thrust, roll, pitch, yaw].
    """
    staged_attitude: Array  # (N, M, 4)
    """Staged attitude control command for the drone that has not been applied yet.

    See `Sim.attitude_control` for more details.
    """
    attitude_steps: Array  # (N, 1)
    """Last simulation steps that the attitude control command was applied."""
    attitude_freq: int = field(pytree_node=False)
    """Frequency of the attitude control command."""
    thrust: Array  # (N, M, 4)
    """Thrust control command for the drone.

    A command consists of [thrust1, thrust2, thrust3, thrust4] for each motor.
    """
    thrust_steps: Array  # (N, 1)
    """Last simulation steps that the thrust control command was applied."""
    thrust_freq: int = field(pytree_node=False)
    """Frequency of the thrust control command."""
    rpms: Array  # (N, M, 4)
    """RPMs for each motor."""
    rpy_err_i: Array  # (N, M, 3)
    """Integral of the rpy error."""
    pos_err_i: Array  # (N, M, 3)
    """Integral of the position error."""
    prev_ang_vel: Array  # (N, M, 3)
    """Aangular velocity from the last controller step.

    Required to compute the integral term in the attitude controller.
    """

    @staticmethod
    def create(
        n_worlds: int,
        n_drones: int,
        state_freq: int = 100,
        attitude_freq: int = 500,
        thrust_freq: int = 500,
        device: Device | str = "cpu",
    ) -> SimControls:
        """Create a default set of controls for the simulation."""
        device = jax.devices(device)[0] if isinstance(device, str) else device
        return SimControls(
            state=jnp.zeros((n_worlds, n_drones, 13), device=device),
            state_steps=-jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device),
            state_freq=state_freq,
            attitude=jnp.zeros((n_worlds, n_drones, 4), device=device),
            staged_attitude=jnp.zeros((n_worlds, n_drones, 4), device=device),
            attitude_steps=-jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device),
            attitude_freq=attitude_freq,
            thrust=jnp.ones((n_worlds, n_drones, 4), device=device),
            thrust_steps=-jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device),
            thrust_freq=thrust_freq,
            rpms=jnp.zeros((n_worlds, n_drones, 4), device=device),
            rpy_err_i=jnp.zeros((n_worlds, n_drones, 3), device=device),
            pos_err_i=jnp.zeros((n_worlds, n_drones, 3), device=device),
            prev_ang_vel=jnp.zeros((n_worlds, n_drones, 3), device=device),
        )


@dataclass
class SimParams:
    # Variable params (for domain randomization) => (N, M, shape)
    MASS: Array  # (N, M, 1)
    """Mass of the drone."""
    J: Array  # (N, M, 3, 3)
    """Inertia matrix of the drone."""
    J_INV: Array  # (N, M, 3, 3)
    """Inverse of the inertia matrix of the drone."""
    L: Array  # (N, M, 1)
    """Arm length of the drone, aka distance of the motors from the center of mass."""

    # TODO maybe params, maybe constants?
    KF: float = field(pytree_node=False)  # (N, M, 1)
    """RPM squared to Force factor."""
    KM: float = field(pytree_node=False)  # (N, M, 1)
    """RPM squared to Torque factor."""
    THRUST_MIN: float = field(pytree_node=False)  # (N, M, 1)
    """Min thrust per motor."""
    THRUST_MAX: float = field(pytree_node=False)  # (N, M, 1)
    """Max thrust per motor."""
    THRUST_TAU: float = field(pytree_node=False)  # (N, M, 1)
    # TODO maybe N, M, 4, for each of the motors individually
    """Time constant for the thrust dynamics."""

    # Constants
    GRAVITY_VEC: Array = field(pytree_node=False)
    # MIX_MATRIX: Array = field(pytree_node=False) # TODO not needed? => remove
    SIGN_MATRIX: Array = field(pytree_node=False)
    PWM_MIN: float = field(pytree_node=False)
    PWM_MAX: float = field(pytree_node=False)

    # System Identification (SI) parameters
    SI_ROLL: Array = field(pytree_node=False)
    SI_PITCH: Array = field(pytree_node=False)
    SI_YAW: Array = field(pytree_node=False)
    SI_PARAMS: Array = field(pytree_node=False)
    SI_ACC: Array = field(pytree_node=False)

    # System Identification parameters for the double integrator (DI) model
    DI_ROLL: Array = field(pytree_node=False)
    DI_PITCH: Array = field(pytree_node=False)
    DI_YAW: Array = field(pytree_node=False)
    DI_PARAMS: Array = field(pytree_node=False)
    DI_ACC: Array = field(pytree_node=False)

    @staticmethod
    def create(
        n_worlds: int,
        n_drones: int,
        mass: float,
        J: Array,
        device: Device,
        L: float | None = None,
        KF: float | None = None,
        KM: float | None = None,
        THRUST_MIN: float | None = None,
        THRUST_MAX: float | None = None,
        THRUST_TAU: float | None = None,
    ) -> SimParams:
        """Create a default set of parameters for the simulation."""
        MASS = jnp.ones((n_worlds, n_drones, 1), device=device) * mass
        j = jnp.array(J, device=device)
        j_inv = jnp.linalg.inv(j)
        J = jnp.tile(j[None, None, :, :], (n_worlds, n_drones, 1, 1))
        J_INV = jnp.tile(j_inv[None, None, :, :], (n_worlds, n_drones, 1, 1))

        constants = Constants.from_config("cf2x_L250")  # TODO make dependent on actual drone config

        if L is None:
            L = constants.L
        if KF is None:
            KF = constants.KF
        if KM is None:
            KM = constants.KM
        if THRUST_MIN is None:
            THRUST_MIN = constants.THRUST_MIN
        if THRUST_MAX is None:
            THRUST_MAX = constants.THRUST_MAX
        if THRUST_TAU is None:
            THRUST_TAU = constants.THRUST_TAU

        return SimParams(
            MASS=MASS,
            J=J,
            J_INV=J_INV,
            L=L,
            KF=KF,
            KM=KM,
            THRUST_MIN=THRUST_MIN,
            THRUST_MAX=THRUST_MAX,
            THRUST_TAU=THRUST_TAU * 2,  # TODO remove
            GRAVITY_VEC=constants.GRAVITY_VEC,
            SIGN_MATRIX=constants.SIGN_MATRIX,
            PWM_MIN=constants.PWM_MIN,
            PWM_MAX=constants.PWM_MAX,
            SI_ROLL=constants.SI_ROLL,
            SI_PITCH=constants.SI_PITCH,
            SI_YAW=constants.SI_YAW,
            SI_PARAMS=constants.SI_PARAMS,
            SI_ACC=constants.SI_ACC,
            DI_ROLL=constants.DI_ROLL,
            DI_PITCH=constants.DI_PITCH,
            DI_YAW=constants.DI_YAW,
            DI_PARAMS=constants.DI_PARAMS,
            DI_ACC=constants.DI_ACC,
        )


@dataclass
class SimCore:
    freq: int = field(pytree_node=False)
    """Frequency of the simulation."""
    steps: Array  # (N, 1)
    """Simulation steps taken since the last reset."""
    n_worlds: int = field(pytree_node=False)
    """Number of worlds in the simulation."""
    n_drones: int = field(pytree_node=False)
    """Number of drones in the simulation."""
    drone_ids: Array  # (1, M)
    """MuJoCo IDs of the drones in the simulation."""
    rng_key: Array  # (N, 1)
    """Random number generator key for the simulation."""

    @staticmethod
    def create(
        freq: int,
        n_worlds: int,
        n_drones: int,
        drone_ids: Array,
        rng_key: int | Array,
        device: Device,
    ) -> SimCore:
        """Create a default set of core simulation parameters."""
        steps = jnp.zeros((n_worlds, 1), dtype=jnp.int32, device=device)
        if isinstance(rng_key, int):  # Only convert to an PRNG key if its not already one
            rng_key = jax.random.key(rng_key)
        rng_key = jax.device_put(rng_key, device)
        return SimCore(
            freq=freq,
            steps=steps,
            n_worlds=n_worlds,
            n_drones=n_drones,
            drone_ids=jnp.array(drone_ids, dtype=jnp.int32, device=device),
            rng_key=rng_key,
        )


@dataclass
class SimData:
    states: SimState
    """State of the simulation."""
    states_deriv: SimStateDeriv
    """Derivative of the state of the simulation."""
    controls: SimControls
    """Drone control values."""
    params: SimParams
    """Drone parameters."""
    core: SimCore
    """Core parameters of the simulation."""
    mjx_data: Data
    """MuJoCo data structure."""
    mjx_model: Model | None
    """MuJoCo model structure.

    Can be set to None for performance optimizations. See `Sim.build_step` for more details.
    """
