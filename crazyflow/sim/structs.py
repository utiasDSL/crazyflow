from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field
from jax import Array, Device

from crazyflow.control import Control
from crazyflow.control.mellinger import MellingerStateData


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
    force: Array  # (N, M, 3)  # CoM force
    """Force applied to the drone's center of mass in the world frame."""
    torque: Array  # (N, M, 3)  # CoM torque
    """Torque applied to the drone's center of mass in the world frame."""
    motor_forces: Array  # (N, M, 4)  # Motor forces along body frame z axis
    """Motor forces along body frame z axis."""
    motor_torques: Array  # (N, M, 4)  # Motor torques around the body frame z axis
    """Motor torques around the body frame z axis."""

    @staticmethod
    def create(n_worlds: int, n_drones: int, device: Device) -> SimState:
        """Create a default set of states for the simulation."""
        pos = jnp.zeros((n_worlds, n_drones, 3), device=device)
        quat = jnp.zeros((n_worlds, n_drones, 4), device=device)
        quat = quat.at[..., -1].set(1.0)
        vel = jnp.zeros((n_worlds, n_drones, 3), device=device)
        ang_vel = jnp.zeros((n_worlds, n_drones, 3), device=device)
        force = jnp.zeros((n_worlds, n_drones, 3), device=device)
        torque = jnp.zeros((n_worlds, n_drones, 3), device=device)
        motor_forces = jnp.zeros((n_worlds, n_drones, 4), device=device)
        motor_torques = jnp.zeros((n_worlds, n_drones, 4), device=device)
        return SimState(
            pos=pos,
            quat=quat,
            force=force,
            torque=torque,
            motor_forces=motor_forces,
            motor_torques=motor_torques,
            vel=vel,
            ang_vel=ang_vel,
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

    @staticmethod
    def create(n_worlds: int, n_drones: int, device: Device) -> SimStateDeriv:
        """Create a default set of state derivatives for the simulation."""
        dpos = jnp.zeros((n_worlds, n_drones, 3), device=device)
        drot = jnp.zeros((n_worlds, n_drones, 3), device=device)
        dvel = jnp.zeros((n_worlds, n_drones, 3), device=device)
        dang_vel = jnp.zeros((n_worlds, n_drones, 3), device=device)
        return SimStateDeriv(dpos=dpos, drot=drot, dvel=dvel, dang_vel=dang_vel)


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
    last_rpy: Array  # (N, M, 3)
    """Last rpy for 'xyz' euler angles.

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
            thrust=jnp.zeros((n_worlds, n_drones, 4), device=device),
            thrust_steps=-jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device),
            thrust_freq=thrust_freq,
            rpms=jnp.zeros((n_worlds, n_drones, 4), device=device),
            rpy_err_i=jnp.zeros((n_worlds, n_drones, 3), device=device),
            pos_err_i=jnp.zeros((n_worlds, n_drones, 3), device=device),
            last_rpy=jnp.zeros((n_worlds, n_drones, 3), device=device),
        )


class ControlData(typing.Protocol):
    cmd: Array  # (N, M, X)
    """Control command for the drone."""
    steps: Array  # (N, 1)
    """Last simulation steps that the state control command was applied."""
    freq: int
    """Frequency of the state control command."""
    # Parameters for the controller
    params: typing.Any


@dataclass
class SimControlsNew:
    state: ControlData | None = None
    """State control data."""
    attitude: ControlData | None = None
    """Attitude control data."""
    thrust: ControlData | None = None
    """Thrust control data."""

    @staticmethod
    def create(
        n_worlds: int,
        n_drones: int,
        control: Control,
        state_freq: int | None,
        attitude_freq: int | None,
        thrust_freq: int | None,
        device: Device,
    ) -> SimControlsNew:
        """Create a default set of controls for the simulation."""
        match control:
            case Control.state:
                state = MellingerStateData.create(n_worlds, n_drones, state_freq, "", device)
                attitude = None  # MellingerAttitudeData.create(n_worlds, n_drones, device)
                thrust = None  # MellingerThrustData.create(n_worlds, n_drones, device)
                return SimControlsNew(state=state, attitude=attitude, thrust=thrust)
            case Control.attitude:
                attitude = None  # MellingerAttitudeData.create(n_worlds, n_drones, device)
                thrust = None  # MellingerThrustData.create(n_worlds, n_drones, device)
                return SimControlsNew(attitude=attitude, thrust=thrust)
            case Control.thrust:
                thrust = None  # MellingerThrustData.create(n_worlds, n_drones, device)
                return SimControlsNew(thrust=thrust)
            case _:
                raise ValueError(f"Control mode {control} not implemented")


@dataclass
class SimParams:
    mass: Array  # (N, M, 1)
    """Mass of the drone."""
    J: Array  # (N, M, 3, 3)
    """Inertia matrix of the drone."""
    J_INV: Array  # (N, M, 3, 3)
    """Inverse of the inertia matrix of the drone."""

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, mass: float, J: Array, J_INV: Array, device: Device
    ) -> SimParams:
        """Create a default set of parameters for the simulation."""
        mass = jnp.ones((n_worlds, n_drones, 1), device=device) * mass
        j, j_inv = jnp.array(J, device=device), jnp.array(J_INV, device=device)
        J = jnp.tile(j[None, None, :, :], (n_worlds, n_drones, 1, 1))
        J_INV = jnp.tile(j_inv[None, None, :, :], (n_worlds, n_drones, 1, 1))
        return SimParams(mass=mass, J=J, J_INV=J_INV)


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
    mjx_synced: Array  # (1,)
    """Whether the simulation data is synchronized with the MuJoCo model."""

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
            mjx_synced=jnp.array(False, dtype=jnp.bool_, device=device),
        )


@dataclass
class SimData:
    states: SimState
    """State of the simulation."""
    states_deriv: SimStateDeriv
    """Derivative of the state of the simulation."""
    controls: SimControls
    """Drone control values."""
    new_controls: SimControlsNew
    """New style control data TODO improve this"""
    params: SimParams
    """Drone parameters."""
    core: SimCore
    """Core parameters of the simulation."""
