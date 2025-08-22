from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field
from jax import Array, Device

from crazyflow.control import Control
from crazyflow.control.mellinger import (
    MellingerAttitudeData,
    MellingerForceTorqueData,
    MellingerStateData,
)


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
    rotor_vel: Array  # (N, M, 4)  # Motor forces along body frame z axis
    """Motor forces along body frame z axis."""

    @staticmethod
    def create(n_worlds: int, n_drones: int, device: Device) -> SimState:
        """Create a default set of states for the simulation."""
        zeros_3d = jnp.zeros((n_worlds, n_drones, 3), device=device)
        q_identity = jnp.zeros((n_worlds, n_drones, 4), device=device)
        q_identity = q_identity.at[..., -1].set(1.0)
        rotor_vel = jnp.zeros((n_worlds, n_drones, 4), device=device)
        return SimState(
            pos=zeros_3d,
            quat=q_identity,
            vel=zeros_3d,
            ang_vel=zeros_3d,
            force=zeros_3d,
            torque=zeros_3d,
            rotor_vel=rotor_vel,
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


@typing.runtime_checkable
class ControlData(typing.Protocol):
    staged_cmd: Array  # (N, M, X)
    """Staged control command for the drone.

    The most recent control input gets staged here until the next controller tick and is then
    committed to the cmd field.
    """
    cmd: Array  # (N, M, X)
    """Control command for the drone."""
    staged_cmd: Array  # (N, M, X)
    """Staged control command for the drone."""
    steps: Array  # (N, 1)
    """Last simulation steps that the state control command was applied."""
    freq: int
    """Frequency of the state control command."""
    # Parameters for the controller
    params: tuple[typing.Any, ...]


@dataclass
class SimControls:
    state: ControlData | None
    """State control data."""
    attitude: ControlData | None
    """Attitude control data."""
    force_torque: ControlData | None
    """Force and torque control data."""
    rotor_vel: Array  # (N, M, 4)
    """Desired motor speed."""

    @staticmethod
    def create(
        n_worlds: int,
        n_drones: int,
        control: Control,
        drone_model: str,
        state_freq: int | None,
        attitude_freq: int | None,
        force_torque_freq: int | None,
        device: Device,
    ) -> SimControls:
        """Create a default set of controls for the simulation."""
        rotor_vel = jnp.zeros((n_worlds, n_drones, 4), device=device)
        match control:
            case Control.state:
                state = MellingerStateData.create(
                    n_worlds, n_drones, state_freq, drone_model, device
                )
                attitude = MellingerAttitudeData.create(
                    n_worlds, n_drones, attitude_freq, drone_model, device
                )
                force_torque = MellingerForceTorqueData.create(
                    n_worlds, n_drones, force_torque_freq, drone_model, device
                )
                return SimControls(
                    state=state, attitude=attitude, force_torque=force_torque, rotor_vel=rotor_vel
                )
            case Control.attitude:
                attitude = attitude = MellingerAttitudeData.create(
                    n_worlds, n_drones, attitude_freq, drone_model, device
                )
                force_torque = MellingerForceTorqueData.create(
                    n_worlds, n_drones, force_torque_freq, drone_model, device
                )
                return SimControls(
                    state=None, attitude=attitude, force_torque=force_torque, rotor_vel=rotor_vel
                )
            case Control.force_torque:
                force_torque = MellingerForceTorqueData.create(
                    n_worlds, n_drones, force_torque_freq, drone_model, device
                )
                return SimControls(
                    state=None, attitude=None, force_torque=force_torque, rotor_vel=rotor_vel
                )
            case _:
                raise ValueError(f"Control mode {control} not implemented")


@dataclass
class SimParams:
    mass: Array  # (N, M, 1)
    """Mass of the drone."""
    J: Array  # (N, M, 3, 3)
    """Inertia matrix of the drone."""
    J_inv: Array  # (N, M, 3, 3)
    """Inverse of the inertia matrix of the drone."""

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, mass: float, J: Array, J_inv: Array, device: Device
    ) -> SimParams:
        """Create a default set of parameters for the simulation."""
        mass = jnp.ones((n_worlds, n_drones, 1), device=device) * mass
        j, j_inv = jnp.array(J, device=device), jnp.array(J_inv, device=device)
        J = jnp.tile(j[None, None, :, :], (n_worlds, n_drones, 1, 1))
        J_inv = jnp.tile(j_inv[None, None, :, :], (n_worlds, n_drones, 1, 1))
        return SimParams(mass=mass, J=J, J_inv=J_inv)


@dataclass
class SimConstants:
    L: Array  # ()
    """Arm length of the drone."""
    MIXING_MATRIX: Array  # (3, 4)
    """Mixing matrix of the drone."""
    KF: Array  # ()
    """Force constant of the drone."""
    KM: Array  # ()
    """Torque constant of the drone."""

    @staticmethod
    def create(
        L: float, mixing_matrix: Array, KF: float, KM: float, device: Device
    ) -> SimConstants:
        """Create a default set of constants for the simulation."""
        L = jnp.array(L, device=device, dtype=jnp.float32)
        mixing_matrix = jnp.array(mixing_matrix, device=device, dtype=jnp.float32)
        KF = jnp.array(KF, device=device, dtype=jnp.float32)
        KM = jnp.array(KM, device=device, dtype=jnp.float32)
        return SimConstants(L=L, MIXING_MATRIX=mixing_matrix, KF=KF, KM=KM)


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
    """Drone controller data."""
    params: SimParams
    """Drone parameters."""
    constants: SimConstants
    """Drone constants."""
    core: SimCore
    """Core parameters of the simulation."""
