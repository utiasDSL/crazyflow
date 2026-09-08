from __future__ import annotations

import typing
from typing import Any

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
from crazyflow.dynamics import Dynamics
from crazyflow.dynamics.first_principles import Params as FirstPrinciplesParams
from crazyflow.dynamics.so_rpy import Params as SoRpyParams
from crazyflow.dynamics.so_rpy_rotor import Params as SoRpyRotorParams
from crazyflow.dynamics.so_rpy_rotor_drag import Params as SoRpyRotorDragParams
from crazyflow.utils import CORE_NDIM_KEY


@dataclass
class SimState:
    pos: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Position of the drone's center of mass."""
    quat: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 4)
    """Quaternion of the drone's orientation."""
    vel: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Velocity of the drone's center of mass in the world frame."""
    ang_vel: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Angular velocity of the drone in the body frame."""
    force: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3) CoM force
    """Force applied to the drone's center of mass in the world frame."""
    torque: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3) CoM torque
    """Torque applied to the drone's center of mass in the world frame."""
    rotor_vel: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 4) in RPM
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
    vel: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Derivative of the position of the drone's center of mass."""
    ang_vel: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Derivative of the quaternion of the drone's orientation as angular velocity."""
    acc: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Derivative of the velocity of the drone's center of mass."""
    ang_acc: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Derivative of the angular velocity of the drone's center of mass."""
    rotor_acc: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 4)
    """Derivative of the rotor velocity."""

    @staticmethod
    def create(n_worlds: int, n_drones: int, device: Device) -> SimStateDeriv:
        """Create a default set of state derivatives for the simulation."""
        zeros_3d = jnp.zeros((n_worlds, n_drones, 3), device=device)
        zeros_4d = jnp.zeros((n_worlds, n_drones, 4), device=device)
        return SimStateDeriv(
            vel=zeros_3d, ang_vel=zeros_3d, acc=zeros_3d, ang_acc=zeros_3d, rotor_acc=zeros_4d
        )


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
    steps: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, 1)
    """Last simulation steps that the state control command was applied."""
    freq: int
    """Frequency of the state control command."""
    # Parameters for the controller
    params: dict[str, Array]


@dataclass
class SimControls:
    mode: Control = field(pytree_node=False)
    """Control mode of the simulation."""
    state: ControlData | None
    """State control data."""
    attitude: ControlData | None
    """Attitude control data."""
    force_torque: ControlData | None
    """Force and torque control data."""
    rotor_vel: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 4)
    """Desired motor speed."""

    @staticmethod
    def create(
        n_worlds: int,
        n_drones: int,
        control: Control,
        drone: str,
        state_freq: int | None,
        attitude_freq: int | None,
        force_torque_freq: int | None,
        device: Device,
    ) -> SimControls:
        """Create a default set of controls for the simulation."""
        rotor_vel = jnp.zeros((n_worlds, n_drones, 4), device=device)
        match control:
            case Control.state:
                state = MellingerStateData.create(n_worlds, n_drones, state_freq, drone, device)
                attitude = MellingerAttitudeData.create(
                    n_worlds, n_drones, attitude_freq, drone, device
                )
                force_torque = MellingerForceTorqueData.create(
                    n_worlds, n_drones, force_torque_freq, drone, device
                )
                return SimControls(
                    mode=control,
                    state=state,
                    attitude=attitude,
                    force_torque=force_torque,
                    rotor_vel=rotor_vel,
                )
            case Control.attitude:
                attitude = attitude = MellingerAttitudeData.create(
                    n_worlds, n_drones, attitude_freq, drone, device
                )
                force_torque = MellingerForceTorqueData.create(
                    n_worlds, n_drones, force_torque_freq, drone, device
                )
                return SimControls(
                    mode=control,
                    state=None,
                    attitude=attitude,
                    force_torque=force_torque,
                    rotor_vel=rotor_vel,
                )
            case Control.force_torque:
                force_torque = MellingerForceTorqueData.create(
                    n_worlds, n_drones, force_torque_freq, drone, device
                )
                return SimControls(
                    mode=control,
                    state=None,
                    attitude=None,
                    force_torque=force_torque,
                    rotor_vel=rotor_vel,
                )
            case Control.rotor_vel:
                return SimControls(
                    mode=control, state=None, attitude=None, force_torque=None, rotor_vel=rotor_vel
                )
            case _:
                raise ValueError(f"Control mode {control} not implemented")


class SimParams(typing.Protocol):
    mass: Array  # (1,)
    """Mass of the drone."""
    gravity_vec: Array  # (3,)
    """Gravity vector of the drone."""
    J: Array  # (3, 3)
    """Inertia matrix of the drone."""
    J_inv: Array  # (3, 3)
    """Inverse of the inertia matrix of the drone."""

    @staticmethod
    def create(dynamics: Dynamics, drone: str, device: Device) -> SimParams:
        """Create the default parameters for the simulation."""
        match dynamics:
            case Dynamics.first_principles:
                return FirstPrinciplesParams.create(drone, device)
            case Dynamics.so_rpy:
                return SoRpyParams.create(drone, device)
            case Dynamics.so_rpy_rotor:
                return SoRpyRotorParams.create(drone, device)
            case Dynamics.so_rpy_rotor_drag:
                return SoRpyRotorDragParams.create(drone, device)
            case _:
                raise ValueError(f"Dynamics mode {dynamics} not implemented")


@dataclass
class SimCore:
    freq: int = field(pytree_node=False)
    """Frequency of the simulation."""
    steps: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, 1)
    """Simulation steps taken since the last reset."""
    n_worlds: int = field(pytree_node=False)
    """Number of worlds in the simulation."""
    n_drones: int = field(pytree_node=False)
    """Number of drones in the simulation."""
    drone_mocap_ids: Array = field(metadata={CORE_NDIM_KEY: 1})  # (M,)
    """MuJoCo mocap IDs of the drone bodies."""
    rng_key: Array = field(metadata={CORE_NDIM_KEY: 0})  # ()
    """Random number generator key for the simulation."""
    mjx_synced: Array = field(metadata={CORE_NDIM_KEY: 0})  # ()
    """Whether the simulation data is synchronized with the MuJoCo mjx_data."""

    @staticmethod
    def create(
        freq: int,
        n_worlds: int,
        n_drones: int,
        drone_mocap_ids: Array,
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
            drone_mocap_ids=jnp.array(drone_mocap_ids, dtype=jnp.int32, device=device),
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
    core: SimCore
    """Core parameters of the simulation."""
    plugins: dict[str, Any] = field(default_factory=dict)
    """Arbitrary data for plugins to store state in the simulation.

    Data that is batched over worlds has to declare its dimensions without batch axes using the
    CORE_NDIM_KEY metadata, so that resets can mask it and sharding can distribute it. Dicts have no
    fields to declare on, so batched data has to be stored in a struct.
    """
