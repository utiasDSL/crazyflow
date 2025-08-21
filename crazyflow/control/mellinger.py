from __future__ import annotations

import jax.numpy as jnp
from drone_models.controller.mellinger.params import AttitudeParams, ForceTorqueParams, StateParams
from flax.struct import dataclass, field
from jax import Array, Device

from crazyflow.utils import named_tuple2device


@dataclass
class MellingerStateData:
    cmd: Array  # (N, M, 13)
    """Full state control command for the drone.

    A command consists of [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate].
    We currently do not use the acceleration and angle rate components. This is subject to change.
    """
    staged_cmd: Array  # (N, M, 13)
    """Staging buffer to store the most recent command until the next controller tick."""
    steps: Array  # (N, 1)
    """Last simulation steps that the state control command was applied."""
    freq: int = field(pytree_node=False)
    """Frequency of the state control command."""
    pos_err_i: Array  # (N, M, 3)
    """Integral errors of the state control command."""
    # Parameters for the state controller
    params: StateParams

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, freq: int, drone_model: str, device: Device
    ) -> MellingerStateData:
        """Create a default set of state data for the simulation."""
        cmd = jnp.zeros((n_worlds, n_drones, 13), device=device)
        steps = -jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device)
        pos_err_i = jnp.zeros((n_worlds, n_drones, 3), device=device)
        params = named_tuple2device(StateParams.load(drone_model), device)
        return MellingerStateData(
            cmd=cmd, staged_cmd=cmd, steps=steps, freq=freq, pos_err_i=pos_err_i, params=params
        )


@dataclass
class MellingerAttitudeData:
    cmd: Array  # (N, M, 4)
    """Full attitude control command for the drone.

    A command consists of [roll, pitch, yaw, collective thrust].
    """
    staged_cmd: Array  # (N, M, 4)
    """Staging buffer to store the most recent command until the next controller tick."""
    steps: Array  # (N, 1)
    """Last simulation steps that the attitude control command was applied."""
    freq: int = field(pytree_node=False)
    """Frequency of the attitude control command."""
    r_int_error: Array  # (N, M, 3)
    """Integral errors of the attitude control command."""
    last_ang_vel: Array  # (N, M, 3)
    """Last angular velocity of the drone."""
    # Parameters for the attitude controller
    params: AttitudeParams

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, freq: int, drone_model: str, device: Device
    ) -> MellingerAttitudeData:
        """Create a default set of attitude data for the simulation."""
        cmd = jnp.zeros((n_worlds, n_drones, 4), device=device)
        steps = -jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device)
        zeros_3d = jnp.zeros((n_worlds, n_drones, 3), device=device)
        params = named_tuple2device(AttitudeParams.load(drone_model), device)
        return MellingerAttitudeData(
            cmd=cmd,
            staged_cmd=cmd,
            steps=steps,
            freq=freq,
            r_int_error=zeros_3d,
            last_ang_vel=zeros_3d,
            params=params,
        )


@dataclass
class MellingerForceTorqueData:
    cmd: Array  # (N, M, 4)
    """Force-torque command for the drone.

    A command consists of [fz, tx, ty, tz].
    """
    staged_cmd: Array  # (N, M, 4)
    """Staging buffer to store the most recent command until the next controller tick."""
    steps: Array  # (N, 1)
    """Last simulation steps that the force and torque control command was applied."""
    freq: int = field(pytree_node=False)
    """Frequency of the force and torque control command."""
    # Parameters for the force and torque controller
    params: ForceTorqueParams

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, freq: int, drone_model: str, device: Device
    ) -> MellingerForceTorqueData:
        zero_4d = jnp.zeros((n_worlds, n_drones, 4), device=device)
        steps = -jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device)
        params = named_tuple2device(ForceTorqueParams.load(drone_model), device)
        return MellingerForceTorqueData(
            cmd=zero_4d, staged_cmd=zero_4d, steps=steps, freq=freq, params=params
        )
