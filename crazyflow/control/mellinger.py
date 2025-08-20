from __future__ import annotations

import jax.numpy as jnp
from drone_models.controller.mellinger import (
    MellingerAttitudeParams,
    MellingerForceTorqueParams,
    MellingerStateParams,
)
from flax.struct import dataclass, field
from jax import Array, Device


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
    params: MellingerStateParams

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, freq: int, drone_model: str, device: Device
    ) -> MellingerStateData:
        """Create a default set of state data for the simulation."""
        cmd = jnp.zeros((n_worlds, n_drones, 13), device=device)
        steps = -jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device)
        pos_err_i = jnp.zeros((n_worlds, n_drones, 3), device=device)
        params = MellingerStateParams.load(drone_model)
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
    params: MellingerAttitudeParams

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, freq: int, drone_model: str, device: Device
    ) -> MellingerAttitudeData:
        """Create a default set of attitude data for the simulation."""
        cmd = jnp.zeros((n_worlds, n_drones, 4), device=device)
        steps = -jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device)
        zeros_3d = jnp.zeros((n_worlds, n_drones, 3), device=device)
        params = MellingerAttitudeParams.load(drone_model)
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
    cmd_force: Array  # (N, M, 1)
    """Force command for the drone.

    A command consists of [fz].
    """
    cmd_torque: Array  # (N, M, 3)
    """Torque command for the drone.

    A command consists of [tx, ty, tz].
    """
    staged_cmd_force: Array  # (N, M, 1)
    staged_cmd_torque: Array  # (N, M, 3)
    """Staging buffer to store the most recent command until the next controller tick."""
    steps: Array  # (N, 1)
    """Last simulation steps that the force and torque control command was applied."""
    freq: int = field(pytree_node=False)
    """Frequency of the force and torque control command."""
    # Parameters for the force and torque controller
    params: MellingerForceTorqueParams

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, freq: int, drone_model: str, device: Device
    ) -> MellingerForceTorqueData:
        zero_1d = jnp.zeros((n_worlds, n_drones, 1), device=device)
        zero_3d = jnp.zeros((n_worlds, n_drones, 3), device=device)
        steps = -jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device)
        params = MellingerForceTorqueParams.load(drone_model)
        return MellingerForceTorqueData(
            cmd_force=zero_1d,
            cmd_torque=zero_3d,
            staged_cmd_force=zero_1d,
            staged_cmd_torque=zero_3d,
            steps=steps,
            freq=freq,
            params=params,
        )
