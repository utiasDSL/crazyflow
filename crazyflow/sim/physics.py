"""Physics models for the simulation."""

from __future__ import annotations

from enum import Enum
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from drone_models.first_principles import dynamics as first_principles_dynamics
from drone_models.first_principles.params import FirstPrinciplesParams
from drone_models.so_rpy import dynamics as so_rpy_dynamics
from drone_models.so_rpy.params import SoRpyParams
from flax.struct import dataclass
from jax import Array
from jax.numpy import vectorize
from jax.scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from jax import Device

    from crazyflow.sim.structs import FirstPrinciplesData, SimData, SoRpyData


class Physics(str, Enum):
    """Physics mode for the simulation."""

    first_principles = "first_principles"
    so_rpy = "so_rpy"
    so_rpy_rotor = "so_rpy_rotor"
    so_rpy_rotor_drag = "so_rpy_rotor_drag"
    default = first_principles


# TODO: Check if we cannot reuse the FirstPrinciplesParams class from drone-models or at least make
# the data conversion more efficient for all
@dataclass
class FirstPrinciplesData:
    mass: Array  # (N, M, 1)
    """Mass of the drone."""
    gravity_vec: Array  # (N, M, 3)
    """Gravity vector of the drone."""
    J: Array  # (N, M, 3, 3)
    """Inertia matrix of the drone."""
    J_inv: Array  # (N, M, 3, 3)
    """Inverse of the inertia matrix of the drone."""
    KF: Array  # (N, M, 1)
    """Force constant of the drone."""
    KM: Array  # (N, M, 1)
    """Torque constant of the drone."""
    L: Array  # (N, M, 1)
    """Arm length of the drone."""
    mixing_matrix: Array  # (N, M, 3, 4)
    """Mixing matrix of the drone."""
    thrust_tau: Array  # (N, M, 1)
    """Rotor speed dynamics time constant of the drone."""

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, drone_model: str, device: Device
    ) -> FirstPrinciplesData:
        """Create a default set of parameters for the simulation."""
        params = FirstPrinciplesParams.load(drone_model)
        mass = jnp.full((n_worlds, n_drones, 1), params.mass, device=device)
        gravity_vec = jnp.asarray(params.gravity_vec, device=device)
        J = jnp.tile(params.J[None, None, :, :], (n_worlds, n_drones, 1, 1))
        J = jax.device_put(J, device)
        J_inv = jnp.tile(params.J_inv[None, None, :, :], (n_worlds, n_drones, 1, 1))
        J_inv = jax.device_put(J_inv, device)
        KF = jnp.asarray(params.KF, device=device)
        KM = jnp.asarray(params.KM, device=device)
        L = jnp.asarray(params.L, device=device)
        mixing_matrix = jnp.asarray(params.mixing_matrix, device=device)
        thrust_tau = jnp.asarray(params.thrust_tau, device=device)
        return FirstPrinciplesData(
            mass=mass,
            gravity_vec=gravity_vec,
            J=J,
            J_inv=J_inv,
            KF=KF,
            KM=KM,
            L=L,
            mixing_matrix=mixing_matrix,
            thrust_tau=thrust_tau,
        )


def first_principles_physics(data: SimData) -> SimData:
    """Compute the forces and torques from the first principle physics model."""
    states = data.states
    params: FirstPrinciplesData = data.params
    vel, _, acc, ang_acc, rotor_acc = first_principles_dynamics(
        pos=states.pos,
        quat=states.quat,
        vel=states.vel,
        ang_vel=states.ang_vel,
        cmd=data.controls.rotor_vel,
        rotor_vel=states.rotor_vel,
        dist_f=states.force,
        dist_t=states.torque,
        mass=params.mass,
        gravity_vec=params.gravity_vec,
        J=params.J,
        J_inv=params.J_inv,
        KF=params.KF,
        KM=params.KM,
        L=params.L,
        mixing_matrix=params.mixing_matrix,
        thrust_tau=params.thrust_tau,
    )
    states_deriv = data.states_deriv.replace(
        vel=vel, ang_vel=states.ang_vel, acc=acc, ang_acc=ang_acc, rotor_acc=rotor_acc
    )
    return data.replace(states_deriv=states_deriv)


@dataclass
class SoRpyData:
    mass: Array  # (N, M, 1)
    """Mass of the drone."""
    gravity_vec: Array  # (N, M, 3)
    """Gravity vector of the drone."""
    J: Array  # (N, M, 3, 3)
    """Inertia matrix of the drone."""
    J_inv: Array  # (N, M, 3, 3)
    """Inverse of the inertia matrix of the drone."""
    acc_coef: Array  # (N, M, 1)
    """Coefficient for the acceleration."""
    cmd_f_coef: Array  # (N, M, 1)
    """Coefficient for the collective thrust."""
    rpy_coef: Array  # (N, M, 1)
    """Coefficient for the roll pitch yaw dynamics."""
    rpy_rates_coef: Array  # (N, M, 1)
    """Coefficient for the roll pitch yaw rates dynamics."""
    cmd_rpy_coef: Array  # (N, M, 1)
    """Coefficient for the roll pitch yaw command dynamics."""

    @staticmethod
    def create(n_worlds: int, n_drones: int, drone_model: str, device: Device) -> SoRpyData:
        """Create a default set of parameters for the simulation."""
        params = SoRpyParams.load(drone_model)
        mass = jnp.full((n_worlds, n_drones, 1), params.mass, device=device)
        gravity_vec = jnp.asarray(params.gravity_vec, device=device)
        J = jnp.tile(params.J[None, None, :, :], (n_worlds, n_drones, 1, 1))
        J = jax.device_put(J, device)
        J_inv = jnp.tile(params.J_inv[None, None, :, :], (n_worlds, n_drones, 1, 1))
        J_inv = jax.device_put(J_inv, device)
        acc_coef = jnp.asarray(params.acc_coef, device=device)
        cmd_f_coef = jnp.asarray(params.cmd_f_coef, device=device)
        rpy_coef = jnp.asarray(params.rpy_coef, device=device)
        rpy_rates_coef = jnp.asarray(params.rpy_rates_coef, device=device)
        cmd_rpy_coef = jnp.asarray(params.cmd_rpy_coef, device=device)
        return SoRpyData(
            mass=mass,
            gravity_vec=gravity_vec,
            J=J,
            J_inv=J_inv,
            acc_coef=acc_coef,
            cmd_f_coef=cmd_f_coef,
            rpy_coef=rpy_coef,
            rpy_rates_coef=rpy_rates_coef,
            cmd_rpy_coef=cmd_rpy_coef,
        )


def so_rpy_physics(data: SimData) -> SimData:
    """Compute the forces and torques from the so_rpy physics model."""
    states = data.states
    params: SoRpyData = data.params
    vel, _, acc, ang_acc, _ = so_rpy_dynamics(
        pos=states.pos,
        quat=states.quat,
        vel=states.vel,
        ang_vel=states.ang_vel,
        cmd=data.controls.attitude.cmd,
        dist_f=states.force,
        dist_t=states.torque,
        mass=params.mass,
        gravity_vec=params.gravity_vec,
        J=params.J,
        J_inv=params.J_inv,
        acc_coef=params.acc_coef,
        cmd_f_coef=params.cmd_f_coef,
        rpy_coef=params.rpy_coef,
        rpy_rates_coef=params.rpy_rates_coef,
        cmd_rpy_coef=params.cmd_rpy_coef,
    )
    states_deriv = data.states_deriv.replace(
        vel=vel, ang_vel=states.ang_vel, acc=acc, ang_acc=ang_acc
    )
    return data.replace(states_deriv=states_deriv)


# TODO: Remove this function from the tests
@jax.jit
@partial(vectorize, signature="(3),(4)->(3)")
def ang_vel2rpy_rates(ang_vel: Array, quat: Array) -> Array:
    """Convert angular velocity to rpy rates.

    Args:
        ang_vel: The angular velocity in the body frame.
        quat: The current orientation.

    Returns:
        The rpy rates in the body frame, following the 'xyz' convention.
    """
    rpy = R.from_quat(quat).as_euler("xyz")
    sin_phi, cos_phi = jnp.sin(rpy[0]), jnp.cos(rpy[0])
    cos_theta, tan_theta = jnp.cos(rpy[1]), jnp.tan(rpy[1])
    conv_mat = jnp.array(
        [
            [1, sin_phi * tan_theta, cos_phi * tan_theta],
            [0, cos_phi, -sin_phi],
            [0, sin_phi / cos_theta, cos_phi / cos_theta],
        ]
    )
    return conv_mat @ ang_vel
