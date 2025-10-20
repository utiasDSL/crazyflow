"""Physics models for the simulation."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from drone_models.core import load_params
from drone_models.first_principles import dynamics as first_principles_dynamics
from drone_models.so_rpy import dynamics as so_rpy_dynamics
from drone_models.so_rpy_rotor import dynamics as so_rpy_rotor_dynamics
from drone_models.so_rpy_rotor_drag import dynamics as so_rpy_rotor_drag_dynamics
from flax.struct import dataclass
from jax import Array

if TYPE_CHECKING:
    from jax import Device

    from crazyflow.sim.data import SimData


class Physics(str, Enum):
    """Physics mode for the simulation."""

    first_principles = "first_principles"
    so_rpy = "so_rpy"
    so_rpy_rotor = "so_rpy_rotor"
    so_rpy_rotor_drag = "so_rpy_rotor_drag"
    default = first_principles


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
    rpm2thrust: Array  # (N, M, 1)
    """Force constant of the drone."""
    rpm2torque: Array  # (N, M, 1)
    """Torque constant of the drone."""
    L: Array  # (N, M, 1)
    """Arm length of the drone."""
    mixing_matrix: Array  # (N, M, 3, 4)
    """Mixing matrix of the drone."""
    rotor_tau: Array  # (N, M, 1)
    """Rotor speed dynamics time constant of the drone."""

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, drone_model: str, device: Device
    ) -> FirstPrinciplesData:
        """Create a default set of parameters for the simulation."""
        p = load_params("first_principles", drone_model)
        J = jax.device_put(jnp.tile(p["J"][None, None, :, :], (n_worlds, n_drones, 1, 1)), device)
        return FirstPrinciplesData(
            mass=jnp.full((n_worlds, n_drones, 1), p["mass"], device=device),
            gravity_vec=jnp.asarray(p["gravity_vec"], device=device),
            J=J,
            J_inv=jnp.linalg.inv(J),
            rpm2thrust=jnp.asarray(p["rpm2thrust"], device=device),
            rpm2torque=jnp.asarray(p["rpm2torque"], device=device),
            L=jnp.asarray(p["L"], device=device),
            mixing_matrix=jnp.asarray(p["mixing_matrix"], device=device),
            rotor_tau=jnp.asarray(p["rotor_tau"], device=device),
        )


def first_principles_physics(data: SimData) -> SimData:
    """Compute the forces and torques from the first principle physics model."""
    params: FirstPrinciplesData = data.params
    vel, _, acc, ang_acc, rotor_acc = first_principles_dynamics(
        pos=data.states.pos,
        quat=data.states.quat,
        vel=data.states.vel,
        ang_vel=data.states.ang_vel,
        cmd=data.controls.rotor_vel,
        rotor_vel=data.states.rotor_vel,
        dist_f=data.states.force,
        dist_t=data.states.torque,
        mass=params.mass,
        gravity_vec=params.gravity_vec,
        J=params.J,
        J_inv=params.J_inv,
        rpm2thrust=params.rpm2thrust,
        rpm2torque=params.rpm2torque,
        L=params.L,
        mixing_matrix=params.mixing_matrix,
        rotor_tau=params.rotor_tau,
    )
    states_deriv = data.states_deriv.replace(
        vel=vel, ang_vel=data.states.ang_vel, acc=acc, ang_acc=ang_acc, rotor_acc=rotor_acc
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
        p = load_params("so_rpy", drone_model)
        J = jax.device_put(jnp.tile(p["J"][None, None, :, :], (n_worlds, n_drones, 1, 1)), device)
        return SoRpyData(
            mass=jnp.full((n_worlds, n_drones, 1), p["mass"], device=device),
            gravity_vec=jnp.asarray(p["gravity_vec"], device=device),
            J=J,
            J_inv=jnp.linalg.inv(J),
            acc_coef=jnp.asarray(p["acc_coef"], device=device),
            cmd_f_coef=jnp.asarray(p["cmd_f_coef"], device=device),
            rpy_coef=jnp.asarray(p["rpy_coef"], device=device),
            rpy_rates_coef=jnp.asarray(p["rpy_rates_coef"], device=device),
            cmd_rpy_coef=jnp.asarray(p["cmd_rpy_coef"], device=device),
        )


def so_rpy_physics(data: SimData) -> SimData:
    """Compute the forces and torques from the so_rpy physics model."""
    params: SoRpyData = data.params
    vel, _, acc, ang_acc, _ = so_rpy_dynamics(
        pos=data.states.pos,
        quat=data.states.quat,
        vel=data.states.vel,
        ang_vel=data.states.ang_vel,
        cmd=data.controls.attitude.cmd,
        dist_f=data.states.force,
        dist_t=data.states.torque,
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
        vel=vel, ang_vel=data.states.ang_vel, acc=acc, ang_acc=ang_acc
    )
    return data.replace(states_deriv=states_deriv)


@dataclass
class SoRpyRotorData:
    mass: Array  # (N, M, 1)
    """Mass of the drone."""
    gravity_vec: Array  # (N, M, 3)
    """Gravity vector of the drone."""
    J: Array  # (N, M, 3, 3)
    """Inertia matrix of the drone."""
    J_inv: Array  # (N, M, 3, 3)
    """Inverse of the inertia matrix of the drone."""
    thrust_time_coef: Array  # (N, M, 1)
    """Rotor coefficient of the drone."""
    acc_coef: Array  # (N, M, 1)
    """Acceleration coefficient of the drone."""
    cmd_f_coef: Array  # (N, M, 1)
    """Collective thrust coefficient of the drone."""
    rpy_coef: Array  # (N, M, 1)
    """Roll pitch yaw coefficient of the drone."""
    rpy_rates_coef: Array  # (N, M, 1)
    """Roll pitch yaw rates coefficient of the drone."""
    cmd_rpy_coef: Array  # (N, M, 1)
    """Roll pitch yaw command coefficient of the drone."""

    @staticmethod
    def create(n_worlds: int, n_drones: int, drone_model: str, device: Device) -> SoRpyRotorData:
        """Create a default set of parameters for the simulation."""
        p = load_params("so_rpy_rotor", drone_model)
        J = jax.device_put(jnp.tile(p["J"][None, None, :, :], (n_worlds, n_drones, 1, 1)), device)
        return SoRpyRotorData(
            mass=jnp.full((n_worlds, n_drones, 1), p["mass"], device=device),
            gravity_vec=jnp.asarray(p["gravity_vec"], device=device),
            J=J,
            J_inv=jnp.linalg.inv(J),
            thrust_time_coef=jnp.asarray(p["thrust_time_coef"], device=device),
            acc_coef=jnp.asarray(p["acc_coef"], device=device),
            cmd_f_coef=jnp.asarray(p["cmd_f_coef"], device=device),
            rpy_coef=jnp.asarray(p["rpy_coef"], device=device),
            rpy_rates_coef=jnp.asarray(p["rpy_rates_coef"], device=device),
            cmd_rpy_coef=jnp.asarray(p["cmd_rpy_coef"], device=device),
        )


def so_rpy_rotor_physics(data: SimData) -> SimData:
    """Compute the forces and torques from the so_rpy_rotor physics model."""
    params: SoRpyRotorData = data.params
    vel, _, acc, ang_acc, _ = so_rpy_rotor_dynamics(
        pos=data.states.pos,
        quat=data.states.quat,
        vel=data.states.vel,
        ang_vel=data.states.ang_vel,
        cmd=data.controls.attitude.cmd,
        rotor_vel=None,  # TODO: Add rotor_vel once the parameters are available
        dist_f=data.states.force,
        dist_t=data.states.torque,
        mass=params.mass,
        gravity_vec=params.gravity_vec,
        J=params.J,
        J_inv=params.J_inv,
        thrust_time_coef=params.thrust_time_coef,
        acc_coef=params.acc_coef,
        cmd_f_coef=params.cmd_f_coef,
        rpy_coef=params.rpy_coef,
        rpy_rates_coef=params.rpy_rates_coef,
        cmd_rpy_coef=params.cmd_rpy_coef,
    )
    # TODO: Add rotor_vel to the states_deriv
    states_deriv = data.states_deriv.replace(
        vel=vel, ang_vel=data.states.ang_vel, acc=acc, ang_acc=ang_acc
    )
    return data.replace(states_deriv=states_deriv)


@dataclass
class SoRpyRotorDragData:
    mass: Array  # (N, M, 1)
    """Mass of the drone."""
    gravity_vec: Array  # (N, M, 3)
    """Gravity vector of the drone."""
    J: Array  # (N, M, 3, 3)
    """Inertia matrix of the drone."""
    J_inv: Array  # (N, M, 3, 3)
    """Inverse of the inertia matrix of the drone."""
    thrust_time_coef: Array  # (N, M, 1)
    """Rotor coefficient of the drone."""
    acc_coef: Array  # (N, M, 1)
    """Acceleration coefficient of the drone."""
    cmd_f_coef: Array  # (N, M, 1)
    """Collective thrust coefficient of the drone."""
    rpy_coef: Array  # (N, M, 1)
    """Roll pitch yaw coefficient of the drone."""
    rpy_rates_coef: Array  # (N, M, 1)
    """Roll pitch yaw rates coefficient of the drone."""
    cmd_rpy_coef: Array  # (N, M, 1)
    """Roll pitch yaw command coefficient of the drone."""
    drag_linear_coef: Array  # (N, M, 1)
    """Linear drag coefficient of the drone."""
    drag_square_coef: Array  # (N, M, 1)
    """Square drag coefficient of the drone."""

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, drone_model: str, device: Device
    ) -> SoRpyRotorDragData:
        """Create a default set of parameters for the simulation."""
        p = load_params("so_rpy_rotor_drag", drone_model)
        J = jax.device_put(jnp.tile(p["J"][None, None, :, :], (n_worlds, n_drones, 1, 1)), device)
        return SoRpyRotorDragData(
            mass=jnp.full((n_worlds, n_drones, 1), p["mass"], device=device),
            gravity_vec=jnp.asarray(p["gravity_vec"], device=device),
            J=J,
            J_inv=jnp.linalg.inv(J),
            thrust_time_coef=jnp.asarray(p["thrust_time_coef"], device=device),
            acc_coef=jnp.asarray(p["acc_coef"], device=device),
            cmd_f_coef=jnp.asarray(p["cmd_f_coef"], device=device),
            rpy_coef=jnp.asarray(p["rpy_coef"], device=device),
            rpy_rates_coef=jnp.asarray(p["rpy_rates_coef"], device=device),
            cmd_rpy_coef=jnp.asarray(p["cmd_rpy_coef"], device=device),
            drag_linear_coef=jnp.asarray(p["drag_linear_coef"], device=device),
            drag_square_coef=jnp.asarray(p["drag_square_coef"], device=device),
        )


def so_rpy_rotor_drag_physics(data: SimData) -> SimData:
    """Compute the forces and torques from the so_rpy_rotor_drag physics model."""
    params: SoRpyRotorDragData = data.params
    vel, _, acc, ang_acc, rotor_acc = so_rpy_rotor_drag_dynamics(
        pos=data.states.pos,
        quat=data.states.quat,
        vel=data.states.vel,
        ang_vel=data.states.ang_vel,
        cmd=data.controls.attitude.cmd,
        rotor_vel=data.states.rotor_vel,
        dist_f=data.states.force,
        dist_t=data.states.torque,
        mass=params.mass,
        gravity_vec=params.gravity_vec,
        J=params.J,
        J_inv=params.J_inv,
        thrust_time_coef=params.thrust_time_coef,
        acc_coef=params.acc_coef,
        cmd_f_coef=params.cmd_f_coef,
        rpy_coef=params.rpy_coef,
        rpy_rates_coef=params.rpy_rates_coef,
        cmd_rpy_coef=params.cmd_rpy_coef,
        drag_linear_coef=params.drag_linear_coef,
        drag_square_coef=params.drag_square_coef,
    )
    states_deriv = data.states_deriv.replace(
        vel=vel, ang_vel=data.states.ang_vel, acc=acc, ang_acc=ang_acc, rotor_acc=rotor_acc
    )
    return data.replace(states_deriv=states_deriv)
