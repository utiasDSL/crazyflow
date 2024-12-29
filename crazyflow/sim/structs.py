from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field
from jax import Array, Device

if TYPE_CHECKING:
    from mujoco.mjx import Data, Model


@dataclass
class SimState:
    pos: Array  # (N, M, 3)
    quat: Array  # (N, M, 4)
    vel: Array  # (N, M, 3)
    rpy_rates: Array  # (N, M, 3)
    force: Array  # (N, M, 3)  # CoM force
    torque: Array  # (N, M, 3)  # CoM torque
    motor_forces: Array  # (N, M, 4)  # Motor forces along body frame z axis
    motor_torques: Array  # (N, M, 4)  # Motor torques around the body frame z axis

    @staticmethod
    def create(n_worlds: int, n_drones: int, device: Device) -> SimState:
        """Create a default set of states for the simulation."""
        pos = jnp.zeros((n_worlds, n_drones, 3), device=device)
        quat = jnp.zeros((n_worlds, n_drones, 4), device=device)
        quat = quat.at[..., -1].set(1.0)
        vel = jnp.zeros((n_worlds, n_drones, 3), device=device)
        rpy_rates = jnp.zeros((n_worlds, n_drones, 3), device=device)
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
            rpy_rates=rpy_rates,
        )


@dataclass
class SimStateDeriv:
    dpos: Array  # (N, M, 3)
    drot: Array  # (N, M, 3)
    dvel: Array  # (N, M, 3)
    drpy_rates: Array  # (N, M, 3)

    @staticmethod
    def create(n_worlds: int, n_drones: int, device: Device) -> SimStateDeriv:
        """Create a default set of state derivatives for the simulation."""
        dpos = jnp.zeros((n_worlds, n_drones, 3), device=device)
        drot = jnp.zeros((n_worlds, n_drones, 3), device=device)
        dvel = jnp.zeros((n_worlds, n_drones, 3), device=device)
        drpy_rates = jnp.zeros((n_worlds, n_drones, 3), device=device)
        return SimStateDeriv(dpos=dpos, drot=drot, dvel=dvel, drpy_rates=drpy_rates)


@dataclass
class SimControls:
    state: Array  # (N, M, 13)
    state_steps: Array  # (N, 1)
    state_freq: int = field(pytree_node=False)
    attitude: Array  # (N, M, 4)
    staged_attitude: Array  # (N, M, 4)
    attitude_steps: Array  # (N, 1)
    attitude_freq: int = field(pytree_node=False)
    thrust: Array  # (N, M, 4)
    thrust_steps: Array  # (N, 1)
    thrust_freq: int = field(pytree_node=False)
    rpms: Array  # (N, M, 4)
    rpy_err_i: Array  # (N, M, 3)
    pos_err_i: Array  # (N, M, 3)
    last_rpy: Array  # (N, M, 3)

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


@dataclass
class SimParams:
    mass: Array  # (N, M, 1)
    J: Array  # (N, M, 3, 3)
    J_INV: Array  # (N, M, 3, 3)

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
    steps: Array  # (N, 1)
    n_worlds: int = field(pytree_node=False)
    n_drones: int = field(pytree_node=False)
    drone_ids: Array  # (1, M)
    rng_key: Array  # (N, 1)

    @staticmethod
    def create(
        freq: int, n_worlds: int, n_drones: int, drone_ids: Array, rng_key: int, device: Device
    ) -> SimCore:
        """Create a default set of core simulation parameters."""
        steps = jnp.zeros((n_worlds, 1), dtype=jnp.int32, device=device)
        rng_key = jax.device_put(jax.random.key(rng_key), device)
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
    states_deriv: SimStateDeriv
    controls: SimControls
    params: SimParams
    core: SimCore
    mjx_data: Data
    mjx_model: Model | None
