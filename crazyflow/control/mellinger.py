from __future__ import annotations

import jax.numpy as jnp
from drone_models.controller.mellinger import MellingerStateParams
from flax.struct import dataclass, field
from jax import Array, Device


@dataclass
class MellingerStateData:
    cmd: Array  # (N, M, 13)
    """Full state control command for the drone.

    A command consists of [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate].
    We currently do not use the acceleration and angle rate components. This is subject to change.
    """
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
        steps = jnp.zeros((n_worlds, 1), dtype=jnp.int32, device=device)
        pos_err_i = jnp.zeros((n_worlds, n_drones, 3), device=device)
        params = MellingerStateParams.load(drone_model)
        return MellingerStateData(
            cmd=cmd, steps=steps, freq=freq, pos_err_i=pos_err_i, params=params
        )


# @dataclass
# class MellingerAttitudeData:
#     cmd: Array  # (N, M, 4)
#     """Full attitude control command for the drone.

#     A command consists of [collective thrust, roll, pitch, yaw].
#     """
#     steps: Array  # (N, 1)
#     """Last simulation steps that the attitude control command was applied."""
#     freq: int = field(pytree_node=False)
#     """Frequency of the attitude control command."""
#     pos_err_i: Array  # (N, M, 3)
#     """Integral errors of the attitude control command."""
#     # Parameters for the attitude controller
#     params: MellingerAttitudeParams

#     @staticmethod
#     def create(
#         n_worlds: int, n_drones: int, freq: int, drone_model: str, device: Device
#     ) -> MellingerAttitudeData:
#         """Create a default set of attitude data for the simulation."""
#         cmd = jnp.zeros((n_worlds, n_drones, 4), device=device)
#         steps = jnp.zeros((n_worlds, 1), dtype=jnp.int32, device=device)
#         pos_err_i = jnp.zeros((n_worlds, n_drones, 3), device=device)
#         params = MellingerAttitudeParams.load(drone_model)
#         return MellingerAttitudeData(
#             cmd=cmd, steps=steps, freq=freq, pos_err_i=pos_err_i, params=params
#         )
