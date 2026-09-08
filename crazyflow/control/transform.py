"""Transformations between physical parameters of the quadrotors.

Bundles conversions between motor forces, rotor velocities, and PWM commands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from array_api_compat import array_namespace
from array_api_compat import device as xp_device

from crazyflow.utils import to_xp

if TYPE_CHECKING:
    from crazyflow._typing import Array  # To be changed to array_api_typing later


def motor_force2rotor_vel(motor_forces: Array, rpm2thrust: Array) -> Array:
    """Convert motor forces to rotor velocities, where f=a*rpm^2+b*rpm+c.

    Args:
        motor_forces: Motor forces in SI units with shape (..., N).
        rpm2thrust: RPM to thrust conversion factors with shape (3,), shared (1, 3) or one curve per
            motor (N, 3), optionally with leading batch axes.

    Returns:
        Array of rotor velocities in RPMs with shape (..., N).
    """
    xp = array_namespace(motor_forces)
    rpm2thrust = to_xp(rpm2thrust, xp=xp, device=xp_device(motor_forces))
    # shared (1, 3) and per-motor (..., N, 3) coefficients both broadcast against motor_forces.
    c, b, a = rpm2thrust[..., 0], rpm2thrust[..., 1], rpm2thrust[..., 2]
    return (-b + xp.sqrt(b**2 - 4 * a * (c - motor_forces))) / (2 * a)


def force2pwm(thrust: Array | float, thrust_max: Array | float, pwm_max: Array | float) -> Array:
    """Convert thrust in N to thrust in PWM.

    Args:
        thrust: Array or float of the thrust in [N]
        thrust_max: Maximum thrust in [N]
        pwm_max: Maximum PWM value

    Returns:
        Thrust converted in PWM.
    """
    return thrust / thrust_max * pwm_max


def pwm2force(
    pwm: Array | float, thrust_max: Array | float, pwm_max: Array | float
) -> Array | float:
    """Convert pwm thrust command to actual thrust.

    Args:
        pwm: Array or float of the pwm value
        thrust_max: Maximum thrust in [N]
        pwm_max: Maximum PWM value

    Returns:
        thrust: Array or float thrust in [N]
    """
    return pwm / pwm_max * thrust_max
