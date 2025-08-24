"""Functional programming implementation of the onboard controller.

We reimplement the onboard controller for two reasons:
- We cannot use the C++ bindings of the firmware to differentiate through the onboard controller.
- We need to implement it with JAX to enable efficient, batched computations.

Since our controller is a PID controller, it requires integration of the error over time. We opt for
a functional implementation to avoid storing any state in the class. Doing so would either prevent
us from easily scaling across batches and drones with JAX's `vmap`, or require us to support batches
and multiple drones explicitly in the controller.
"""

from enum import Enum

import jax
from jax import Array


class Control(str, Enum):
    """Control type of the simulated onboard controller."""

    state = "state"
    """State control takes [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate].
    
    Note:
        Recommended frequency is >=20 Hz.

    Warning:
        Currently, we only use positions, velocities, and yaw. The rest of the state is ignored.
        This is subject to change in the future.
    """
    attitude = "attitude"
    """Attitude control takes [collective thrust, roll, pitch, yaw].

    Note:
        Recommended frequency is >=100 Hz.
    """
    force_torque = "force_torque"
    """Force and torque control takes [fx, fy, fz, tx, ty, tz].

    Note:
        Recommended frequency is >=500 Hz.
    """
    default = attitude


@jax.jit
def controllable(step: Array, freq: int, control_steps: Array, control_freq: int) -> Array:
    """Check which worlds can currently update their controllers.

    Args:
        step: The current step of the simulation.
        freq: The frequency of the simulation.
        control_steps: The steps at which the controllers were last updated.
        control_freq: The frequency of the controllers.

    Returns:
        A boolean mask of shape (n_worlds,) that is True at the worlds where the controllers can be
        updated.
    """
    return ((step - control_steps) >= (freq / control_freq)) | (control_steps == -1)


# TODO: Remove these constants
KF: float = 3.16e-10
PWM2RPM_SCALE: float = 0.2685
PWM2RPM_CONST: float = 4070.3
MIN_PWM: float = 20000
MAX_PWM: float = 65535
MIN_RPM: float = PWM2RPM_SCALE * MIN_PWM + PWM2RPM_CONST
MAX_RPM: float = PWM2RPM_SCALE * MAX_PWM + PWM2RPM_CONST
MIN_THRUST: float = KF * MIN_RPM**2
MAX_THRUST: float = KF * MAX_RPM**2
