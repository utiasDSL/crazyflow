"""Core functionalities for controller parametrization."""

from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ParamSpec, TypeVar

import jax
from jax import Array

from crazyflow.utils import filter_to_signature, to_xp
from crazyflow.utils import parametrize as _parametrize

if TYPE_CHECKING:
    from types import FunctionType, ModuleType

    from crazyflow._typing import Array  # To be changed to array_api_typing later
    from crazyflow.drones import Drone

P = ParamSpec("P")
R = TypeVar("R")


class Control(StrEnum):
    """Control type of the simulated onboard controller."""

    state = "state"
    """State control takes [x, y, z, vx, vy, vz, ax, ay, az, qx, qy, qz, qw, wx, wy, wz].

    Note:
        Recommended frequency is >=20 Hz.

    Warning:
        Only the yaw of the attitude quaternion is used, as in the firmware. The so_rpy family
        ignores the body rate setpoint.
    """
    attitude = "attitude"
    """Attitude control takes [roll, pitch, yaw, collective thrust].

    Note:
        Recommended frequency is >=100 Hz.
    """
    body_rate = "body_rate"
    """Body rate control takes [wx, wy, wz, collective thrust].

    Note:
        Recommended frequency is >=200 Hz.
    """
    force_torque = "force_torque"
    """Force and torque control takes [fc, tx, ty, tz].

    Note:
        Recommended frequency is >=500 Hz.
    """
    rotor_vel = "rotor_vel"
    """Rotor velocity control takes [w1, w2, w3, w4] in RPMs.

    Note:
        Recommended frequency is >=500 Hz.
    """
    default = attitude


def parametrize(
    fn: Callable[P, R], drone: Drone, xp: ModuleType | None = None, device: str | None = None
) -> Callable[P, R]:
    """Parametrize a controller function with the default controller parameters for a drone.

    Args:
        fn: The controller function to parametrize.
        drone: The drone to use.
        xp: The array API module to use. If not provided, numpy is used.
        device: The device to use. If None, the device is inferred from the xp module.

    Example:
    ```python
    import numpy as np
    from crazyflow.control import parametrize
    from crazyflow.control.mellinger import state2attitude
    from crazyflow.drones import Drone
    from scipy.spatial.transform import Rotation as R

    ctrl = parametrize(state2attitude, Drone.cf2x_L250)
    pos, quat = np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
    vel, cmd = np.zeros(3), np.zeros(16)
    cmd[9:13] = R.from_euler("z", 0.0).as_quat()
    rpyt, int_pos_err = ctrl(pos, quat, vel, cmd)
    ```

    Returns:
        The parametrized controller function with all keyword argument only parameters filled in.
    """
    return _parametrize(fn, drone, load_fn_params, xp=xp, device=device)


def load_params(
    controller: str, drone: Drone, xp: ModuleType | None = None, device: str | None = None
) -> dict[str, dict[str, Array]]:
    """Load all parameters of a drone for a controller.

    Returns the drone's table of ``crazyflow/control/<controller>/params.toml``: the ``core``
    section shared by all functions of the controller, and one section per controller function.

    Args:
        controller: Name of the controller package, e.g. ``"mellinger"``.
        drone: The drone configuration, e.g. ``Drone.cf2x_L250``.
        xp: The array API module to use. If not provided, numpy is used.
        device: The device to use. If None, the device is inferred from the xp module.

    Raises:
        KeyError: If ``controller`` has no ``params.toml`` or ``drone`` has no section in it.
    """
    params_path = Path(__file__).parent / f"{controller}/params.toml"
    if not params_path.exists():
        raise KeyError(f"Controller `{controller}` not found")
    with open(params_path, "rb") as f:
        params = tomllib.load(f)
    if drone not in params:
        raise KeyError(f"Drone `{drone}` not found in {controller}/params.toml")
    return to_xp(params[drone], xp=xp, device=device)


def load_fn_params(
    fn: FunctionType, drone: Drone, xp: ModuleType | None = None, device: str | None = None
) -> dict[str, Array]:
    """Load the parameters a controller function accepts.

    The controller is derived from the function's package. Merges the ``core`` section with the
    function's section (function values take precedence), then keeps only the parameters in
    ``fn``'s signature.

    Args:
        fn: The controller function for which to load parameters.
        drone: The drone configuration, e.g. ``Drone.cf2x_L250``.
        xp: The array API module to use. If not provided, numpy is used.
        device: The device to use. If None, the device is inferred from the xp module.

    Returns:
        A flat dict mapping parameter names to arrays in the requested array namespace.
    """
    assert callable(fn), f"Expected a function, got {type(fn)}"
    params = load_params(fn.__module__.split(".")[-2], drone, xp=xp, device=device)
    return filter_to_signature(params.get("core", {}) | params.get(fn.__name__, {}), fn)


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
