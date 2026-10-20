"""Core tools for registering and capability checking for the drone dynamics."""

from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

import numpy as np

from crazyflow.drones import Drone
from crazyflow.utils import filter_to_signature, to_xp
from crazyflow.utils import parametrize as _parametrize

if TYPE_CHECKING:
    from types import FunctionType, ModuleType

F = TypeVar("F", bound=Callable[..., Any])
P = ParamSpec("P")
R = TypeVar("R")


class Dynamics(StrEnum):
    """Dynamics mode for the simulation."""

    first_principles = "first_principles"
    so_rpy = "so_rpy"
    so_rpy_rotor = "so_rpy_rotor"
    so_rpy_rotor_drag = "so_rpy_rotor_drag"
    default = first_principles


def supports(rotor_dynamics: bool = True) -> Callable[[F], F]:
    """Decorator that declares which optional inputs a dynamics function supports.

    The decorator attaches a ``__dynamics_features__`` attribute to the wrapper, which
    [dynamics_features][crazyflow.dynamics.dynamics_features] reads.

    Args:
        rotor_dynamics: Whether the decorated function models rotor velocity dynamics. Set to
            ``False`` for models that do not accept or integrate ``rotor_vel`` (e.g. ``so_rpy``).
            Defaults to ``True``.

    Returns:
        The function decorated with capability flags.
    """

    def decorator(fn: F) -> F:
        setattr(fn, "__dynamics_features__", {"rotor_dynamics": rotor_dynamics})
        return fn

    return decorator


def parametrize(
    fn: Callable[P, R], drone: Drone, xp: ModuleType | None = None, device: str | None = None
) -> Callable[P, R]:
    """Parametrize a dynamics function with the default dynamics parameters for a drone.

    Args:
        fn: The dynamics function to parametrize.
        drone: The drone to use.
        xp: The array API module to use. If not provided, numpy is used.
        device: The device to use. If none, the device is inferred from the xp module.

    Example:
    ```python
    import numpy as np
    from crazyflow.drones import Drone
    from crazyflow.dynamics.core import parametrize
    from crazyflow.dynamics.first_principles import dynamics

    dynamics_fn = parametrize(dynamics, Drone.cf2x_L250)
    pos, quat = np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
    vel, ang_vel = np.zeros(3), np.zeros(3)
    rotor_vel, cmd = np.zeros(4), np.zeros(4)
    pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = dynamics_fn(
        pos=pos, quat=quat, vel=vel, ang_vel=ang_vel, cmd=cmd, rotor_vel=rotor_vel
    )
    ```

    Returns:
        The parametrized dynamics function with all keyword argument only parameters filled in.
    """
    return _parametrize(fn, drone, load_fn_params, xp=xp, device=device)


def load_params(
    dynamics: Dynamics, drone: Drone, xp: ModuleType | None = None, device: str | None = None
) -> dict:
    """Load all parameters of a drone for a dynamics model.

    Merges the global parameters in ``crazyflow/dynamics/params.toml`` with the drone's section in
    ``crazyflow/dynamics/<dynamics>/params.toml`` and adds ``J_inv``.

    Args:
        dynamics: The dynamics model, e.g. ``Dynamics.so_rpy``.
        drone: The drone configuration, e.g. ``Drone.cf2x_L250``.
        xp: Array API module used to convert parameter values. If ``None``, NumPy is used.
        device: The device to use for the arrays. If ``None``, the device is inferred from the xp
            module.

    Returns:
        A flat dict mapping parameter names to arrays in the requested array namespace.

    Raises:
        ValueError: If ``dynamics`` or ``drone`` is unknown.
        KeyError: If ``drone`` has no section for ``dynamics``.
    """
    dynamics = Dynamics(dynamics)
    if dynamics not in supported_dynamics(drone):
        raise KeyError(f"Drone `{drone}` not found in {dynamics}/params.toml")
    with open(Path(__file__).parent / "params.toml", "rb") as f:
        global_params = tomllib.load(f)
    with open(Path(__file__).parent / f"{dynamics}/params.toml", "rb") as f:
        dynamics_params = tomllib.load(f)
    params = global_params | dynamics_params[drone]
    # Make sure J_inv does not have a dtype fixed before conversion to xp arrays to avoid fixing it
    # to np.float64 when other frameworks might prefer a different dtype.
    params["J_inv"] = np.linalg.inv(params["J"]).tolist()
    return to_xp(params, xp=xp, device=device)


def load_fn_params(
    fn: FunctionType, drone: Drone, xp: ModuleType | None = None, device: str | None = None
) -> dict:
    """Load the parameters a dynamics function accepts.

    The dynamics model is derived from the function's package, so ``fn`` must be defined in
    ``crazyflow/dynamics/<dynamics>/``. Only the keyword-only parameters of ``fn`` are kept.

    Args:
        fn: The dynamics function for which to load parameters.
        drone: The drone configuration, e.g. ``Drone.cf2x_L250``.
        xp: Array API module used to convert parameter values. If ``None``, NumPy is used.
        device: The device to use for the arrays. If ``None``, the device is inferred from the xp
            module.

    Returns:
        A flat dict mapping parameter names to arrays in the requested array namespace.
    """
    assert callable(fn), f"Expected a function, got {type(fn)}"
    dynamics = Dynamics(fn.__module__.split(".")[-2])
    return filter_to_signature(load_params(dynamics, drone, xp=xp, device=device), fn)


def _param_sections(dynamics: Dynamics) -> set[str]:
    """Return the drone sections declared in a dynamics model's ``params.toml``."""
    with open(Path(__file__).parent / f"{dynamics}/params.toml", "rb") as f:
        return set(tomllib.load(f))


def supported_drones(dynamics: Dynamics) -> tuple[Drone, ...]:
    """Return the drones that ``dynamics`` can be parametrized for.

    A drone is supported when ``crazyflow/dynamics/<dynamics>/params.toml`` has a section for it.

    Args:
        dynamics: The dynamics model, e.g. ``Dynamics.so_rpy``.

    Returns:
        The supported drones in the order of [Drone][crazyflow.drones.Drone].
    """
    dynamics = Dynamics(dynamics)
    return tuple(drone for drone in Drone if drone in _param_sections(dynamics))


def supported_dynamics(drone: Drone) -> tuple[Dynamics, ...]:
    """Return the dynamics models that ``drone`` can be simulated with.

    A model is supported when its ``crazyflow/dynamics/<dynamics>/params.toml`` has a section for
    ``drone``.

    Args:
        drone: The drone configuration, e.g. ``Drone.cf2x_L250``.

    Returns:
        The supported models in the order of [Dynamics][crazyflow.dynamics.Dynamics].
    """
    drone = Drone(drone)
    return tuple(d for d in Dynamics if drone in _param_sections(d))
