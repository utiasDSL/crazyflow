"""Quadrotor dynamics for estimation, control, and simulation.

This package provides numeric and symbolic quadrotor dynamics at multiple fidelity levels. The
dynamics are implemented as pure functions compatible with any Array API backend (NumPy, JAX,
PyTorch, etc.) and with CasADi for symbolic computation.

The dynamics are at the core of Crazyflow's simulation. However, they are written to be as
self-contained as possible, so that they can be used independently for other purposes, such as state
estimation or control design.

Use [parametrize][crazyflow.dynamics.parametrize] to bind a dynamics function to a named drone
configuration, and ``available_dynamics`` to enumerate all registered dynamics.
"""

from functools import partial
from typing import Callable

from crazyflow.dynamics.core import (
    Dynamics,
    load_fn_params,
    load_params,
    parametrize,
    supported_drones,
    supported_dynamics,
)
from crazyflow.dynamics.first_principles import dynamics as _first_principles_dynamics
from crazyflow.dynamics.so_rpy import dynamics as _so_rpy_dynamics
from crazyflow.dynamics.so_rpy_rotor import dynamics as _so_rpy_rotor_dynamics
from crazyflow.dynamics.so_rpy_rotor_drag import dynamics as _so_rpy_rotor_drag_dynamics

__all__ = [
    "parametrize",
    "load_params",
    "load_fn_params",
    "available_dynamics",
    "dynamics_features",
    "supported_drones",
    "supported_dynamics",
    "Dynamics",
]


available_dynamics: dict[Dynamics, Callable] = {
    Dynamics.first_principles: _first_principles_dynamics,
    Dynamics.so_rpy: _so_rpy_dynamics,
    Dynamics.so_rpy_rotor: _so_rpy_rotor_dynamics,
    Dynamics.so_rpy_rotor_drag: _so_rpy_rotor_drag_dynamics,
}
"""Unparametrized dynamics functions keyed by [Dynamics][crazyflow.dynamics.Dynamics] mode."""


def dynamics_features(dynamics: Callable) -> dict[str, bool]:
    """Return the feature flags declared by a dynamics function.

    Feature flags are set by the [supports][crazyflow.dynamics.core.supports] decorator on each
    dynamics function and describe which optional inputs the dynamics accepts.

    Args:
        dynamics: A dynamics function, or a ``functools.partial`` wrapping one (as
            returned by [parametrize][crazyflow.dynamics.parametrize]).

    Returns:
        A dict of feature names to booleans. Currently contains:
            - ``"rotor_dynamics"``: ``True`` if the dynamics accepts and integrates
              ``rotor_vel``, ``False`` if passing ``rotor_vel`` raises a
              ``ValueError``.

    Example:
    ```python
    from crazyflow.dynamics import dynamics_features
    from crazyflow.dynamics.first_principles import dynamics

    dynamics_features(dynamics)  # {'rotor_dynamics': True}
    ```
    """
    if isinstance(dynamics, partial):
        return dynamics_features(dynamics.func)
    return getattr(dynamics, "__dynamics_features__")
