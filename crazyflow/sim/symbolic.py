from __future__ import annotations

from typing import TYPE_CHECKING

from drone_models import parametrize
from drone_models.first_principles import symbolic_dynamics as first_principles_symbolic_dynamics
from drone_models.so_rpy import symbolic_dynamics as so_rpy_symbolic_dynamics

from crazyflow.sim.physics import Physics
from crazyflow.sim.structs import Control

if TYPE_CHECKING:
    import casadi as cs

    from crazyflow.sim import Sim


def symbolic_from_sim(
    sim: Sim, model_rotor_vel: bool = False, model_dist_f: bool = False, model_dist_t: bool = False
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """Create a symbolic model from a simulation object.

    Args:
        sim: The simulation object.
        model_rotor_vel: Flag to model the rotor velocity.
        model_dist_f: Flag to model the distributed force.
        model_dist_t: Flag to model the distributed torque.

    Returns:
        The four symbolic expressions for X_dot, X, U, Y.
    """
    if sim.control != Control.attitude:
        raise ValueError("Symbolic model dynamics only support attitude control")
    match sim.physics:
        case Physics.first_principles:
            return parametrize(first_principles_symbolic_dynamics, sim.drone_model)(
                model_rotor_vel=model_rotor_vel,
                model_dist_f=model_dist_f,
                model_dist_t=model_dist_t,
            )
        case Physics.so_rpy:
            return parametrize(so_rpy_symbolic_dynamics, sim.drone_model)(
                model_rotor_vel=model_rotor_vel,
                model_dist_f=model_dist_f,
                model_dist_t=model_dist_t,
            )
        case _:
            raise ValueError(f"Physics mode {sim.physics} not supported")
