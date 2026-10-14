from enum import StrEnum

from crazyflow.dynamics import Dynamics


class CollisionType(StrEnum):
    """Collision geometry of the drones, selected with ``select_collision_type``."""

    sphere = "sphere"
    box = "box"
    hull = "hull"
    default = sphere


from crazyflow.sim.sim import Sim  # noqa: E402  (sim.py imports CollisionType from this package)

__all__ = ["Sim", "Dynamics", "CollisionType"]
