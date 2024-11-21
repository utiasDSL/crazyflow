from flax.struct import dataclass, field
from jax import Array


@dataclass
class SimState:
    step: Array
    pos: Array
    quat: Array
    vel: Array
    ang_vel: Array
    rpy_rates: Array
    device: str = field(default_factory=lambda: "cpu", pytree_node=False)


@dataclass
class SimControls:
    state: Array
    attitude: Array
    thrust: Array
    rpms: Array
    rpy_err_i: Array
    pos_err_i: Array
    last_rpy: Array
    device: str = field(default_factory=lambda: "cpu", pytree_node=False)


@dataclass
class SimParams:
    mass: Array
    J: Array
    J_INV: Array
    device: str = field(default_factory=lambda: "cpu", pytree_node=False)
