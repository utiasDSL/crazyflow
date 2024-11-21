from flax.struct import dataclass
from jax import Array


@dataclass
class SimState:
    pos: Array
    quat: Array
    vel: Array
    ang_vel: Array
    rpy_rates: Array


@dataclass
class SimControls:
    state: Array
    attitude: Array
    thrust: Array
    rpms: Array
    rpy_err_i: Array
    pos_err_i: Array
    last_rpy: Array


@dataclass
class SimParams:
    mass: Array
    J: Array
    J_INV: Array
