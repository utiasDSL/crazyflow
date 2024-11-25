import jax.numpy as jnp
from jax import Array

# Physical constants
GRAVITY: float = 9.81

# Drone constants
ARM_LEN: float = 0.46
MIX_MATRIX: Array = jnp.array([[-0.5, -0.5, -1], [-0.5, 0.5, 1], [0.5, 0.5, -1], [0.5, -0.5, 1]])
SIGN_MIX_MATRIX: Array = jnp.sign(MIX_MATRIX)
MASS: float = 0.027
