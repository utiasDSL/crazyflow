import numpy as np
from numpy.typing import NDArray

# Physical constants
GRAVITY: float = 9.81

# Drone constants
ARM_LEN: float = 0.0325 * np.sqrt(2)
MIX_MATRIX: NDArray = np.array([[-0.5, -0.5, -1], [-0.5, 0.5, 1], [0.5, 0.5, -1], [0.5, -0.5, 1]])
SIGN_MIX_MATRIX: NDArray = np.sign(MIX_MATRIX)
MASS: float = 0.027
J: NDArray = np.array([[2.3951e-5, 0, 0], [0, 2.3951e-5, 0], [0, 0, 3.2347e-5]])
J_INV: NDArray = np.linalg.inv(J)
