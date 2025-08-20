import numpy as np
from numpy.typing import NDArray

# Physical constants
GRAVITY: float = 9.81

# Drone constants
ARM_LEN: float = 0.0325 * np.sqrt(2)
# fmt: off
MIX_MATRIX: NDArray = np.array([[-0.5, -0.5, -1],
                                [-0.5,  0.5,  1],
                                [ 0.5,  0.5, -1],
                                [ 0.5, -0.5,  1]])
# fmt: on
SIGN_MIX_MATRIX: NDArray = np.sign(MIX_MATRIX)
# Crazyflie 2.1 mass as measured in the lab with battery included
MASS: float = 0.033
J: NDArray = np.array([[2.3951e-5, 0, 0], [0, 2.3951e-5, 0], [0, 0, 3.2347e-5]])
J_INV: NDArray = np.linalg.inv(J)
KF: float = 8.701227710666256e-10
KM: float = 7.94e-12
