import numpy as np
from numpy.typing import NDArray

# TODO: Remove all constants
# Physical constants
GRAVITY: float = 9.81
# Crazyflie 2.1 mass as measured in the lab with battery included
MASS: float = 0.033
J: NDArray = np.array([[2.3951e-5, 0, 0], [0, 2.3951e-5, 0], [0, 0, 3.2347e-5]])
