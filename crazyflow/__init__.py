import os

os.environ["SCIPY_ARRAY_API"] = "1"

import crazyflow.envs  # noqa: F401, ensure gymnasium envs are registered
from crazyflow.control import Control
from crazyflow.sim import Physics, Sim

__all__ = ["Sim", "Physics", "Control"]
__version__ = "0.0.2"
