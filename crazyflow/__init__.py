import os
import sys

# SciPy array API check. We use the most recent array API features, which require the
# SCIPY_ARRAY_API environment variable to be set to "1". This flag MUST be set before importing
# scipy, because scipy's C extensions cannot be unloaded once they have been imported. Therefore, we
# have to error out if the flag is not set. Otherwise, we immediately import scipy to ensure that no
# other package sets the flag to a different value before importing scipy.

if "scipy" in sys.modules and os.environ.get("SCIPY_ARRAY_API") != "1":
    msg = """scipy has already been imported and the 'SCIPY_ARRAY_API' environment variable has not
    been set. Please restart your Python session and set SCIPY_ARRAY_API="1" before importing any
    packages that depend on scipy, or import this package first to automatically set the flag."""
    raise RuntimeError(msg)

os.environ["SCIPY_ARRAY_API"] = "1"
import scipy  # noqa: F401, ensure scipy uses array API features

import crazyflow.envs  # noqa: F401, ensure gymnasium envs are registered
from crazyflow.control import Control
from crazyflow.sim import Physics, Sim

__all__ = ["Sim", "Physics", "Control"]
__version__ = "0.0.2"
