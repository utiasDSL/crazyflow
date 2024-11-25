class CrazyflowError(Exception):
    """Base class for all Crazyflow errors."""


class ConfigError(CrazyflowError):
    """Error raised when the configuration is invalid."""


class NotInitializedError(CrazyflowError):
    """Error raised when a component is not initialized."""
