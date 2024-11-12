class CrazyflowError(Exception):
    """Base class for all Crazyflow errors."""


class ConfigError(CrazyflowError):
    """Error raised when the configuration is invalid."""
