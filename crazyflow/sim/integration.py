from enum import Enum


class Integrator(str, Enum):
    euler = "euler"
    # TODO: Implement rk4
    default = euler  # TODO: Replace with rk4
