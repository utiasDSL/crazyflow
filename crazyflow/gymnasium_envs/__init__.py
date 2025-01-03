from gymnasium.envs.registration import register

from crazyflow.gymnasium_envs.crazyflow import (
    CrazyflowEnvFigureEightTrajectory,
    CrazyflowEnvLanding,
    CrazyflowEnvReachGoal,
    CrazyflowEnvTargetVelocity,
    CrazyflowRL,
)

__all__ = [
    "CrazyflowEnvReachGoal",
    "CrazyflowEnvTargetVelocity",
    "CrazyflowEnvLanding",
    "CrazyflowRL",
    "CrazyflowEnvFigureEightTrajectory",
]

register(
    id="DroneReachPos-v0",
    vector_entry_point="crazyflow.gymnasium_envs.crazyflow:CrazyflowEnvReachGoal",
)

register(
    id="DroneReachVel-v0",
    vector_entry_point="crazyflow.gymnasium_envs.crazyflow:CrazyflowEnvTargetVelocity",
)

register(
    id="DroneLanding-v0",
    vector_entry_point="crazyflow.gymnasium_envs.crazyflow:CrazyflowEnvLanding",
)

register(
    id="DroneFigureEightTrajectory-v0",
    vector_entry_point="crazyflow.gymnasium_envs.crazyflow:CrazyflowEnvFigureEightTrajectory",
)
