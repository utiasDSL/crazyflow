from gymnasium.envs.registration import register

from crazyflow.gymnasium_envs.crazyflow import CrazyflowEnvReachGoal, CrazyflowEnvTargetVelocity

__all__ = ["CrazyflowEnvReachGoal", "CrazyflowEnvTargetVelocity"]

register(
    id="DroneReachPos-v0",
    vector_entry_point="crazyflow.gymnasium_envs.crazyflow:CrazyflowEnvReachGoal",
)

register(
    id="DroneReachVel-v0",
    vector_entry_point="crazyflow.gymnasium_envs.crazyflow:CrazyflowEnvTargetVelocity",
)
