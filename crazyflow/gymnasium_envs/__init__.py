from gymnasium.envs.registration import register

register(
    id="CrazyflowEnvReachGoal-v0",
    vector_entry_point="crazyflow.gymnasium_envs:CrazyflowEnvReachGoal",
)

register(
    id="CrazyflowEnvTargetVelocity-v0",
    vector_entry_point="crazyflow.gymnasium_envs:CrazyflowEnvTargetVelocity",
)
