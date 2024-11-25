from gymnasium.envs.registration import register

register(
    id="crazyflow_env/CrazyflowVectorEnvReachGoal-v0",
    vector_entry_point="crazyflow.crazyflow_env.envs:CrazyflowVectorEnvReachGoal",
)

register(
    id="crazyflow_env/CrazyflowVectorBaseEnv-v0",
    vector_entry_point="crazyflow.crazyflow_env.envs:CrazyflowVectorBaseEnv",
)

register(
    id="crazyflow_env/CrazyflowVectorEnvTargetVelocity-v0",
    vector_entry_point="crazyflow.crazyflow_env.envs:CrazyflowVectorEnvTargetVelocity",
)
