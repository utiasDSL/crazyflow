from gymnasium.envs.registration import register

register(
    id="crazyflow_env/CrazyflowVectorEnvReachGoal-v0",
    vector_entry_point="crazyflow.crazyflow_env.envs:CrazyflowVectorEnvReachGoal",
)
