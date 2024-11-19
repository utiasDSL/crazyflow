from gymnasium.envs.registration import register

register(
    id="crazyflow_env/CrazyflowVectorEnv-v0",
    entry_point="crazyflow.crazyflow_env.envs:CrazyflowVectorEnv",
    vector_entry_point="crazyflow.crazyflow_env.envs:CrazyflowVectorEnv",
)
