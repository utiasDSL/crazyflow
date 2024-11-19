import gymnasium
import crazyflow.crazyflow_env.envs

# TODO pass additional parameters for setting up the sim
env = gymnasium.make_vec("crazyflow_env/CrazyflowVectorEnv-v0", num_envs=4, num_drones_per_env=3)
pass
