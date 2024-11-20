import gymnasium
import crazyflow.crazyflow_env.envs

# TODO pass additional parameters for setting up the sim
envs = gymnasium.make_vec("crazyflow_env/CrazyflowVectorEnv-v0", num_envs=4, num_drones_per_env=3)

obs, info = envs.reset(seed=42)
_ = envs.action_space.seed(42)

# Step through the environment with randomly sampled actions
for _ in range(10_000):
    action = envs.action_space.sample()
    observation, reward, terminated, truncated, info = envs.step(action)
    # TODO: Add your code here to process the observation, reward, done, and info variables
    if terminated or truncated:
        print("Episode terminated")
        break

envs.close()
