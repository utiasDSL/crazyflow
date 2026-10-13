# Gymnasium Environments

Crazyflow ships a set of [Gymnasium](https://gymnasium.farama.org/) vectorized environments built on top of `Sim`. They follow the standard `VectorEnv` interface and are suitable for training RL agents with frameworks such as Stable Baselines3, CleanRL, or custom JAX trainers.

## Available environments

| Class | Task | Observation | Action |
|---|---|---|---|
| `DroneEnv` | Base class (no reward) | pos, quat, vel, ang_vel | attitude or force/torque |
| `ReachPosEnv` | Reach a target position | pos, quat, vel, ang_vel, target | attitude |
| `ReachVelEnv` | Match a target velocity | vel, ang_vel, target_vel | attitude |
| `LandingEnv` | Land safely | pos, quat, vel, ang_vel | attitude |
| `FigureEightEnv` | Follow a figure-8 trajectory | pos, quat, vel, ang_vel, phase | attitude |

All environments run `num_envs` parallel instances backed by a single `Sim` with `n_worlds=num_envs`.

## Basic usage

```python
import gymnasium
import crazyflow.envs  # noqa: F401 - registers the environments

env = gymnasium.make_vec("DroneFigureEightTrajectory-v0", num_envs=16)
obs, info = env.reset()

for _ in range(500):
    action = env.action_space.sample()  # random policy for illustration
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

## Constructor arguments

All environments accept these common arguments:

| Argument | Default | Description |
|---|---|---|
| `num_envs` | 1 | Number of parallel environments |
| `max_episode_time` | 10.0 | Episode length before truncation, seconds |
| `dynamics` | `Dynamics.so_rpy` | Dynamics |
| `drone` | `"cf2x_L250"` | Drone configuration |
| `freq` | 500 | Dynamics frequency, Hz |
| `device` | `"cpu"` | `"cpu"` or `"gpu"` |
| `reset_randomization` | `None` | Optional `(SimData, SimData, mask) → SimData` function applied at reset (base `DroneEnv` only) |

## Action normalization

`NormalizeActionsWrapper` rescales the action space to `[-1, 1]`, which simplifies policy learning:

```python
import gymnasium
import crazyflow.envs  # noqa: F401 - registers the environments
from crazyflow.envs.norm_actions_wrapper import NormalizeActions

env = NormalizeActions(gymnasium.make_vec("DroneFigureEightTrajectory-v0", num_envs=32))
obs, info = env.reset()
action = env.action_space.sample()  # in [-1, 1]^4
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

## Reset randomization

Pass a `reset_randomization` callable to vary initial conditions between episodes. The function receives `SimData`, the default `SimData`, and a boolean mask selecting the environments being reset, and must return updated `SimData`:

```python
import jax
from crazyflow.envs.drone_env import DroneEnv
from crazyflow.sim.data import SimData
from crazyflow.utils import leaf_replace


def randomize(data: SimData, default_data: SimData, mask: jax.Array | None) -> SimData:
    key, subkey = jax.random.split(data.core.rng_key)
    data = data.replace(core=data.core.replace(rng_key=key))  # Make sure to update the rng_key
    noise = jax.random.normal(subkey, data.states.pos.shape) * 0.05
    states = leaf_replace(data.states, mask, pos=data.states.pos + noise)
    return data.replace(states=states)


env = DroneEnv(num_envs=64, reset_randomization=randomize)
env.close()
```

## Next steps

- [Examples](../examples/environments.md) — figure-8 and RL training examples
- [Functional API](functional-api.md) — building fully jittable training loops with `jax.lax.scan`
