# Functional API

The object-oriented API is convenient for scripting, but it relies on Python-level mutations to `sim.data`. JAX's tracing mechanism used inside `jax.jit`, `jax.grad`, and `jax.lax.scan`, cannot see through Python-level side effects. This means that as soon as you try to compile a loop or differentiate through a trajectory using the OO methods, JAX either raises an error or silently produces incorrect results because it cannot build a computation graph from Python mutations.

The functional API addresses this by expressing the same operations as pure functions that take `SimData` and return updated `SimData`. There is no hidden state, so JAX can trace, compile, and differentiate through arbitrary compositions of these functions.

```python
import jax
import jax.numpy as jnp
import crazyflow.sim.functional as F
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(n_worlds=1, n_drones=1, control=Control.attitude, freq=500)
sim.reset()

data, default_data = sim.data, sim.default_data
step, reset = sim.build_step_fn(), sim.build_reset_fn()

cmd = jnp.zeros((1, 1, 4), dtype=jnp.float32)
cmd = cmd.at[..., 3].set(float(data.params.mass[0]) * 9.81)


@jax.jit
def run(data, cmd):
    data = F.attitude_control(data, cmd)
    data = step(data, 10)
    return data


data = run(data, cmd)
assert data.states.pos.shape == (1, 1, 3)
```

`F.attitude_control`, `step`, and `reset` are all pure functions that take `SimData` and return `SimData`. They can be composed freely and passed to `jax.grad`, `jax.vmap`, or `jax.lax.scan`.

## `build_step_fn` and `build_reset_fn`

These two methods return compiled, purely functional step and reset functions. They can also be used to recreate those functions after the [pipelines](pipelines.md) have been modified.

- **`sim.build_step_fn()`** returns a `jax.jit`-compiled function with signature `(SimData, n_steps: int) -> SimData`.
- **`sim.build_reset_fn()`** returns a `jax.jit`-compiled function with signature `(data: SimData, default_data: SimData, mask: Array | None) -> SimData`.

The typical setup is:

```python
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(n_worlds=4, n_drones=1, control=Control.attitude)
sim.reset()

data, default_data = sim.data, sim.default_data
step, reset = sim.build_step_fn(), sim.build_reset_fn()
```

From this point, `data` is a plain JAX pytree and `step` and `reset` are compiled functions with no reference to the Python `Sim` object.

## Purely functional controller functions

`crazyflow.sim.functional` mirrors all four `Sim` control methods as pure functions:

```python
import crazyflow.sim.functional as F
```

| Function | Description |
|---|---|
| `F.state_control(data, controls)` | Stage a state command |
| `F.attitude_control(data, controls)` | Stage an attitude command |
| `F.force_torque_control(data, controls)` | Stage a force/torque command |
| `F.rotor_vel_control(data, controls)` | Stage rotor velocity commands |
| `F.controllable(data)` | Boolean mask — which worlds may update their controller this step |

These are exactly the same operations as the OO methods, but `SimData` is passed explicitly and returned rather than mutating `sim.data` in place. All functions return a new `SimData`; the original is never modified.

## Running simulation inside a compiled function

Any code that calls `step` or the functional control functions can be wrapped in `@jax.jit`. The first call compiles; subsequent calls with the same array shapes are instant.

```python
import jax
import jax.numpy as jnp
import crazyflow.sim.functional as F
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(n_worlds=1, n_drones=1, control=Control.attitude, freq=500)
sim.reset()

data, default_data = sim.data, sim.default_data
step, reset = sim.build_step_fn(), sim.build_reset_fn()

cmd = jnp.zeros((1, 1, 4), dtype=jnp.float32)
cmd = cmd.at[..., 3].set(float(data.params.mass[0]) * 9.81)


@jax.jit
def simulate_episode(data, default_data, cmd):
    data = reset(data, default_data)
    data = F.attitude_control(data, cmd)
    data = step(data, 10)
    return data


data = simulate_episode(data, default_data, cmd)
pos = data.states.pos[0, 0]
```

Calling `step(data, n_steps)` is more efficient than a Python loop over `step(data, 1)` because the entire sequence compiles into a single XLA program.

## Differentiating through the dynamics

Because every operation is a pure JAX function, `jax.grad` can differentiate through the entire dynamics pipeline. Here we start the drone 2 m above the floor and compute the gradient of a height-tracking loss with respect to the attitude command.

```python
import jax
import jax.numpy as jnp
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(control=Control.attitude, attitude_freq=50)
sim.reset()

# Place the drone 2 m above the floor, above the 1 m target
data = sim.data.replace(states=sim.data.states.replace(pos=sim.data.states.pos.at[..., 2].set(2.0)))
step = sim.build_step_fn()


def loss(cmd, data):
    data = data.replace(
        controls=data.controls.replace(attitude=data.controls.attitude.replace(staged_cmd=cmd))
    )
    data = step(data, 10)
    return (data.states.pos[0, 0, 2] - 1.0) ** 2  # squared error to 1 m


grad_fn = jax.jit(jax.grad(loss))

cmd = jnp.zeros((1, 1, 4), dtype=jnp.float32)
cmd = cmd.at[..., 3].set(float(data.params.mass[0]) * 9.81)

grad = grad_fn(cmd, data)
# Drone is above the target: reducing thrust lowers it toward 1 m.
# The gradient is positive — descent (less thrust) is the correct direction.
assert float(grad[0, 0, 3]) > 0.0
```

## Working with SimData directly

`SimData` supports `.replace()` at every level of nesting. This is the standard way to set initial conditions or inject custom state before a rollout:

```python
import jax.numpy as jnp
from crazyflow.sim import Sim

sim = Sim(n_worlds=4, n_drones=1)
sim.reset()

# Set all 4 worlds to different starting heights
new_pos = sim.data.states.pos.at[:, 0, 2].set(jnp.array([0.2, 0.4, 0.6, 0.8]))
sim.data = sim.data.replace(states=sim.data.states.replace(pos=new_pos))
```

## Next steps

- [Examples](../examples/index.md) — complete runnable scripts for gradient descent, batched simulation, and RL
- [Pipelines](pipelines.md) — customising the pipeline that `build_step_fn` compiles
