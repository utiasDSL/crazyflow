# Pipelines

Crazyflow has two pipelines, one for stepping and one for resetting. Each is an ordered dictionary of pure JAX functions that transform `SimData`. Stages are keyed by a unique string name so they can be addressed directly without relying on positional indices.

`crazyflow.sim.pipeline` provides helper functions for safely modifying a pipeline:

| Function | Description |
|---|---|
| `append_fn(pipeline, fn, name=None)` | Add a stage at the end |
| `prepend_fn(pipeline, fn, name=None)` | Add a stage at the beginning |
| `insert_fn_before(pipeline, anchor, fn, name=None)` | Insert before a named stage |
| `insert_fn_after(pipeline, anchor, fn, name=None)` | Insert after a named stage |
| `replace_fn(pipeline, fn, name)` | Swap the function of an existing stage |
| `remove_fn(pipeline, name)` | Remove a stage by name |

All helpers raise `KeyError` on duplicate or missing names. Stage names default to `fn.__name__`. Pass an explicit `name` for anonymous callables such as `functools.partial` objects.

Both pipelines are constructed at `Sim` initialisation and compiled into a single `jax.jit`-cached function by `build_step_fn()` / `build_reset_fn()`. Modify the pipeline and then call the corresponding build function to recompile.

## The step pipeline

`sim.step_pipeline` contains multiple stages by default:

1. **Control functions** — convert the staged command through the control hierarchy (state → attitude → force/torque → rotor velocities, depending on the selected mode)
2. **Integrator** (`integration`) — advance the ODE one dynamics step (Euler, RK4, or symplectic Euler)
3. **Rotor clip** (`clip_rotor_vel`) — clip the motor speeds (first principles) or the collective thrust (fitted models) to the physical limits of the motors
4. **Step counter** (`increment_steps`) — increment `data.core.steps`
5. **Floor clip** (`clip_floor_pos`) — prevent drones from passing through the floor

```pycon
>>> from crazyflow.sim import Sim
>>> sim = Sim()
>>> print(tuple(sim.step_pipeline.keys()))
('attitude_controller', 'force_torque_controller', 'integration', 'clip_rotor_vel', 'increment_steps', 'clip_floor_pos')

```

## The reset pipeline

`sim.reset_pipeline` holds a single `reset` stage that restores `SimData` to the default state. Every stage appended after it runs in order on the restored data. Each reset stage has the signature `(data: SimData, default_data: SimData, mask: Array | None) -> SimData`. The `default_data` argument holds the freshly-restored default state, which is useful for selectively reverting fields.

Populate `sim.reset_pipeline` to add episode-level randomization without modifying the default state.

## Modifying the step pipeline

Stages are addressed by name. Use `insert_fn_before` / `insert_fn_after` to place a function relative to an existing stage, `append_fn` to add it at the end, and `replace_fn` to swap a stage's implementation. New stages are named after the function's `__name__` unless an explicit name is given; names must be unique within a pipeline.

```python
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.pipeline import insert_fn_before

sim = Sim()


def disturbance_fn(data: SimData) -> SimData:
    return data.replace(states=data.states.replace(vel=data.states.vel + 1e-5))


insert_fn_before(sim.step_pipeline, "integration", disturbance_fn)
sim.build_step_fn()  # recompile
```

!!! warning
    Always call `sim.build_step_fn()` after modifying `sim.step_pipeline`. Without it, `sim.step()` still runs the previously compiled kernel and silently ignores your changes.

To see how to modify the step pipeline with a stochastic disturbance, see the [Disturbance injection example](../examples/index.md#disturbance-injection).

## Modifying the reset pipeline

Add a function to the reset pipeline to vary initial conditions between episodes. The function receives the freshly-restored `data` and an optional `mask` of worlds that were reset.

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.pipeline import append_fn


def randomize_initial_pos(data: SimData, default_data: SimData, mask: Array | None) -> SimData:
    key, subkey = jax.random.split(data.core.rng_key)
    noise = jax.random.normal(subkey, data.states.pos.shape) * 0.1  # ±10 cm
    return data.replace(
        states=data.states.replace(pos=data.states.pos + noise), core=data.core.replace(rng_key=key)
    )


sim = Sim(n_worlds=16)
append_fn(sim.reset_pipeline, randomize_initial_pos)
sim.build_reset_fn()  # recompile
sim.reset()
# Each of the 16 worlds now starts at a slightly different position
```

```{ .python continuation }
def randomize_vel(data: SimData, default_data: SimData, mask: Array | None) -> SimData:
    key, subkey = jax.random.split(data.core.rng_key)
    noise = jax.random.normal(subkey, data.states.vel.shape) * 0.05
    return data.replace(
        states=data.states.replace(vel=data.states.vel + noise),
        core=data.core.replace(rng_key=key),
    )

def log_reset(data: SimData, default_data: SimData, mask: Array | None) -> SimData:
    return data  # a pure pass-through stage, e.g. a hook for metrics

for fn in (randomize_vel, log_reset):
    append_fn(sim.reset_pipeline, fn)
sim.build_reset_fn()

mask = np.zeros(16, dtype=bool)
mask[0] = True  # Only randomize the first world
sim.reset(mask=mask)  # The mask is optional; omit it to reset and randomize all worlds
```

Reset stages run in tuple order, with each stage receiving the output of the previous one. The mask ensures that parameter updates apply only to worlds selected by `sim.reset()`.

## Removing a stage

Remove any stage by name. A common case is removing the floor clip when computing gradients through a trajectory that starts high above the ground:

```python
from crazyflow.sim import Sim
from crazyflow.sim.pipeline import remove_fn

sim = Sim()
remove_fn(sim.step_pipeline, "clip_floor_pos")
sim.build_step_fn()
```

!!! note
    The rotor limits are enforced by the `clip_rotor_vel` stage, which clips the rotor state after each integration step. Higher order integration methods like RK4 evaluate the dynamics at intermediate, unclipped states, so the `rotor_vel` seen by the model can transiently exceed the limits within a single step.

## Writing a custom stage

A step pipeline function must have the signature `(SimData) -> SimData`. A reset pipeline function must have the signature `(SimData, SimData, Array | None) -> SimData` where the second argument is the default (freshly-restored) data. Both must be pure JAX functions with no Python-level side effects, so they can be traced and compiled.

```python
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.pipeline import append_fn


def my_step_stage(data: SimData) -> SimData:
    # JAX operations only — return updated data
    return data.replace(states=data.states.replace(pos=data.states.pos + 0.01))


sim = Sim()
append_fn(sim.step_pipeline, my_step_stage)
sim.build_step_fn()
```

## Randomization and disturbances

The step and reset pipelines enable efficient parameter randomization and disturbance injection. Randomizations should be applied in the reset pipeline, where they run once per episode. Disturbances go into the step pipeline, where they are applied on every dynamics step.

### Randomizing parameters per world and drone

All parameters are shared by all worlds and drones by default. However, they also accept leading batch axes `(n_worlds, n_drones)` to use individual parameters per world and per drone. Parameters of the first-principles model that belong to a motor carry a motor axis of size 1 when shared, so they can also be randomized per motor: the arm length `L` and the propeller inertia `prop_inertia` go from `(1,)` to `(n_worlds, n_drones, 4)`, and the thrust curve `rpm2thrust`, the torque curve `rpm2torque` and the rotor dynamics coefficients `rotor_dyn_coef` go from `(1, K)` to `(n_worlds, n_drones, 4, K)`.

Use the `default_data` argument as the base value so that repeated resets do not compound, and `leaf_replace` to only touch the worlds and drones selected by the mask:

```python
import jax
from jax import Array

from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.pipeline import append_fn
from crazyflow.utils import leaf_replace


def randomize_thrust_curve(data: SimData, default_data: SimData, mask: Array | None) -> SimData:
    key, subkey = jax.random.split(data.core.rng_key)
    shape = (data.core.n_worlds, data.core.n_drones, 4, 3)  # One curve per motor
    scale = jax.random.uniform(subkey, shape, minval=0.9, maxval=1.1)  # +-10% per coefficient
    rpm2thrust = default_data.params.rpm2thrust * scale  # (1, 3) -> (n_worlds, n_drones, 4, 3)
    params = leaf_replace(data.params, mask, rpm2thrust=rpm2thrust)
    return data.replace(params=params, core=data.core.replace(rng_key=key))


sim = Sim(n_worlds=4)
append_fn(sim.reset_pipeline, randomize_thrust_curve)
sim.build_reset_fn()
sim.reset()
assert sim.data.params.rpm2thrust.shape == (4, 1, 4, 3)
```

!!! warning "Split and update the rng key"
    JAX random keys are stateless: drawing from the same key always produces the same numbers. Every stage that samples must split `data.core.rng_key`, draw from the new subkey, and write the other half back with `data.core.replace(rng_key=key)`. The reset stage restores everything except the rng key, so the key keeps advancing across resets.

A masked reset first restores the parameters of the selected worlds to their defaults and then runs the randomization on them, while the other worlds keep their values. See the [domain randomization example](../examples/index.md#domain-randomization) for a randomization of every parameter of the first-principles model, and [The world axis](world-axis.md) for how per-world arrays are told apart from shared ones.

!!! note "Recompilation"
    The compiled step and reset functions are specialized on the shapes in `SimData`. The first randomization that turns a shared `(1, 3)` parameter into a `(n_worlds, n_drones, 4, 3)` array changes those shapes, so the following `sim.step()` and `sim.reset()` calls recompile once. The shapes are stable afterwards, unless something restores the shared shape again, e.g. a reset with a pipeline that no longer randomizes the parameter, which triggers another recompilation.

To avoid the recompilation, or to keep the batched shape across changes of the reset pipeline, broadcast the parameters once and bake the shape into the default data:

```{ .python continuation }
import jax.numpy as jnp

sim = Sim(n_worlds=4)
params = sim.data.params
rpm2thrust = jnp.broadcast_to(params.rpm2thrust, (4, 1, 4, 3))  # Same values, batched shape
sim.data = sim.data.replace(params=params.replace(rpm2thrust=rpm2thrust))
sim.build_default_data()  # Resets now restore the batched shape instead of the shared one
```

### Injecting disturbances

Disturbances act on every dynamics tick, so they must be added to the step pipeline instead. The dynamics read an external force `data.states.force` and torque `data.states.torque` acting on the center of mass in the world frame. Both have shape `(n_worlds, n_drones, 3)` and are zero by default, so a stage that fills them applies a disturbance per world and per drone:

```python
import jax

from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.pipeline import insert_fn_before


def wind(data: SimData) -> SimData:
    key, subkey = jax.random.split(data.core.rng_key)
    force = jax.random.normal(subkey, data.states.force.shape) * 0.02  # N, world frame
    states = data.states.replace(force=force)
    return data.replace(states=states, core=data.core.replace(rng_key=key))


sim = Sim(n_worlds=4)
insert_fn_before(sim.step_pipeline, "integration", wind)
sim.build_step_fn()
sim.step()
```

See the [disturbance injection example](../examples/index.md#disturbance-injection) for a full run comparing disturbed and undisturbed trajectories.


## Next steps

- [Functional API](functional-api.md) — how `build_step_fn` fits into a compiled training loop
- [Examples](../examples/index.md) — disturbance injection and domain randomization scripts
