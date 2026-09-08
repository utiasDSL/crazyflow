# The world axis

The simulation is batched over worlds, so most arrays in `sim.data` carry a leading `n_worlds` axis. Resetting selected worlds and distributing worlds across devices both need to know which arrays those are. Here we explain how that is declared and what depends on it.

## How it is declared

A world axis is an ordinary axis, so no shape on its own tells you whether it indexes worlds. With three worlds, `params.drag_matrix` of shape `(3, 3)` and `params.gravity_vec` of shape `(3,)` both look world-batched while neither is.

Each field therefore declares its number of dimensions without batch axes with the `CORE_NDIM_KEY` metadata. `states.pos` is a vector, so its core ndim is 1, and `params.drag_matrix` is a matrix, so its core ndim is 2. An array carries a world axis exactly when its own `ndim` exceeds the core ndim of its field:

```python
from crazyflow.sim import Sim
from crazyflow.utils import world_mask

sim = Sim(n_worlds=3)
mask = world_mask(sim.data)

assert mask.states.pos  # (n_worlds, n_drones, 3), indexed by world
assert not mask.params.mass  # (1,), shared by all worlds
assert not mask.params.gravity_vec  # (3,), shared by all worlds
assert not mask.params.drag_matrix  # (3, 3), shared by all worlds
```

`world_mask` returns a pytree of booleans matching the data, with one flag per array. Arrays whose field declares no core ndim are shared by all worlds.

## Resets

`sim.reset()` restores the whole simulation, both the world-indexed and the shared arrays, with the rng key as the only exception.

`sim.reset(mask)` is the constrained form. It selects worlds along the world axis, so it can only restore arrays that have one. A shared array cannot be restored for some worlds and not others, so a masked reset leaves it untouched:

```{ .python continuation }
import jax.numpy as jnp

gravity = jnp.array([1.0, 2.0, 3.0])
sim.data = sim.data.replace(params=sim.data.params.replace(gravity_vec=gravity))
sim.reset(jnp.array([True, False, False]))

assert jnp.array_equal(sim.data.params.gravity_vec, gravity)  # Shared, so left alone
assert jnp.all(sim.data.states.pos[0] == 0.0)  # World 0 was reset
```

To keep a change to a shared parameter across masked resets, write it into `sim.default_data` as well by calling `sim.build_default_data()`. See [Stepping and resetting](oo-api.md#stepping-and-resetting) for the reset API.

## Sharding

Distributing the simulation partitions the world axis of the arrays that have one and replicates the rest, using the same declarations. See [Sharding](sharding.md) for the mesh and placement API.

## Randomizing a shared parameter

A parameter that is shared by default gains a world axis the moment you randomize it per world, and its `ndim` then exceeds the declared core ndim:

```{ .python continuation }
n = sim.n_worlds
scale = jnp.arange(1, n + 1)[:, None, None, None]
drag = jnp.broadcast_to(sim.data.params.drag_matrix, (n, 1, 3, 3)) * scale
sim.data = sim.data.replace(params=sim.data.params.replace(drag_matrix=drag))
sim.build_default_data()  # Keep the randomized shape across full resets

assert world_mask(sim.data).params.drag_matrix
```

From there it behaves like any other world-indexed array. A masked reset restores it for the masked worlds, and sharding partitions it instead of replicating it. Setting it back to a `(3, 3)` array makes it shared again.

## Your own state

Anything you add to `sim.data.plugins` follows the same rule. A `dict` has no fields, so bare arrays are shared, which means a masked reset leaves them alone and sharding replicates them. If you need per-world state, declare it on a struct:

```python
import flax.struct
from jax import Array

from crazyflow.utils import CORE_NDIM_KEY


@flax.struct.dataclass
class PluginState:
    buffer: Array = flax.struct.field(metadata={CORE_NDIM_KEY: 1})
    lookup: Array = None
```

If you store this under `sim.data.plugins`, a `buffer` of shape `(n_worlds, 3)` exceeds its core ndim of 1, so it is restored for the masked worlds and partitioned when the simulation is sharded. A `lookup` of any shape declares no core ndim, so it is only restored by a full reset and replicated across devices.

## Next steps

- [Sharding](sharding.md) — distributing the worlds over several devices
- [Pipelines](pipelines.md) — customising what runs on a step and on a reset
