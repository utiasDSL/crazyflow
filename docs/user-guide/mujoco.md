# MuJoCo Integration

Crazyflow focuses on drone dynamics and controllers. However, we still want to provide rendering and collision checking, and to do that we leverage [MuJoCo](https://mujoco.org/) and its JAX port [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html). We keep an MJX representation of the scene in sync with Crazyflow's dynamics state and invoke MJX functions where needed: collision queries, forward kinematics, and sensor rendering. GUI rendering uses the CPU-side MuJoCo renderer directly.

## MuJoCo and MJX objects

Crazyflow maintains two parallel representations at all times:

| Object | Type | Purpose |
|---|---|---|
| `sim.mj_model` | `mujoco.MjModel` | Reference model, used for GUI rendering |
| `sim.mj_data` | `mujoco.MjData` | Scratch MuJoCo data buffer, only used to initialise MJX |
| `sim.mjx_model` | `mjx.Model` | JAX pytree of the model (static, shared across worlds) |
| `sim.mjx_data` | `mjx.Data` | JAX pytree of the scene state, batched over `n_worlds` |

`mjx_data` does not hold the dynamics state. It holds the scene geometry state (body transforms, contact distances, camera positions), derived from `sim.data` through an explicit sync step whenever rendering or collision queries are needed.

## MJCF and scene construction

The scene is built programmatically from MJCF (MuJoCo's XML format) at `Sim` construction time using the `MjSpec` API. The process is:

1. Load the base scene from `crazyflow/scene.xml` (floor, lighting, and sky).
2. Load the drone MJCF bundled with `crazyflow.drones` (under `crazyflow/drones`).
3. Mark the drone body as mocap. Mocap bodies are kinematically driven by external position and quaternion updates rather than joints, which avoids the O(nv²) cost of computing constraint matrices and saves memory.
4. Attach one copy per drone to a frame in the world body.
5. Compile the spec into `mj_model`, then convert to `mjx_model` and `mjx_data` via `mjx.put_model` and `mjx.put_data`. Vmap `mjx_data` across `n_worlds`.

The spec is accessible as `sim.spec` before compilation, and `sim.mj_model` / `sim.mjx_model` after.

## Fused drone model (`fused_mjx_model`)

Each drone MJCF defines two bodies. `drone` has separate visual meshes for the PCB, motors, propellers, LEDs, and battery. `drone_fused` bakes the visuals into a single mesh. Passing `fused_mjx_model=True` to `Sim` selects the fused body:

```{ .python }
from crazyflow.sim import Sim

sim = Sim(n_worlds=1, n_drones=1, fused_mjx_model=True)
```

The difference between the two is purely visual. The fused body consists of a single geom with one mesh, which shrinks its memory footprint. The cost is visual detail. Use it for large swarms or headless runs, and keep the default for detailed rendering.

## Adding objects to the scene

Custom geometry (gates, obstacles, walls, or any MJCF body) can be added by editing `sim.spec` and calling `sim.build_mjx()`. The new geometry is available for collision and rendering but has no effect on the drone dynamics, which are computed independently in JAX.

```{ .python }
import mujoco
from crazyflow.sim import Sim

sim = Sim(n_worlds=1, n_drones=1)

# Define a box body as an inline XML string (or load from a file)
box_xml = """
<mujoco>
  <worldbody>
    <body name="obstacle">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""
obstacle_spec = mujoco.MjSpec.from_string(box_xml)

# Attach one or more instances to a new frame in the scene
frame = sim.spec.worldbody.add_frame()
for i, pos in enumerate([[1.0, 0.0, 0.5], [2.0, 0.0, 0.5]]):
    body = obstacle_spec.body("obstacle")
    attached = frame.attach_body(body, "", f":{i}")
    attached.pos = pos

# Recompile — closes the viewer if open, rebuilds mj_model and mjx_model/mjx_data
sim.build_mjx()
sim.reset()
```

Loading from a file works identically:

<!-- notest: reads external asset assets/gate.xml -->
```{ .python notest }
import mujoco
gate_spec = mujoco.MjSpec.from_file("assets/gate.xml")
```

For a real-world example, see the drone racing environment in [lsy_drone_racing](https://github.com/learnsyslab/lsy_drone_racing), which loads gate and obstacle specs from MJCF files and attaches them at the configured track positions.

### Setting body positions at runtime

If you mark an attached body as mocap (`attached.mocap = True`), its position can be updated at runtime by writing directly into `sim.mjx_data.mocap_pos` without rebuilding the model. This is how the drone positions themselves are driven.

## Synchronization

The JAX dynamics pipeline writes to `sim.data` but never touches `sim.mjx_data`. `mjx_data` is only needed for collision queries and rendering, which require current body transforms. To avoid computing those on every dynamics step, Crazyflow tracks a `mjx_synced` flag in `sim.data.core`.

After `sim.step()` or `sim.reset()`, `mjx_synced` is set to `False`. The `sim.render()` and `sim.contacts()` methods check the flag; if stale, they call `sync_sim2mjx()` once and set it back to `True`.

`sync_sim2mjx` does three things:

1. Write drone positions and quaternions into `mjx_data.mocap_pos` / `mjx_data.mocap_quat`.
2. `jax.vmap(mjx.kinematics)` to propagate body transforms through the kinematic tree.
3. `jax.vmap(mjx.camlight)` and `jax.vmap(mjx.collision)` for rendering and contact detection respectively.

These run only once per render or contact call, regardless of how many dynamics steps were taken since the last sync.

<!-- notest: requires rendering -->
```{ .python notest }
for i in range(10):
    sim.step(5)                       # JAX dynamics only, mjx_synced = False
    if i % 5 == 0:
        sim.render(mode="rgb_array")  # syncs once: kinematics + camlight + collision
```

## Advanced: the sync flag and avoiding redundant MJX calls

`sync_sim2mjx` runs kinematics, collision detection, and camera transforms in one shot. The `mjx_synced` flag ensures this happens at most once between dynamics steps: once the flag is set, any further calls to `sim.render()` or `sim.contacts()` within the same tick skip the sync entirely and operate on the already-computed MJX state. The flag is only cleared when `sim.data` actually changes, so if the dynamics state has not advanced, the expensive MJX operations are not repeated.

This means the order of calls matters. Grouping all rendering and contact queries together after a step lets them share a single sync:

<!-- notest: requires rendering -->
```{ .python notest }
sim.step(5)
contacts = sim.contacts()     # sync runs here
sim.render(mode="rgb_array")  # flag already set, no second sync
```

Interleaving a step between them forces two syncs:

<!-- notest: requires rendering -->
```{ .python notest }
contacts = sim.contacts()     # sync runs here
sim.step(5)                   # flag cleared
sim.render(mode="rgb_array")  # sync runs again
```

## Advanced: fusing mjx_data into a contact check function

Passing `sim.mjx_data` as an argument to a `@jax.jit`-compiled function is expensive. JAX must flatten the entire pytree at the JIT boundary on every call, and `mjx_data` contains many leaves. For contact checking that runs in a tight loop, this overhead matters.

The solution is to **close over** `mjx_data` rather than pass it as an argument. With `mjx_data` captured in the function closure, JAX treats it as a constant and only flattens it once at compile time. At call time, only the small dynamic state needs to be canonicalized.

The drone racing environment in [lsy_drone_racing](https://github.com/learnsyslab/lsy_drone_racing) uses this pattern to build a contact check function:

```python
from jax import Array

from crazyflow.sim import Sim
from crazyflow.sim.sim import sync_sim2mjx
from crazyflow.sim.data import SimData

sim = Sim(n_worlds=1, n_drones=1)
sim.reset()

_mjx_data = sim.mjx_data  # captured in closure
contact_fn = sim.build_contact_fn()


def check_contacts(sim_data: SimData, obstacle_mocap_pos: Array) -> Array:
    # Update obstacle positions and sync inside JIT
    mjx_data = _mjx_data.replace(mocap_pos=obstacle_mocap_pos)
    _, mjx_data = sync_sim2mjx(sim_data, mjx_data, sim.mjx_model)
    return contact_fn(mjx_data)


in_contact = check_contacts(sim.data, sim.mjx_data.mocap_pos)  # shape (n_worlds, n_drones)
```

`_mjx_data` is fused into the closure and compiled as a constant. Only `sim_data` and the obstacle positions cross the JIT boundary at runtime — a much smaller pytree than passing the full `mjx_data`.

## Next steps

- [Pipelines](pipelines.md) — inserting custom stages into the step and reset pipelines
- [Examples](../examples/rendering.md#cameras-and-rgbd) — FPV camera and RGBD rendering
- [Examples](../examples/contacts.md#contact-queries) — contact detection with box collision geometry
