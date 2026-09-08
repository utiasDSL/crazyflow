# Visualization

Crazyflow supports onscreen interactive rendering and offscreen RGB/depth capture through MuJoCo's renderer. Rendering is fully decoupled from the dynamics step: call `sim.render()` at any frequency independently of how fast the simulation runs.

<!-- IMAGE: MuJoCo viewer showing a single drone hovering -->

<!-- IMAGE: Side-by-side RGB and depth frames from the FPV camera -->

<!-- IMAGE: LED deck demo — 25 drones with per-drone colour and emission -->

<!-- IMAGE: Collision box geometry vs default sphere geometry -->

## Render modes

`sim.render()` accepts a `mode` argument that controls what it returns:

| Mode | Return value | Description |
|---|---|---|
| `"human"` (default) | `None` | Opens an interactive MuJoCo viewer window |
| `"rgb_array"` | `(H, W, 3) uint8` | Offscreen RGB frame |
| `"depth_array"` | `(H, W) float32` | Offscreen depth frame in meters |
| `"rgbd_tuple"` | `(rgb, depth)` | Both channels as a tuple |

<!-- notest: requires rendering -->
```{ .python notest }
sim.render()                                   # interactive window
rgb = sim.render(mode="rgb_array")             # numpy array (H, W, 3)
depth = sim.render(mode="depth_array")         # numpy array (H, W)
rgb, depth = sim.render(mode="rgbd_tuple", camera="fpv_cam:0", width=320, height=240)
sim.close()                                    # close the viewer
```

## Cameras

Pass a camera name or integer ID to select which camera to render from. The default (`camera=-1`) uses the free camera. Each drone ships with a first-person view camera named `fpv_cam:<drone_index>`:

<!-- notest: requires rendering -->
```{ .python notest }
sim.render(camera="fpv_cam:0")   # first-person view from drone 0
sim.render(camera=0)             # camera by integer ID
```

## Raycasting and depth sensing

For obstacle sensing or perception-based controllers, `render_depth` fires a ray from each camera pixel and returns per-pixel distances — faster than full RGB rendering because it skips lighting and colour computation:

<!-- notest: requires rendering -->
```{ .python notest }
import jax.numpy as jnp
from crazyflow.sim.sensors import build_render_depth_fn, render_depth

# One-shot depth render — returns (n_worlds, H, W)
dist = render_depth(sim, camera=0, resolution=(100, 100), include_drone=False)
dist = dist.at[dist > 1.5].set(jnp.nan)

# Compiled variant for repeated calls
render_fn = build_render_depth_fn(
    sim.mjx_model,
    camera=0,
    resolution=(200, 200),
    geomgroup=(1, 1, 0, 1, 1, 1, 1, 1),  # exclude drone visual mesh
)
dist = render_fn(sim)
```

## Changing materials at runtime

`change_material` updates the RGBA colour and emission intensity of any named material on any subset of drones without rebuilding the model:

<!-- notest: requires rendering -->
```{ .python notest }
import numpy as np
from crazyflow.sim.visualize import change_material

drone_ids = np.arange(sim.n_drones)
change_material(sim, mat_name="led_top", drone_ids=drone_ids, rgba=np.array([1, 0.3, 0, 1]), emission=0.8)
sim.render()
```

## Rendering world 0 vs other worlds

`sim.render()` always renders a single world at a time. Pass `world=<index>` to choose which one:

<!-- notest: requires rendering -->
```{ .python notest }
sim.render(world=0)   # default
sim.render(world=3)   # render world 3
```

## Sync and performance

Rendering triggers an implicit synchronization between the JAX dynamics state (`sim.data`) and the MuJoCo render buffers (`sim.mjx_data`). This sync computes full forward kinematics, camera transforms, and collision geometry — it is the most expensive operation per render call. See [MuJoCo Integration](mujoco.md#synchronization) for details on how to avoid redundant syncs.

## Next steps

- [MuJoCo Integration](mujoco.md) — how the scene is built, how to add objects, and sync internals
- [Examples](../examples/index.md#cameras-and-rgbd) — FPV camera and RGBD capture
- [Examples](../examples/index.md#raycasting-and-depth-sensing) — compiled depth renderer
- [Examples](../examples/index.md#led-deck-and-materials) — per-drone colour control
