# Gaussian Splat Rendering

Crazyflow renders photorealistic images with 3D gaussian splatting through [splax](https://github.com/learnsyslab/splax). A web-based viewer visualizes the simulation as splats. A camera sensor renders batched RGB(-D) images across all worlds. Both combine a static scene splat with one splat per drone that follows the drone's pose.

<!-- TODO: Add image of the web viewer showing the hall splat with a drone splat mid-flight -->

## Installation

Install crazyflow with the `splats` extra. It pulls in splax with its viewer dependencies.

```bash
pip install "crazyflow[splats]"
```

The web viewer runs on any device. The camera sensor rasterizes with CUDA kernels and requires an NVIDIA GPU.

## Demo assets

Demo splats of a flight hall and a Crazyflie drone are published in the [amacati/splats](https://huggingface.co/datasets/amacati/splats) dataset. splax's `fetch` downloads a URL into a local cache on first use and returns the cached path afterwards. The cache lives at `~/.cache/splax` and can be moved with the `SPLAX_CACHE` environment variable. Passing `force=True` refreshes the cache.

<!-- notest: requires splax and network access -->
```{ .python notest }
from splax.io import fetch

assets = "https://huggingface.co/datasets/amacati/splats/resolve/main"
scene = fetch(f"{assets}/robot_hall.ply")
drone = fetch(f"{assets}/cf21B_500.ply")
```

## Attaching splats

Splats are loaded from 3D gaussian splatting `.ply` files into a `SplatData` struct stored in the simulation's plugin data under `sim.data.plugins["splats"]`. They therefore persist across resets and travel with the sim data through `jax.jit`. The files must be aligned with the simulation frames. The scene splat is aligned with the MuJoCo world frame, with +z up and metric scale. The drone splat is aligned with the drone body frame and centered on its center of mass.

<!-- notest: requires splax and splat .ply files -->
```{ .python notest }
from crazyflow.sim import Sim
from crazyflow.sim.splat import attach_splats

sim = Sim(n_worlds=4, n_drones=2)
attach_splats(sim, scene=scene, drone=drone)
```

The drone splat is replicated once per drone, and `SplatData.slices` records each drone's index range in the buffer. Both arguments are optional. Pass only `scene` for a static environment, or only `drone` for splats in an empty world.

## Web viewer

`SplatViewer` renders the simulation in a web-based viewer served at `http://localhost:8080`. Constructing the viewer streams the splats to the browser once. `update` then only pushes the current drone poses, so the viewer runs at real-time rates on any device. The viewer is independent of `sim.render()`. Both can be open at the same time, e.g. to compare the mesh scene and the splat scene side by side.

<!-- notest: requires splax and splat .ply files -->
```{ .python notest }
from crazyflow.sim.splat import SplatViewer

viewer = SplatViewer(sim)
for _ in range(500):
    sim.step(sim.freq // sim.control_freq)
    viewer.update(sim, world=0)
viewer.close()
```

## Camera sensor

`render_splat_rgb` from `crazyflow.sim.sensors.splat` renders RGB images from any model camera, batched over all worlds and drones. It uses splax's CUDA rasterizer, so the simulation must run on the GPU. Scene gaussians stay static. Each drone's gaussians follow its current pose via splax's dynamic transforms, without copying the splat buffer.

<!-- notest: requires splax and a CUDA GPU -->
```{ .python notest }
from crazyflow.sim.sensors.splat import build_render_splat_fn, render_splat_rgb

sim = Sim(n_worlds=4, n_drones=2, device="gpu")
attach_splats(sim, scene=scene, drone=drone)
imgs = render_splat_rgb(sim, resolution=(320, 240))  # (4, 2, 240, 320, 3) in [0, 1]
imgs = render_splat_rgb(sim, drones=0, resolution=(320, 240))  # single drone: (4, 1, 240, 320, 3)

# Bake input args into a compiled function. The result is a pure function of sim.data, so it can be
# traced, jitted, and differentiated.
render_fn = build_render_splat_fn(sim, resolution=(320, 240))
imgs = render_fn(sim.data)
```

`exclude_self=True` hides each drone's own splat from its own camera, so a drone sees the others but not itself.

## Depth sensor

`render_splat_rgbd` adds depth as a fourth channel.

<!-- notest: requires splax and a CUDA GPU -->
```{ .python notest }
from crazyflow.sim.sensors.splat import build_render_splat_rgbd_fn, render_splat_rgbd

rgbd = render_splat_rgbd(sim, resolution=(320, 240), max_range=8.0)  # (4, 2, 240, 320, 4)
rgb, depth = rgbd[..., :3], rgbd[..., 3]

# Bake camera intrinsics, sensor range, and splat metadata into a compiled function
render_fn = build_render_splat_rgbd_fn(sim, resolution=(320, 240), max_range=8.0)
rgbd = render_fn(sim.data)
```

Gaussians are semi-transparent, so a pixel only carries a usable depth once enough of them accumulate behind it. Pixels whose coverage stays below `alpha_threshold` count as empty space and report `max_range`, which is also the value depth is clipped to.

The values are depth along the camera's optical axis in meters. `render_depth` from `crazyflow.sim.sensors.depth` raycasts the MuJoCo geometry instead and returns ray distances, so the two are not interchangeable.

## Next steps

- [Examples](../examples/rendering.md#splat-viewer) — web-based splat viewer
- [Examples](../examples/rendering.md#splat-camera) — batched FPV cameras across worlds and drones
- [Examples](../examples/rendering.md#splat-depth-camera) — compiled RGB-D renderer
- [Examples](../examples/rendering.md#differentiable-splat-rendering) — photometric pose gradients
