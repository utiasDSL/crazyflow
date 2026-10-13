# Rendering examples

Crazyflow has two renderers. MuJoCo renders the simulated geometry, and gaussian splats render photorealistic images of a captured scene.

---

## MuJoCo rendering

Rendering RGB-D images, changing materials at runtime, and raycasting depth. See [MuJoCo Rendering](../user-guide/mujoco-rendering.md) for the rendering modes and cameras.

### Cameras and RGBD

Offscreen rendering returns RGB-D images on every frame. The FPV camera (`fpv_cam`) is attached to the drone and moves with it.

<figure class="example-media">
  <img src="../../img/examples/cameras.gif" alt="RGB-D camera outputs from a Crazyflow drone simulation">
</figure>

<!-- notest: imported script, covered by tests/integration/test_examples.py -->
```{ .python notest }
--8<-- "examples/rendering/cameras.py"
```

```bash
python examples/rendering/cameras.py
```

---

### LED deck and materials

`change_material` updates the RGBA colour and emission of any named material on any subset of drones at runtime.

<figure class="example-media example-media--compact">
  <img src="../../img/examples/led_decks.png" alt="Crazyflow drones with runtime-controlled LED deck materials">
</figure>

<!-- notest: imported script, covered by tests/integration/test_examples.py -->
```{ .python notest }
--8<-- "examples/rendering/led_deck.py"
```

```bash
python examples/rendering/led_deck.py
```

---

### Raycasting and depth sensing

`render_depth` fires rays from a camera and returns per-pixel distances. This is faster than full RGB rendering and useful for obstacle sensing or depth-based controllers.

<!-- notest: imported script, covered by tests/integration/test_examples.py -->
```{ .python notest }
--8<-- "examples/rendering/raycasting.py"
```

```bash
python examples/rendering/raycasting.py
```

---

## Gaussian splat rendering

Viewing the simulation as gaussian splats, rendering RGB-D images from the drone cameras, and differentiating through the renderer. The examples need the `splats` extra (`pip install "crazyflow[splats]"`) and download the demo splats from the [amacati/splats](https://huggingface.co/datasets/amacati/splats) dataset on first run. See [Gaussian Splat Rendering](../user-guide/splat-rendering.md) for the setup.

### Splat viewer

`SplatViewer` serves a web viewer at `http://localhost:8080`. The splats are streamed to the browser once, after which `update` only pushes the drone poses, so the viewer runs in real time on any device.

<!-- notest: imported script, covered by tests/integration/test_examples.py -->
```{ .python notest }
--8<-- "examples/rendering/splat_viewer.py"
```

```bash
python examples/rendering/splat_viewer.py
```

---

### Splat camera

`render_splat_rgb` renders the FPV camera of every drone in every world in a single batched call. Two drones circle the hall center while facing each other. The splat sensors use splax's CUDA rasterizer and require a GPU.

<figure class="example-media example-media--compact">
  <video src="../../img/examples/splat_camera.mp4" autoplay loop muted playsinline aria-label="Gaussian splat FPV camera views of two drones circling in a flight hall"></video>
</figure>

<!-- notest: imported script, covered by tests/integration/test_examples.py -->
```{ .python notest }
--8<-- "examples/rendering/splat_camera.py"
```

```bash
python examples/rendering/splat_camera.py
```

---

### Splat depth camera

`build_render_splat_rgbd_fn` compiles an RGB-D renderer that is a pure function of `sim.data`. The drone flies an elliptical lap through the hall while the depth channel is displayed. Splat depth is measured along the optical axis, whereas `render_depth` returns ray distances.

<figure class="example-media example-media--compact">
  <video src="../../img/examples/splat_depth.mp4" autoplay loop muted playsinline aria-label="Gaussian splat depth camera of a drone flying a lap through a flight hall"></video>
</figure>

<!-- notest: imported script, covered by tests/integration/test_examples.py -->
```{ .python notest }
--8<-- "examples/rendering/splat_depth.py"
```

```bash
python examples/rendering/splat_depth.py
```

---

### Differentiable splat rendering

Splat rendering is differentiable w.r.t. the camera pose. A reference FPV image is rendered at a baseline pose, and its photometric error is differentiated over small z and yaw offsets. The resulting gradients match the gradient of the squared pose error in sign and shape.

<figure class="example-media">
  <img src="../../img/examples/splat_gradients.png" alt="Photometric splat gradients compared with known pose gradients for z and yaw offsets">
</figure>

<!-- notest: imported script, covered by tests/integration/test_examples.py -->
```{ .python notest }
--8<-- "examples/rendering/splat_gradients.py"
```

```bash
python examples/rendering/splat_gradients.py
```
