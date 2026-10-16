# Rendering examples

Crazyflow has two renderers. MuJoCo renders the simulated geometry, and gaussian splats render photorealistic images of a captured scene.

---

## MuJoCo rendering

Rendering RGB-D images, changing materials at runtime, raycasting depth, drawing trajectories, and configuring the viewer camera. See [MuJoCo Rendering](../user-guide/mujoco-rendering.md) for the rendering modes and cameras.

### Cameras and RGBD

<figure class="example-media">
  <img src="../../img/examples/cameras.gif" alt="RGB-D camera outputs from a Crazyflow drone simulation">
</figure>

--8<-- example "examples/rendering/cameras.py"

---

### LED deck and materials

<figure class="example-media example-media--compact">
  <img src="../../img/examples/led_decks.png" alt="Crazyflow drones with runtime-controlled LED deck materials">
</figure>

--8<-- example "examples/rendering/led_deck.py"

---

### Raycasting and depth sensing

--8<-- example "examples/rendering/raycasting.py"

---

### Drawing trajectories

--8<-- example "examples/rendering/draw_line.py"

---

### Camera configuration

--8<-- example "examples/rendering/cam_config.py"

---

## Gaussian splat rendering

Viewing the simulation as gaussian splats, rendering RGB-D images from the drone cameras, and differentiating through the renderer. The examples download the demo splats from the [amacati/splats](https://huggingface.co/datasets/amacati/splats) dataset on first run. See [Gaussian Splat Rendering](../user-guide/splat-rendering.md) for the setup.

### Splat viewer

--8<-- example "examples/rendering/splat_viewer.py"

---

### Splat camera

--8<-- example "examples/rendering/splat_camera.py"

---

### Splat depth camera

--8<-- example "examples/rendering/splat_depth.py"

---

### Differentiable splat rendering

--8<-- example "examples/rendering/splat_gradients.py"
