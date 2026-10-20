"""Render RGB-D images offscreen from the drone's FPV camera.

Offscreen rendering returns RGB-D images on every frame. The FPV camera (`fpv_cam`) is attached to
the drone and moves with it. Other cameras are selected by name or id in `sim.render`.

Run with `--plot` to animate the images or `--save` to save them as cameras.gif, which requires
matplotlib from the `tests` environment.
"""

import argparse
import os
import time

os.environ["SCIPY_ARRAY_API"] = "1"

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from crazyflow.control import Control
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.sim.integration import Integrator


def trajectory(t: float, t_tot: float) -> np.ndarray:
    phi = 2 * np.pi * t / t_tot + np.pi
    circle = np.array([np.cos(phi), np.sin(phi)])
    yaw = 2 * np.pi * t / t_tot
    cmd = np.zeros((1, 1, 16))
    cmd[..., :2] = circle  # xy
    cmd[..., 2] = 0.1 + 0.5 * t / t_tot  # z
    cmd[..., 9:13] = R.from_euler("z", yaw).as_quat()

    return cmd


def add_smiley(sim: Sim):
    # Add 3d object to sim
    # create box spec from an XML string
    box_xml = """
    <mujoco model="box_model">
      <worldbody>
        <body name="cube" pos="0 0 0">
          <geom type="box" size="0.05 0.05 0.05" rgba="0.8 0.4 0.2 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    box_spec = mujoco.MjSpec.from_string(box_xml)
    frame = sim.spec.worldbody.add_frame()
    boxes = [
        # eyes
        ((0.0, -0.15, 0.6), (1, 0, 0, 0)),
        ((0.0, 0.15, 0.6), (1, 0, 0, 0)),
        # mouth
        ((0.0, -0.2, 0.4), (1, 0, 0, 0)),
        ((0.0, 0.2, 0.4), (1, 0, 0, 0)),
        ((0.0, -0.1, 0.3), (1, 0, 0, 0)),
        ((0.0, 0.0, 0.3), (1, 0, 0, 0)),
        ((0.0, 0.1, 0.3), (1, 0, 0, 0)),
    ]
    for i, x in enumerate(boxes):
        box_body = box_spec.body("cube")
        box = frame.attach_body(box_body, "", f":{i}")
        box.pos = x[0]
        box.quat = x[1]
    sim.build_mjx()
    sim.build_reset_fn()


def main(plot: bool = False, save: bool = False):
    """Render RGB-D images from the fpv camera and optionally animate or save them."""
    # Setup sim
    sim = Sim(
        n_drones=1,
        control=Control.state,
        integrator=Integrator.rk4,
        dynamics=Dynamics.first_principles,
        drone="cf2x_T350",
    )
    add_smiley(sim)
    sim.reset()
    pos = sim.data.states.pos.at[...].set([-1, 0, 0])
    states = sim.data.states.replace(pos=pos)
    sim.data = sim.data.replace(states=states)
    duration = 5
    fps = 50
    resolution = (160, 120)
    timings, frames = [], []

    for _ in range(int(duration * fps)):
        t = sim.data.core.steps[0, 0] / sim.freq
        sim.state_control(trajectory(t, duration))
        sim.step(sim.freq // fps)

        t1 = time.perf_counter()
        # mode: Either "human" for the regular window, "rgb_array" for an RGB array,
        #       "depth_array" for a depth array, or "rgbd_tuple" for both at the same time.
        # camera: The name or id of the camera. The names are specified in the corresponding
        #         xml file in crazyflow/drones. For example, "fpv_cam:0" is the first-person view
        #         camera of the first drone, "track_cam:0" is the tracking camera of the first
        #         drone. Id -1 is the global camera.
        rgbd = sim.render(
            width=resolution[0], height=resolution[1], mode="rgbd_tuple", camera="fpv_cam:0"
        )
        t2 = time.perf_counter()
        timings.append(t2 - t1)
        if rgbd is not None:
            frames.append(rgbd)

    sim.close()

    t_mean = np.mean(timings)
    print(f"Average render time {t_mean * 1000:.2f}ms, eqivalent to {1 / t_mean:.2f}fps")
    print("For more optimized depth rendering, check out the raycasting.py example.")
    if plot or save:
        plot_results(frames, fps, show=plot, save=save)


def plot_results(frames: list[tuple[np.ndarray, np.ndarray]], fps: int, show: bool, save: bool):
    """Animate the RGB and depth frames and optionally save them as cameras.gif."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax1.imshow(frames[0][0])
    ax1.set_title("RGB")
    ax1.axis("off")
    im2 = ax2.imshow(frames[0][1], cmap="viridis")
    ax2.set_title("Depth")
    ax2.axis("off")
    fig.tight_layout()

    def update_frame(i: int):  # noqa: ANN202
        rgb, depth = frames[i]
        im1.set_data(rgb)
        im2.set_data(depth)
        im2.set_clim(np.nanmin(depth), np.nanmax(depth))
        return im1, im2

    anim = animation.FuncAnimation(
        fig, update_frame, frames=len(frames), interval=1000 / fps, blit=True, repeat=False
    )
    if save:
        anim.save("cameras.gif", writer="pillow", fps=fps)
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Animate the images")
    parser.add_argument("--save", action="store_true", help="Save the images as cameras.gif")
    args = parser.parse_args()
    main(plot=args.plot, save=args.save)
