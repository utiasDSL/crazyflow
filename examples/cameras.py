import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib import animation

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.integration import Integrator
from crazyflow.sim.physics import Physics


def control(t: float, t_tot: float) -> np.ndarray:
    phi = 2 * np.pi * t / t_tot + np.pi
    circle = np.array([np.cos(phi), np.sin(phi)])
    cmd = np.zeros((1, 1, 13))
    cmd[..., :2] = circle  # xy
    cmd[..., 2] = 0.1 + 0.5 * t / t_tot  # z
    cmd[..., -4] = 1.9 * np.pi * t / t_tot  # yaw

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


def main(show_plot: bool = False, save_plot: bool = False):
    """Example showing the rendering feature and saving a gif via FuncAnimation."""
    # Setup sim
    sim = Sim(
        n_drones=1,
        control=Control.state,
        integrator=Integrator.rk4,
        physics=Physics.first_principles,
        drone_model="cf2x_T350",
    )
    add_smiley(sim)
    sim.reset()
    pos = sim.data.states.pos.at[...].set([-1, 0, 0])
    states = sim.data.states.replace(pos=pos)
    sim.data = sim.data.replace(states=states)
    duration = 8
    fps = 25
    timings = []

    # Set up matplotlib rendering
    resolution = (160, 120)
    rgb = np.zeros((resolution[1], resolution[0], 3))
    d = np.zeros((resolution[1], resolution[0]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax1.imshow(rgb)
    ax1.set_title("RGB")
    ax1.axis("off")
    im2 = ax2.imshow(d, cmap="grey")
    ax2.set_title("Depth")
    ax2.axis("off")
    fig.tight_layout()

    # Animation setup
    def update_frame(_):  # noqa: ANN202
        t = sim.data.core.steps[0, 0] / sim.freq
        sim.state_control(control(t, duration))
        sim.step(sim.freq // fps)

        t1 = time.perf_counter()
        rgbd = sim.render(
            width=resolution[0], height=resolution[1], mode="rgbd_tuple", camera="fpv_cam:0"
        )
        t2 = time.perf_counter()
        timings.append(t2 - t1)
        if rgbd is None:
            return im1, im2
        rgb, depth = rgbd
        im1.set_data(rgb)
        im2.set_data(depth)
        im2.set_clim(np.nanmin(depth), np.nanmax(depth))
        return im1, im2

    anim = animation.FuncAnimation(fig, update_frame, frames=int(duration * fps), blit=True)
    if show_plot:
        plt.show()  # this is slow
    if save_plot:
        anim.save("cameras.gif", writer="pillow", fps=fps)

    sim.close()

    t_mean = np.mean(timings)
    print(f"Average render time {t_mean * 1000}ms, eqivalent to {1 / t_mean}fps")


if __name__ == "__main__":
    main(show_plot=True, save_plot=False)
