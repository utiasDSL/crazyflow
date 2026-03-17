import jax.numpy as jnp
import matplotlib.pyplot as plt

from crazyflow.sim import Sim
from crazyflow.sim.sensors import build_render_depth_fn, render_depth


def main(plot: bool = False):
    sim = Sim()
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=sim.data.states.pos.at[..., 2].set(0.2))
    )
    # The easiest way to get depth images is to use the render_depth function
    dist = render_depth(sim, camera=0, resolution=(100, 100), include_drone=False)
    dist = dist.at[dist > 1.5].set(jnp.nan)  # Cap max distance for better visualization
    if plot:
        plt.imshow(dist[0], cmap="viridis")
        plt.colorbar(label="Distance (m)")
        plt.title("Raycast Distance from Camera")
        plt.show()
    # We can also build a depth renderer function for better performance if we need maximum speed or
    # more fine-grained control. Here we only render the drone collision geometry to avoid expensive
    # raycasting against the high-poly visual mesh of the drone.
    render_depth_fn = build_render_depth_fn(
        sim.mjx_model, camera=0, resolution=(200, 200), geomgroup=(1, 1, 0, 1, 1, 1, 1, 1)
    )
    dist_fn = render_depth_fn(sim)
    dist_fn = dist_fn.at[dist_fn > 1.5].set(jnp.nan)  # Cap max distance for better visualization
    if plot:
        plt.imshow(dist_fn[0], cmap="viridis")
        plt.colorbar(label="Distance (m)")
        plt.title("Raycast Distance from Camera (Compiled)")
        plt.show()


if __name__ == "__main__":
    main(plot=True)
