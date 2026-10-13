"""Render depth images by raycasting the scene geometry.

`render_depth` fires rays from a camera and returns per-pixel distances. This is faster than full
RGB rendering and useful for obstacle sensing or depth-based controllers. `build_render_depth_fn`
compiles a renderer for repeated calls.

Run with `--plot` to plot the results, which requires matplotlib from the `tests` environment.
"""

import argparse

import jax.numpy as jnp

from crazyflow.sim import Sim
from crazyflow.sim.sensors.depth import build_render_depth_fn, render_depth


def main(plot: bool = False):
    sim = Sim()
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=sim.data.states.pos.at[..., 2].set(0.2))
    )
    # The easiest way to get depth images is to use the render_depth function
    dist = render_depth(sim, camera=0, resolution=(100, 100), include_drone=False)
    dist = dist.at[dist > 1.5].set(jnp.nan)  # Cap max distance for better visualization
    if plot:
        plot_depth(dist[0], "Raycast Distance from Camera")
    # We can also build a depth renderer function for better performance if we need maximum speed or
    # more fine-grained control. Here we only render the drone collision geometry to avoid expensive
    # raycasting against the high-poly visual mesh of the drone.
    render_depth_fn = build_render_depth_fn(
        sim.mjx_model, camera=0, resolution=(200, 200), geomgroup=(1, 1, 0, 1, 1, 1, 1, 1)
    )
    dist_fn = render_depth_fn(sim)
    dist_fn = dist_fn.at[dist_fn > 1.5].set(jnp.nan)  # Cap max distance for better visualization
    if plot:
        plot_depth(dist_fn[0], "Raycast Distance from Camera (Compiled)")


def plot_depth(dist: jnp.ndarray, title: str):
    import matplotlib.pyplot as plt

    plt.imshow(dist, cmap="viridis")
    plt.colorbar(label="Distance (m)")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot the results")
    main(plot=parser.parse_args().plot)
