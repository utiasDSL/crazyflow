import jax.numpy as jnp
import matplotlib.pyplot as plt

from crazyflow.sim import Sim
from crazyflow.sim.sensors import render_depth


def main(plot: bool = False):
    sim = Sim()
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=sim.data.states.pos.at[..., 2].set(0.2))
    )
    dist = render_depth(sim, camera=0, resolution=(100, 100))
    dist = dist.at[dist > 1.5].set(jnp.nan)  # Cap max distance for better visualization
    if not plot:
        return
    plt.imshow(dist[0], cmap="viridis")
    plt.colorbar(label="Distance (m)")
    plt.title("Raycast Distance from Camera")
    plt.show()


if __name__ == "__main__":
    main(plot=True)
