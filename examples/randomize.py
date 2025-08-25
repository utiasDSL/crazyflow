import jax
import numpy as np

from crazyflow.control import Control
from crazyflow.randomize import randomize_inertia, randomize_mass
from crazyflow.sim import Sim
from crazyflow.utils import grid_2d


def main():
    sim = Sim(n_worlds=3, n_drones=4, control=Control.state)
    sim.reset()

    duration = 5.0
    fps = 60

    # Randomize the inertia and mass of the drones
    mask = np.array([True, False, False])  # Only randomize the first world
    mass = sim.data.params.mass
    mass_rng = mass + jax.random.normal(jax.random.key(0), (sim.n_worlds, sim.n_drones, 1)) * 1e-4
    J = sim.data.params.J
    J_rng = J + jax.random.normal(jax.random.key(0), (sim.n_worlds, sim.n_drones, 3, 3)) * 1e-5

    randomize_mass(sim, mass_rng, mask)
    # Note: The mask is optional. We can also randomize all worlds at once
    randomize_mass(sim, mass_rng)
    randomize_inertia(sim, J_rng, mask)

    cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
    cmd[..., 2] = 0.4
    cmd[..., :2] = grid_2d(sim.n_drones) * 0.25

    # Simulate for 5 seconds. Each drone should behave slightly differently due to the randomization
    for i in range(int(duration * sim.control_freq)):
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
