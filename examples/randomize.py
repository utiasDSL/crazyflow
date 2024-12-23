import numpy as np

from crazyflow.constants import MASS, J
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
    mass = np.ones((sim.n_worlds, sim.n_drones)) * MASS
    randomized_j = J + np.random.uniform(-1.5e-5, 1.5e-5, size=(sim.n_worlds, sim.n_drones, 3, 3))
    randomized_mass = mass + np.random.uniform(-0.005, 0.005, size=(sim.n_worlds, sim.n_drones))

    randomize_mass(sim, randomized_mass, mask)
    # Note: The mask is optional. We can also randomize all worlds at once
    randomize_mass(sim, randomized_mass)
    randomize_inertia(sim, randomized_j, mask)

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
