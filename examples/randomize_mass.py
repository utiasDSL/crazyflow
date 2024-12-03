import numpy as np
import jax.numpy as jnp
import jax

from crazyflow.control.controller import Control, Controller
from crazyflow.sim.core import Sim
from crazyflow.sim.physics import Physics
from crazyflow.utils import grid_2d
from crazyflow.randomize import randomize_mass


def main():
    sim = Sim(
        n_worlds=1,
        n_drones=4,
        physics=Physics.analytical,
        control=Control.state,
        controller=Controller.emulatefirmware,
        freq=500,
        control_freq=500,
        device="cpu",
    )

    sim.reset()
    duration = 5.0
    fps = 60

    key = jax.random.PRNGKey(0)
    add_on_mass = jax.random.uniform(key, shape=(sim.n_worlds, sim.n_drones, 1), minval=-0.005, maxval=0.005)
    masses = jnp.ones((sim.n_worlds, sim.n_drones, 1)) * 0.025
    randomized_masses = masses + add_on_mass

    randomize_mass(sim, randomized_masses)

    for i in range(int(duration * sim.freq)):
        if sim.controllable:
            # State cmd is [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate]
            cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
            grid = grid_2d(sim.n_drones)
            cmd[..., 2] = 0.4
            cmd[..., 0:2] = grid * 0.25
            sim.state_control(cmd)
        sim.step()
        if ((i * fps) % sim.freq) < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
