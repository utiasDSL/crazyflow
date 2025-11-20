import numpy as np

from crazyflow.control.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.visualize import change_material


def main():
    """Spawn 25 drones in one world and activate led decks."""
    sim = Sim(n_drones=25, control=Control.state)
    fps = 60
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))
    cmd[..., 3] = sim.data.params.mass[0, 0, 0] * 9.81
    rgbas = np.random.default_rng(0).uniform(0, 1, (sim.n_drones, 4))
    rgbas[..., 3] = 1.0

    init_pos = np.array(sim.data.states.pos[0, :, :])
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
    cmd[:, :, :3] = init_pos
    cmd[:, :, 2] += 1.5

    for i in range(int(10 * sim.control_freq)):
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            even_ids = np.arange(0, sim.n_drones, 2)
            odd_ids = np.arange(1, sim.n_drones, 2)
            emission = np.sin(i / sim.control_freq * np.pi)
            change_material(
                sim,
                mat_name="led_top",
                drone_ids=even_ids,
                rgba=rgbas[even_ids, :],
                emission=emission,
            )
            change_material(
                sim,
                mat_name="led_bot",
                drone_ids=odd_ids,
                rgba=rgbas[odd_ids, :],
                emission=emission,
            )
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
