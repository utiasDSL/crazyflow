import numpy as np

from crazyflow.sim import Physics, Sim
from crazyflow.sim.sim import use_box_collision


def main():
    """Spawn multiple drones in multiple worlds and check for contacts."""
    n_worlds, n_drones = 2, 3
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=Physics.so_rpy, device="cpu")
    use_box_collision(sim, enable=True)  # Enable box collision for all drones
    fps = 60

    cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))
    cmd[..., 3] = sim.data.params.mass[0, 0, 0] * 9.81 * 1.04
    for i in range(int(2 * sim.control_freq)):
        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
            print(f"Contacts: {sim.contacts().any()}")
    sim.close()


if __name__ == "__main__":
    main()
