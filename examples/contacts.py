import numpy as np

from crazyflow.sim import Physics, Sim


def main():
    """Spawn multiple drones in multiple worlds and check for contacts."""
    n_worlds, n_drones = 2, 3
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=Physics.sys_id, device="cpu")
    fps = 60

    cmd = np.array([[[0.3, 0, 0, 0] for _ in range(n_drones)] for _ in range(n_worlds)])
    for i in range(int(2 * sim.control_freq)):
        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
            print(f"Contacts: {sim.contacts().any()}")
    sim.close()


if __name__ == "__main__":
    main()
