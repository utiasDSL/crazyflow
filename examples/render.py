import numpy as np

from crazyflow.sim import Physics, Sim


def main():
    """Spawn 5 drones in one world and render it."""
    n_worlds, n_drones = 1, 100
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=Physics.sys_id, device="cpu")
    fps = 60
    cmd = np.array([[[0.3, 0, 0, 0] for _ in range(sim.n_drones)]])

    for i in range(int(5 * sim.control_freq)):
        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
