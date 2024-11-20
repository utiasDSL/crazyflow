import numpy as np

from crazyflow.sim import Physics, Sim


def main():
    """Spawn 5 drones in one world and render it."""
    n_worlds, n_drones = 1, 100
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=Physics.sys_id, device="gpu", freq=60)
    fps = 60
    
    cmd = np.array([[[0.3, 0, 0, 0] for _ in range(n_drones)] for _ in range(n_worlds)])
    for _ in range(int(5 * fps)):
        start = sim.time
        while sim.time - start < 1.0 / fps:
            sim.attitude_control(cmd)
            sim.step()
        sim.render()
    sim.close()


if __name__ == "__main__":
    main()
