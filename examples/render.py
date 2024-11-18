import numpy as np

from crazyflow.sim import Physics, Sim


def main():
    """Spawn 5 drones in one world and render it."""
    n_worlds, n_drones = 1, 100
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=Physics.sys_id, device="gpu", freq=60)
    fps = 60
    N = int(np.ceil(np.sqrt(n_drones)))
    points = np.linspace(-0.5 * (N - 1), 0.5 * (N - 1), N)
    x, y = np.meshgrid(points, points)
    grid = np.stack((x.flatten(), y.flatten()), axis=-1)
    grid = grid[:n_drones]
    sim.states["pos"] = sim.states["pos"].at[..., :2].set(grid)

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
