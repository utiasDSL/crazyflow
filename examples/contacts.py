import numpy as np

from crazyflow.sim import Physics, Sim


def main():
    """Spawn multiple drones in multiple worlds and check for contacts."""
    n_worlds, n_drones = 2, 3
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=Physics.sys_id, device="cpu")
    fps = 60
    x = np.arange(n_drones) / 3
    sim.states["pos"] = sim.states["pos"].at[..., 0].set(x)

    cmd = np.array([[[0.3, 0, 0, 0] for _ in range(n_drones)] for _ in range(n_worlds)])
    for _ in range(int(5 * fps)):
        start = sim.time
        while sim.time - start < 1.0 / fps:
            sim.attitude_control(cmd)
            sim.step()
        print(f"Contacts: {sim.contacts().any()}")
        sim.render()
    sim.close()


if __name__ == "__main__":
    main()
