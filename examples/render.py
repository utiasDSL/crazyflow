import mujoco as mj

from crazyflow.sim import Physics, Sim


def main():
    """Spawn a single drone in one world and render it."""
    sim = Sim(physics=Physics.sys_id, device="gpu")
    fps = 60
    for _ in range(int(10 * fps)):
        start = sim._data.time
        while sim._data.time - start < 1.0 / fps:
            mj.mj_step(sim._model, sim._data)
        sim.render()
    sim.close()


if __name__ == "__main__":
    main()
