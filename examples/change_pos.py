import numpy as np

from crazyflow.sim import Sim


def main():
    sim = Sim(control="state")
    sim.reset()
    # Replace the position of the first drone in the first world. JAX arrays are immutable, which is
    # why we cannot change the sim.data object in-place. Instead, we need to create a new sim.data
    # object with the desired changes by calling sim.data.replace(). The same logic applies to the
    # sim.data.states object contained within sim.data, and the sim.data.states.pos array. For more
    # information on changing JAX arrays, see:
    # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#in-place-updates
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=sim.data.states.pos.at[0, 0].set(np.array([1, 1, 0.2])))
    )
    control = np.zeros((sim.n_worlds, sim.n_drones, 13))
    control[..., :3] = np.array([[0.0, 0.0, 0.3]])

    for _ in range(3 * sim.control_freq):
        sim.state_control(control)
        sim.step(sim.freq // sim.control_freq)
        sim.render()
    sim.close()


if __name__ == "__main__":
    main()
