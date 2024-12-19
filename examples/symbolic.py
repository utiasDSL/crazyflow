from crazyflow.constants import MASS, J
from crazyflow.sim import Sim
from crazyflow.sim.symbolic import symbolic, symbolic_from_sim


def main():
    # We can create a symbolic model without the simulation
    dt = 1 / 500
    symbolic_model = symbolic(MASS, J, dt)

    # Or we can create a symbolic model directly from the simulation. Note that this will use the
    # nominal parameters of the simulation.
    sim = Sim()
    symbolic_model = symbolic_from_sim(sim)
    assert symbolic_model.nx == 12  # 3 for pos, 3 for orient, 3 for vel, 3 for ang vel
    assert symbolic_model.nu == 4  # collective thrust + 3-dim attitude


if __name__ == "__main__":
    main()
