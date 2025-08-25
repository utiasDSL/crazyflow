import casadi as cs
import numpy as np

from crazyflow.sim import Sim
from crazyflow.sim.symbolic import symbolic_from_sim


def main():
    # We can create a symbolic model directly from the simulation. Note that this will use the
    # nominal parameters of the simulation and choose the control type based on the simulation.
    sim = Sim(physics="so_rpy", freq=500)
    X_dot, X, U, Y = symbolic_from_sim(sim)
    assert X_dot.shape == (13, 1)
    assert X.shape == (13, 1)
    assert U.shape == (4, 1)  # Attitude control
    assert Y.shape == (7, 1)  # 3 for pos and 4 for quat

    # To create a discrete-time model that you can integrate, you can use the integrator function
    # from CasADi.
    fd = cs.integrator("fd", "cvodes", {"x": X, "p": U, "ode": X_dot}, 0, 1 / sim.freq)
    x0 = np.ones((13, 1))
    u = np.ones((4, 1))
    res = fd(x0=x0, p=u)
    assert res["xf"].shape == (13, 1)


if __name__ == "__main__":
    main()
