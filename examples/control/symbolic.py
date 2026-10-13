"""Create symbolic CasADi dynamics for model-based control.

`symbolic_dynamics` returns the dynamics as CasADi expressions with the nominal parameters of the
simulated drone, so they can be used in model-based controllers such as MPC. A CasADi integrator
turns them into discrete-time dynamics at the simulation frequency.
"""

import casadi as cs
import numpy as np

from crazyflow.dynamics import parametrize
from crazyflow.dynamics.so_rpy import symbolic_dynamics
from crazyflow.sim import Sim


def main():
    # We can create a symbolic dynamics directly from the simulation. Note that this will use the
    # nominal parameters of the simulation and choose the control type based on the simulation.
    sim = Sim(dynamics="so_rpy", freq=500)
    X_dot, X, U, Y = parametrize(symbolic_dynamics, sim.drone)()
    assert X_dot.shape == (13, 1)
    assert X.shape == (13, 1)
    assert U.shape == (4, 1)  # Attitude control
    assert Y.shape == (7, 1)  # 3 for pos and 4 for quat

    # To create a discrete-time dynamics that you can integrate, you can use the integrator function
    # from CasADi.
    fd = cs.integrator("fd", "cvodes", {"x": X, "p": U, "ode": X_dot}, 0, 1 / sim.freq)
    x0 = np.ones((13, 1))
    u = np.ones((4, 1))
    res = fd(x0=x0, p=u)
    assert res["xf"].shape == (13, 1)


if __name__ == "__main__":
    main()
