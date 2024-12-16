"""Symbolic model and utilities for drone dynamics and control.

This module provides functionality to create symbolic representations of the drone dynamics,
observations, and cost functions using CasADi. It includes:

* :class:`~.SymbolicModel`: A class that encapsulates the symbolic representation of the drone
  model, including dynamics, observations, and cost functions.
* :func:`symbolic <lsy_drone_racing.sim.symbolic.symbolic>`: A function that creates a
  :class:`~.SymbolicModel` instance for a given drone configuration.
* Helper functions for creating rotation matrices and other mathematical operations.

The symbolic models created by this module can be used for various control and estimation tasks,
such as model predictive control (MPC) or state estimation. They provide a convenient way to work
with the nonlinear dynamics of the drone in a symbolic framework, allowing for automatic
differentiation and optimization.
"""


import casadi as cs
from casadi import MX
from numpy.typing import NDArray

from crazyflow.constants import ARM_LEN, GRAVITY, SIGN_MIX_MATRIX
from crazyflow.control.controller import KF, KM
from crazyflow.sim.core import Sim


class SymbolicModel:
    """Implement the drone dynamics model with symbolic variables.

    x_dot = f(x,u), y = g(x,u), with other pre-defined, symbolic functions (e.g. cost, constraints),
    serve as priors for the controllers.

    Notes:
        * naming convention on symbolic variable and functions.
        * for single-letter symbol, use {}_sym, otherwise use underscore for delimiter.
        * for symbolic functions to be exposed, use {}_func.
    """

    def __init__(
        self, dynamics: dict[str, MX], cost: dict[str, MX], dt: float = 1e-3, solver: str = "cvodes"
    ):
        """Initialize the symbolic model.

        Args:
            dynamics: A dictionary containing the dynamics, observation functions and variables.
            cost: A dictionary containing the cost function and its variables.
            dt: The sampling time.
            solver: The integration algorithm.
        """
        self.x_sym = dynamics["vars"]["X"]
        self.u_sym = dynamics["vars"]["U"]
        self.x_dot = dynamics["dyn_eqn"]
        self.y_sym = self.x_sym if dynamics["obs_eqn"] is None else dynamics["obs_eqn"]
        self.dt = dt  # Sampling time.
        self.solver = solver  # Integration algorithm
        # Variable dimensions.
        self.nx = self.x_sym.shape[0]
        self.nu = self.u_sym.shape[0]
        self.ny = self.y_sym.shape[0]
        # Setup cost function.
        self.cost_func = cost["cost_func"]
        self.Q = cost["vars"]["Q"]
        self.R = cost["vars"]["R"]
        self.Xr = cost["vars"]["Xr"]
        self.Ur = cost["vars"]["Ur"]
        self.setup_model()
        self.setup_linearization()  # Setup Jacobian and Hessian of the dynamics and cost functions

    def setup_model(self):
        """Expose functions to evaluate the model."""
        # Continuous time dynamics.
        self.fc_func = cs.Function("fc", [self.x_sym, self.u_sym], [self.x_dot], ["x", "u"], ["f"])
        # Discrete time dynamics.
        self.fd_func = cs.integrator(
            "fd", self.solver, {"x": self.x_sym, "p": self.u_sym, "ode": self.x_dot}, 0, self.dt
        )
        # Observation model.
        self.g_func = cs.Function("g", [self.x_sym, self.u_sym], [self.y_sym], ["x", "u"], ["g"])

    def setup_linearization(self):
        """Expose functions for the linearized model."""
        # Jacobians w.r.t state & input.
        self.dfdx = cs.jacobian(self.x_dot, self.x_sym)
        self.dfdu = cs.jacobian(self.x_dot, self.u_sym)
        self.df_func = cs.Function(
            "df", [self.x_sym, self.u_sym], [self.dfdx, self.dfdu], ["x", "u"], ["dfdx", "dfdu"]
        )
        self.dgdx = cs.jacobian(self.y_sym, self.x_sym)
        self.dgdu = cs.jacobian(self.y_sym, self.u_sym)
        self.dg_func = cs.Function(
            "dg", [self.x_sym, self.u_sym], [self.dgdx, self.dgdu], ["x", "u"], ["dgdx", "dgdu"]
        )
        # Evaluation point for linearization.
        self.x_eval = MX.sym("x_eval", self.nx, 1)
        self.u_eval = MX.sym("u_eval", self.nu, 1)
        # Linearized dynamics model.
        self.x_dot_linear = (
            self.x_dot
            + self.dfdx @ (self.x_eval - self.x_sym)
            + self.dfdu @ (self.u_eval - self.u_sym)
        )
        self.fc_linear_func = cs.Function(
            "fc",
            [self.x_eval, self.u_eval, self.x_sym, self.u_sym],
            [self.x_dot_linear],
            ["x_eval", "u_eval", "x", "u"],
            ["f_linear"],
        )
        self.fd_linear_func = cs.integrator(
            "fd_linear",
            self.solver,
            {
                "x": self.x_eval,
                "p": cs.vertcat(self.u_eval, self.x_sym, self.u_sym),
                "ode": self.x_dot_linear,
            },
            0,
            self.dt,
        )
        # Linearized observation model.
        self.y_linear = (
            self.y_sym
            + self.dgdx @ (self.x_eval - self.x_sym)
            + self.dgdu @ (self.u_eval - self.u_sym)
        )
        self.g_linear_func = cs.Function(
            "g_linear",
            [self.x_eval, self.u_eval, self.x_sym, self.u_sym],
            [self.y_linear],
            ["x_eval", "u_eval", "x", "u"],
            ["g_linear"],
        )
        # Jacobian and Hessian of cost function.
        self.l_x = cs.jacobian(self.cost_func, self.x_sym)
        self.l_xx = cs.jacobian(self.l_x, self.x_sym)
        self.l_u = cs.jacobian(self.cost_func, self.u_sym)
        self.l_uu = cs.jacobian(self.l_u, self.u_sym)
        self.l_xu = cs.jacobian(self.l_x, self.u_sym)
        l_inputs = [self.x_sym, self.u_sym, self.Xr, self.Ur, self.Q, self.R]
        l_inputs_str = ["x", "u", "Xr", "Ur", "Q", "R"]
        l_outputs = [self.cost_func, self.l_x, self.l_xx, self.l_u, self.l_uu, self.l_xu]
        l_outputs_str = ["l", "l_x", "l_xx", "l_u", "l_uu", "l_xu"]
        self.loss = cs.Function("loss", l_inputs, l_outputs, l_inputs_str, l_outputs_str)


def symbolic(mass: float, J: NDArray, dt: float) -> SymbolicModel:
    """Create symbolic (CasADi) models for dynamics, observation, and cost of a quadcopter.

    Returns:
        The CasADi symbolic model of the environment.
    """
    # Define states.
    z = MX.sym("z")
    z_dot = MX.sym("z_dot")

    # Set up the dynamics model for a 3D quadrotor.
    nx, nu = 12, 4
    Ixx, Iyy, Izz = J.diagonal()
    J = cs.blockcat([[Ixx, 0.0, 0.0], [0.0, Iyy, 0.0], [0.0, 0.0, Izz]])
    Jinv = cs.blockcat([[1.0 / Ixx, 0.0, 0.0], [0.0, 1.0 / Iyy, 0.0], [0.0, 0.0, 1.0 / Izz]])
    gamma = KM / KF
    # System state variables
    x, y = MX.sym("x"), MX.sym("y")
    x_dot, y_dot = MX.sym("x_dot"), MX.sym("y_dot")
    phi, theta, psi = MX.sym("phi"), MX.sym("theta"), MX.sym("psi")
    p, q, r = MX.sym("p"), MX.sym("q"), MX.sym("r")
    # Rotation matrix transforming a vector in the body frame to the world frame. PyBullet Euler
    # angles use the SDFormat for rotation matrices.
    Rob = csRotXYZ(phi, theta, psi)
    # Define state variables.
    X = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r)
    # Define inputs.
    f1, f2, f3, f4 = MX.sym("f1"), MX.sym("f2"), MX.sym("f3"), MX.sym("f4")
    U = cs.vertcat(f1, f2, f3, f4)

    # From Ch. 2 of Luis, Carlos, and Jérôme Le Ny. "Design of a trajectory tracking
    # controller for a nanoquadcopter." arXiv preprint arXiv:1608.05786 (2016).

    # Defining the dynamics function.
    # We are using the velocity of the base wrt to the world frame expressed in the world frame.
    # Note that the reference expresses this in the body frame.
    oVdot_cg_o = Rob @ cs.vertcat(0, 0, f1 + f2 + f3 + f4) / mass - cs.vertcat(0, 0, GRAVITY)
    pos_ddot = oVdot_cg_o
    pos_dot = cs.vertcat(x_dot, y_dot, z_dot)
    # We use the spin directions (signs) from the mix matrix used in the simulation.
    sx, sy, sz = SIGN_MIX_MATRIX[..., 0], SIGN_MIX_MATRIX[..., 1], SIGN_MIX_MATRIX[..., 2]
    Mb = cs.vertcat(
        ARM_LEN / cs.sqrt(2.0) * (sx[0] * f1 + sx[1] * f2 + sx[2] * f3 + sx[3] * f4),
        ARM_LEN / cs.sqrt(2.0) * (sy[0] * f1 + sy[1] * f2 + sy[2] * f3 + sy[3] * f4),
        gamma * (sz[0] * f1 + sz[1] * f2 + sz[2] * f3 + sz[3] * f4),
    )
    rate_dot = Jinv @ (Mb - (cs.skew(cs.vertcat(p, q, r)) @ J @ cs.vertcat(p, q, r)))
    ang_dot = cs.blockcat(
        [
            [1, cs.sin(phi) * cs.tan(theta), cs.cos(phi) * cs.tan(theta)],
            [0, cs.cos(phi), -cs.sin(phi)],
            [0, cs.sin(phi) / cs.cos(theta), cs.cos(phi) / cs.cos(theta)],
        ]
    ) @ cs.vertcat(p, q, r)
    X_dot = cs.vertcat(
        pos_dot[0], pos_ddot[0], pos_dot[1], pos_ddot[1], pos_dot[2], pos_ddot[2], ang_dot, rate_dot
    )

    Y = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r)

    # Define cost (quadratic form).
    Q, R = MX.sym("Q", nx, nx), MX.sym("R", nu, nu)
    Xr, Ur = MX.sym("Xr", nx, 1), MX.sym("Ur", nu, 1)
    cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)
    # Define dynamics and cost dictionaries.
    dynamics = {"dyn_eqn": X_dot, "obs_eqn": Y, "vars": {"X": X, "U": U}}
    cost = {"cost_func": cost_func, "vars": {"X": X, "U": U, "Xr": Xr, "Ur": Ur, "Q": Q, "R": R}}
    return SymbolicModel(dynamics=dynamics, cost=cost, dt=dt)


def symbolic_from_sim(sim: Sim) -> SymbolicModel:
    """Create a symbolic model from a Sim instance.

    Args:
        sim: A Sim instance.

    Note:
        This only returns a model for a single drone with nominal parameters.

    Warning:
        The model is expected to deviate from the true dynamics when the sim parameters are
        randomized.
    """
    mass, J = sim.defaults["params"].mass[0, 0], sim.defaults["params"].J[0, 0]
    return symbolic(mass, J, 1 / sim.freq)


def csRotXYZ(phi: float, theta: float, psi: float) -> MX:
    """Create a rotation matrix from euler angles.

    This represents the extrinsic X-Y-Z (or quivalently the intrinsic Z-Y-X (3-2-1)) euler angle
    rotation.

    Args:
        phi: roll (or rotation about X).
        theta: pitch (or rotation about Y).
        psi: yaw (or rotation about Z).

    Returns:
        R: casadi Rotation matrix
    """
    rx = cs.blockcat([[1, 0, 0], [0, cs.cos(phi), -cs.sin(phi)], [0, cs.sin(phi), cs.cos(phi)]])
    ry = cs.blockcat(
        [[cs.cos(theta), 0, cs.sin(theta)], [0, 1, 0], [-cs.sin(theta), 0, cs.cos(theta)]]
    )
    rz = cs.blockcat([[cs.cos(psi), -cs.sin(psi), 0], [cs.sin(psi), cs.cos(psi), 0], [0, 0, 1]])
    return rz @ ry @ rx
