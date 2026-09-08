"""This module contains functions to identify so_rpy dynamics from data."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Callable, Literal

import jax  # noqa: I001
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from crazyflow.dynamics.so_rpy_rotor_drag import dynamics as dynamics_so_rpy_rotor_drag
from crazyflow.dynamics.so_rpy_rotor_drag.dynamics import dynamics_euler

if TYPE_CHECKING:
    from crazyflow._typing import Array  # To be changed to array_api_typing later

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove unused parameters by partial application to make code cleaner
dynamics_translation = partial(
    dynamics_so_rpy_rotor_drag,
    J=jnp.zeros((3, 3)),
    J_inv=jnp.zeros((3, 3)),
    rpy_coef=jnp.zeros(3),
    rpy_rates_coef=jnp.zeros(3),
    cmd_rpy_coef=jnp.zeros(3),
)

dynamics_rotation = partial(
    dynamics_euler,
    mass=0.1,
    gravity_vec=jnp.array([0, 0, -9.81]),
    thrust_time_coef=0.1,
    acc_coef=0.0,
    drag_matrix=jnp.zeros((3, 3)),
    cmd_f_coef=1.0,
)


def _rmse(x1: Array, x2: Array) -> float:
    return np.sqrt(np.mean((x1 - x2) ** 2))


def _r2(x1: Array, x2: Array) -> float:
    return 1 - np.sum((x1 - x2) ** 2) / np.sum((x1 - np.mean(x1)) ** 2)


# region translation
def _simulate_system_translation(
    quat: Array, vel: Array, cmd_f: Array, t: Array, params: Array, constants: dict[str, Array]
) -> Array:
    """Simulate the dynamical system and return the derivatives.

    Args:
        quat: Orientation (quaternion) of the drone (N, 4)
        vel: Velocity of the drone (N, 3)
        cmd_f: Commanded thrust (N,)
        t: Time samples (N,)
        params: Dynamics parameters [cmd_f_coef, thrust_time_coef, drag_xy_coef, drag_z_coef]
        constants: Additional constants (mass, gravity_vec, etc.)

    returns: predicted acceleration (N, 3)
    """
    N = vel.shape[0]
    zeros_Nx3 = jnp.zeros((N, 3))
    zeros_Nx4 = jnp.zeros((N, 4))
    cmd = jnp.concatenate([zeros_Nx3, cmd_f[..., None]], axis=-1)
    thrust0 = jnp.array([cmd_f[0]])  # Assuming hover at start, i.e., actual = command
    dt = jnp.diff(t)

    # Rollout thrust dynamics
    def _step_thrust(carry: Array, inputs: tuple) -> tuple:
        dt_step, u = inputs

        _, _, _, _, rotor_vel_dot = dynamics_translation(
            pos=zeros_Nx3,
            quat=quat,
            vel=vel,
            ang_vel=zeros_Nx3,
            cmd=u,
            rotor_vel=carry,
            mass=constants["mass"],
            gravity_vec=constants["gravity_vec"],
            thrust_time_coef=params[1],
            acc_coef=0.0,
            drag_matrix=jnp.diag(jnp.array([params[2], params[2], params[3]])),
            cmd_f_coef=params[0],
        )
        x_next = jnp.where(params[1] == 0.0, u[-1], carry + rotor_vel_dot * dt_step)
        return x_next, x_next

    _, thrusts = jax.lax.scan(_step_thrust, thrust0, (dt, cmd[:-1]))
    # prepend thrust0 to match length
    thrusts = jnp.squeeze(thrusts)
    thrusts = jnp.concat([thrust0, thrusts], axis=0)

    # Rollout linear dynamics (vectorized)
    _, _, acc, _, _ = dynamics_translation(
        pos=zeros_Nx3,
        quat=quat,
        vel=vel,
        ang_vel=zeros_Nx3,
        cmd=zeros_Nx4,
        rotor_vel=thrusts[..., None],
        mass=constants["mass"],
        gravity_vec=constants["gravity_vec"],
        thrust_time_coef=params[1],
        acc_coef=0.0,
        drag_matrix=jnp.diag(jnp.array([params[2], params[2], params[3]])),
        cmd_f_coef=params[0],
    )
    return acc


def _build_residuals_fun_translation(
    dynamics: Literal["so_rpy", "so_rpy_rotor", "so_rpy_rotor_drag"],
) -> tuple[Callable, Callable]:
    """Build residual function for the given dynamics type."""

    def _residuals_trans(
        params: Array,
        quat: Array,
        vel: Array,
        cmd_f: Array,
        t: Array,
        constants: dict[str, Array],
        acc_observed: Array,
    ) -> Array:
        acc = _simulate_system_translation(quat, vel, cmd_f, t, params, constants)
        return jnp.linalg.norm(acc_observed - acc, axis=-1)

    # JAX analytic Jacobian
    jac_fun = jax.jacfwd(_residuals_trans)  # Jacobian w.r.t. first arg (params)
    jac_fun = jax.jit(jac_fun)

    def _residual_fun_trans(
        params: Array,
        quat: Array,
        vel: Array,
        cmd_f: Array,
        t: Array,
        constants: dict[str, Array],
        acc_observed: Array,
    ) -> Callable:
        match dynamics:  # Dummy values for other params
            case "so_rpy":
                params_jnp = jnp.array([params[0], 0.0, 0.0, 0.0])
            case "so_rpy_rotor":
                params_jnp = jnp.array([params[0], params[1], 0.0, 0.0])
            case "so_rpy_rotor_drag":
                params_jnp = jnp.array([params[0], params[1], params[2], params[3]])
            case _:
                raise ValueError(f"Unknown dynamics type: {dynamics}")
        return jax.device_get(
            _residuals_trans(params_jnp, quat, vel, cmd_f, t, constants, acc_observed)
        )

    def _residual_fun_trans_jac(
        params: Array,
        quat: Array,
        vel: Array,
        cmd_f: Array,
        t: Array,
        constants: dict[str, Array],
        acc_observed: Array,
    ) -> Callable:
        match dynamics:  # Dummy values for other params
            case "so_rpy":
                params_jnp = jnp.array([params[0], 0.0, 0.0, 0.0])
            case "so_rpy_rotor":
                params_jnp = jnp.array([params[0], params[1], 0.0, 0.0])
            case "so_rpy_rotor_drag":
                params_jnp = jnp.array([params[0], params[1], params[2], params[3]])
            case _:
                raise ValueError(f"Unknown dynamics type: {dynamics}")
        return jax.device_get(jac_fun(params_jnp, quat, vel, cmd_f, t, constants, acc_observed))

    return _residual_fun_trans, _residual_fun_trans_jac


def sys_id_translation(
    dynamics: Literal["so_rpy", "so_rpy_rotor", "so_rpy_rotor_drag"],
    mass: float,
    data: dict[str, Array],
    data_validation: dict[str, Array] | None = None,
    gravity: Array = np.array([0, 0, -9.81]),
    verbose: int = 0,
    plot: bool = False,
) -> dict[str, Array]:
    """Identify the translational part of the so_rpy dynamics from data.

    Args:
        dynamics: Dynamics type to identify.
        mass: Mass of the drone.
        data: Training data containing time, and the SVF values of vel, acc, quat, cmd_f.
        data_validation: Optional validation data containing the same fields as data.
        gravity: Gravity vector in world frame, i.e., [0, 0, -9.81].
        verbose: Verbosity level for the optimizer from 0 to 2.
        plot: Whether to plot the results.

    Returns: Identified dynamics parameters.
    """
    theta0 = [1.0, 1.0, 0.0, 0.0]
    method = "trf"
    xtol, ftol, gtol = 1e-10, 1e-10, 1e-10
    constants = {"mass": mass, "gravity_vec": gravity}
    # Convert the data to jnp arrays for use with jax
    t = jnp.array(data["time"])
    vel = jnp.array(data["SVF_vel"])
    acc = jnp.array(data["SVF_acc"])
    quat = jnp.array(data["SVF_quat"])
    cmd_f = jnp.array(data["SVF_cmd_f"])

    # Identification
    residual_fun_trans, residual_fun_trans_jac = _build_residuals_fun_translation(dynamics)
    res = least_squares(
        residual_fun_trans,
        x0=theta0,
        jac=residual_fun_trans_jac,
        args=(quat, vel, cmd_f, t, constants, acc),
        method=method,
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
        verbose=verbose,
    )

    theta = res.x
    params = {"cmd_f_coef": theta[0]}
    if "rotor" in dynamics:
        params["thrust_time_coef"] = theta[1]
    else:
        theta[1] = 0.0
    if "drag" in dynamics:
        params["drag_xy_coef"] = theta[2]
        params["drag_z_coef"] = theta[3]
    else:
        theta[2] = 0.0
        theta[3] = 0.0

    acc_pred = _simulate_system_translation(quat, vel, cmd_f, t, theta, constants)
    if data_validation is not None:
        t_valid = jnp.array(data_validation["time"])
        vel_valid = jnp.array(data_validation["SVF_vel"])
        acc_valid = jnp.array(data_validation["SVF_acc"])
        quat_valid = jnp.array(data_validation["SVF_quat"])
        cmd_f_valid = jnp.array(data_validation["SVF_cmd_f"])
        acc_pred_valid = _simulate_system_translation(
            quat_valid, vel_valid, cmd_f_valid, t_valid, theta, constants
        )

    # Report
    txt = f"\n=== Stats {dynamics} ==="
    txt += f"\nParameters: {params=}"
    txt += f"\nTraining success={res.success}, results:"
    txt += f"\nRMSE={_rmse(acc, acc_pred):.6f}"
    txt += f"\nR^2={_r2(acc, acc_pred):.4f}"
    if data_validation is not None:
        txt += "\nValidation results:"
        txt += f"\nRMSE={_rmse(acc_valid, acc_pred_valid):.6f}"
        txt += f"\nR^2={_r2(acc_valid, acc_pred_valid):.4f}"
    logger.info(txt)

    # Plotting
    if plot:
        # Plot acceleration
        _, axs = plt.subplots(2 if data_validation is not None else 1, 1, figsize=(12, 5))
        axs = np.atleast_1d(axs)

        # Training data subplot
        axs[0].plot(t, acc, label="Measured acc")
        axs[0].plot(t, acc_pred, "--", label="Predicted acc")
        axs[0].set_xlabel("Time [s]")
        axs[0].set_ylabel("Output")

        # Validation data subplot
        if data_validation is not None:
            axs[1].plot(t_valid, acc_valid, label="Measured acc (valid)")
            axs[1].plot(t_valid, acc_pred_valid, "--", label="Predicted acc (valid)")
            axs[1].set_xlabel("Time [s]")
            axs[1].set_ylabel("Output")

        for ax in axs.flat:
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()

        # Plot commanded thrust vs actual thrust
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
        obs_f = np.linalg.norm((acc - constants["gravity_vec"]) * constants["mass"], axis=-1)
        ax.scatter(cmd_f, obs_f, label="Measured")
        cmd_thrust_lin = np.linspace(0.9 * np.min(cmd_f), 1.1 * np.max(cmd_f), 1000)
        ax.plot(cmd_thrust_lin, theta[0] * cmd_thrust_lin, label="Fit")
        ax.set_xlabel("Commanded Thrust [N]")
        ax.set_ylabel("Actual Thrust [N]")
        ax.set_xlim(0.9 * np.min(cmd_f), 1.1 * np.max(cmd_f))
        ax.set_ylim(0.9 * np.min(obs_f), 1.1 * np.max(obs_f))

        plt.tight_layout()
        plt.show()

    return params


# region rotation
def _simulate_system_rotation(cmd_rpy: Array, t: Array, params: Array) -> Array:
    """Simulate the 2nd-order system and return the trajectory.

    Args:
        cmd_rpy: Commanded orientation (N, 3)
        t: Time samples (N,)
        params: Dynamics parameters [rpy_coef, rpy_rates_coef, cmd_rpy_coef]

    returns: predicted acceleration (N, 3)
    """
    cmd = jnp.concatenate([cmd_rpy, jnp.zeros((cmd_rpy.shape[0], 1))], axis=-1)
    dt = jnp.diff(t)
    x0 = jnp.zeros((2, 3))  # rpy, rpy_rates

    def _step_so_system(carry: Array, inputs: Array) -> tuple:
        dt_step, cmd = inputs
        rpy_coef = jnp.array([params[0], params[0], params[1]])
        rpy_rates_coef = jnp.array([params[2], params[2], params[3]])
        cmd_rpy_coef = jnp.array([params[4], params[4], params[5]])
        rpy, rpy_rates = carry[0], carry[1]

        _, _, _, drpy_rates, _ = dynamics_rotation(
            pos=jnp.array([0.0, 0.0, 0.0]),
            rpy=rpy,
            vel=jnp.array([0.0, 0.0, 0.0]),
            rpy_rates=rpy_rates,
            cmd=cmd,
            rotor_vel=jnp.array([0.0]),
            rpy_coef=rpy_coef,
            rpy_rates_coef=rpy_rates_coef,
            cmd_rpy_coef=cmd_rpy_coef,
        )

        ### Integration
        next_rpy = rpy + rpy_rates * dt_step
        next_rpy_rates = rpy_rates + drpy_rates * dt_step
        x_next = jnp.stack([next_rpy, next_rpy_rates], axis=0)
        return x_next, x_next

    _, xs = jax.lax.scan(_step_so_system, x0, (dt, cmd[:-1]))
    # prepend x0 to match length
    xs = jnp.vstack([jnp.array(x0)[None, :], xs])
    rpy_hat = xs[:, 0]  # output y = x1
    return rpy_hat


def _build_residuals_fun_rotation() -> tuple[Callable, Callable]:
    """Build residual function for the given dynamics type."""

    def _residuals_rot(params: Array, cmd_rpy: Array, t: Array, rpy_observed: Array) -> Array:
        rpy = _simulate_system_rotation(cmd_rpy, t, params)
        # return jnp.linalg.norm(rpy_observed - rpy, axis=-1)
        return jnp.reshape(rpy_observed - rpy, (-1,))

    # JAX analytic Jacobian
    jac_fun = jax.jacfwd(_residuals_rot)  # Jacobian w.r.t. first arg (params)
    jac_fun = jax.jit(jac_fun)

    def _residual_fun_rot(params: Array, cmd_rpy: Array, t: Array, rpy_observed: Array) -> Callable:
        residuals = jax.jit(_residuals_rot)
        return jax.device_get(residuals(params, cmd_rpy, t, rpy_observed))

    def _residual_fun_rot_jac(
        params: Array, cmd_rpy: Array, t: Array, rpy_observed: Array
    ) -> Callable:
        return jax.device_get(jac_fun(params, cmd_rpy, t, rpy_observed))

    return _residual_fun_rot, _residual_fun_rot_jac


def sys_id_rotation(
    data: dict[str, Array],
    data_validation: dict[str, Array] | None = None,
    verbose: int = 0,
    plot: bool = False,
) -> dict[str, Array]:
    """Identify the rotational part of the so_rpy dynamics from data.

    Args:
        data: Training data containing time, and the SVF values of rpy [rad], cmd_rpy [rad].
        data_validation: Optional validation data containing the same fields as data.
        verbose: Verbosity level for the optimizer from 0 to 2.
        plot: Whether to plot the results.

    Returns: Identified dynamics parameters.
    """
    # theta includes the values for roll/pitch (same value) and yaw
    theta0 = np.array([-10.0, -10.0, -1.0, -1.0, 10.0, 10.0])  # ry, ry_rates, cmd_ry
    method = "trf"
    xtol, ftol, gtol = 1e-10, 1e-10, 1e-10
    t = jnp.array(data["time"])
    rpy = jnp.array(data["SVF_rpy"])
    cmd_rpy = jnp.array(data["SVF_cmd_rpy"])
    if data_validation is not None:
        t_valid = jnp.array(data_validation["time"])
        rpy_valid = jnp.array(data_validation["SVF_rpy"])
        cmd_rpy_valid = jnp.array(data_validation["SVF_cmd_rpy"])

    # Identification
    residual_fun_rot, residual_fun_rot_jac = _build_residuals_fun_rotation()
    res = least_squares(
        residual_fun_rot,
        x0=theta0,
        jac=residual_fun_rot_jac,
        args=(cmd_rpy, t, rpy),
        method=method,
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
        verbose=verbose,
    )
    theta = res.x

    rpy_coef = np.array([theta[0], theta[0], theta[1]])
    rpy_rates_coef = np.array([theta[2], theta[2], theta[3]])
    cmd_rpy_coef = np.array([theta[4], theta[4], theta[5]])
    params = {"rpy_coef": rpy_coef, "rpy_rates_coef": rpy_rates_coef, "cmd_rpy_coef": cmd_rpy_coef}

    rpy_pred = _simulate_system_rotation(cmd_rpy, t, theta)
    if data_validation is not None:
        rpy_pred_valid = _simulate_system_rotation(cmd_rpy_valid, t_valid, theta)

    # Report
    txt = "\n=== Stats roll & pitch ==="
    txt += f"\nEstimated:  {rpy_coef=}, {rpy_rates_coef=}, {cmd_rpy_coef=}"
    txt += f"\nTraining success={res.success}, results:"
    txt += f"\nRMSE={_rmse(rpy, rpy_pred):.6f}"
    txt += f"\nR^2={_r2(rpy, rpy_pred):.4f}"

    if data_validation is not None:
        txt += "\nValidation results:"
        txt += f"\nRMSE roll={_rmse(rpy_valid[..., 0], rpy_pred_valid[..., 0]):.6f}"
        txt += f"\nRMSE pitch={_rmse(rpy_valid[..., 1], rpy_pred_valid[..., 1]):.6f}"
        txt += f"\nR^2 roll={_r2(rpy_valid[..., 0], rpy_pred_valid[..., 0]):.4f}"
        txt += f"\nR^2 pitch={_r2(rpy_valid[..., 1], rpy_pred_valid[..., 1]):.4f}"
    logger.info(txt)

    # Plotting
    if plot:
        _, axs = plt.subplots(3, 2 if data_validation is not None else 1, figsize=(20, 12))
        if data_validation is None:
            axs = axs[:, None]
        plt.suptitle("RPY dynamics fit")

        axs[0, 0].plot(t, rpy[..., 0], label="Measured roll")
        axs[0, 0].plot(t, rpy_pred[..., 0], "--", label="Predicted roll")
        axs[0, 0].set_ylabel("Roll [rad]")

        axs[1, 0].plot(t, rpy[..., 1], label="Measured pitch")
        axs[1, 0].plot(t, rpy_pred[..., 1], "--", label="Predicted pitch")
        axs[1, 0].set_ylabel("Pitch [rad]")

        axs[2, 0].plot(t, rpy[..., 2], label="Measured yaw")
        axs[2, 0].plot(t, rpy_pred[..., 2], "--", label="Predicted yaw")
        axs[2, 0].set_xlabel("Time [s]")
        axs[2, 0].set_ylabel("Yaw [rad]")

        if data_validation is not None:
            axs[0, 1].plot(t_valid, rpy_valid[..., 0], label="Measured roll (valid)")
            axs[0, 1].plot(t_valid, rpy_pred_valid[..., 0], "--", label="Predicted roll (valid)")
            axs[0, 1].set_ylabel("Roll [rad]")

            axs[1, 1].plot(t_valid, rpy_valid[..., 1], label="Measured pitch (valid)")
            axs[1, 1].plot(t_valid, rpy_pred_valid[..., 1], "--", label="Predicted pitch (valid)")
            axs[1, 1].set_ylabel("Pitch [rad]")

            axs[2, 1].plot(t_valid, rpy_valid[..., 2], label="Measured yaw (valid)")
            axs[2, 1].plot(t_valid, rpy_pred_valid[..., 2], "--", label="Predicted yaw (valid)")
            axs[2, 1].set_xlabel("Time [s]")
            axs[2, 1].set_ylabel("Yaw [rad]")

        for ax in axs.flat:
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()

    return params
