"""Data preprocessing functions for dynamics identification.

Contains helpers to filter states and compute derivatives using State Variable Filters.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import bilinear, butter, filtfilt, lfilter, lfiltic
from scipy.spatial.transform import Rotation as R

from crazyflow.dynamics.utils.rotation import rpy_rates2ang_vel

if TYPE_CHECKING:
    from crazyflow._typing import Array  # To be changed to array_api_typing later


def preprocessing(data: dict[str, Array]) -> dict[str, Array]:
    """Applies preprocessing to collected data.

    The preprocessing includes outlier detection and interpolation, normalizing orientation
    (assuming hover at start), calculating rpy from quaternions, and calculating rotational error.

    Args:
        data: The raw data dictionary containing time [s], pos [m], quat, cmd_rpy [rad], cmd_f [N].

    Returns:
        The same dict with the following keys added or modified:

        * ``"dt"``: Time step array, shape ``(N-1,)``.
        * ``"time"``: Time shifted so ``time[0] == 0``.
        * ``"quat"``: Quaternions corrected so the initial attitude is identity.
        * ``"rpy"``: Roll/pitch/yaw in radians, shape ``(N, 3)``.
        * ``"z_axis"``: Body z-axis in world frame, shape ``(N, 3)``.
        * ``"eR"``: Rotation error vector (vee of skew-symmetric error matrix), shape ``(N, 3)``.
        * ``"eR_vec"``: Rotation error as rotation vector, shape ``(N, 3)``.
    """
    data["dt"] = np.diff(data["time"])
    data["time"] -= data["time"][0]
    ### Outlier detection + interpolation
    b, a = butter(N=4, Wn=1, fs=1 / np.mean(data["dt"]))
    residuals = data["pos"] - filtfilt(b, a, data["pos"], axis=0)
    outliers = np.abs(residuals) > 0.3
    outliers = np.sum(outliers, axis=-1)
    is_outlier = np.asarray(outliers).astype(bool)
    n_outliers = np.sum(outliers)
    # TODO also check quat for outliers!

    if n_outliers > 0:
        warnings.warn(f"{n_outliers} outliers detected. Interpolating")
        time_good = data["time"][~is_outlier]
        pos_good = data["pos"][~is_outlier]
        quat_good = data["quat"][~is_outlier]
        interp_pos = interp1d(time_good, pos_good, axis=0, fill_value="extrapolate")
        interp_quat = interp1d(time_good, quat_good, axis=0, fill_value="extrapolate")
        data["pos"][is_outlier] = interp_pos(data["time"][is_outlier])
        data["quat"][is_outlier] = interp_quat(data["time"][is_outlier])

    ### Normalizing orientation (assuming zero at start) and calculating rpy
    time_span = 0.1
    time_index = int(time_span / np.mean(data["dt"]))
    quat_avg = np.mean(data["quat"][:time_index], axis=0)
    quat_avg /= np.linalg.norm(quat_avg)
    rot_corr = R.from_quat(quat_avg).inv()
    rot = rot_corr * R.from_quat(data["quat"])
    data["quat"] = rot.as_quat()
    data["rpy"] = rot.as_euler("xyz")
    data["z_axis"] = rot.inv().as_matrix()[..., -1, :]

    ### Rotational error
    rot = R.from_quat(data["quat"])
    R_act = rot.as_matrix()
    R_des = R.from_euler("xyz", data["cmd_rpy"], degrees=False).as_matrix()
    eRM = np.matmul(np.swapaxes(R_des, -1, -2), R_act) - np.matmul(
        np.swapaxes(R_act, -1, -2), R_des
    )
    data["eR"] = np.stack(
        (eRM[..., 2, 1], eRM[..., 0, 2], eRM[..., 1, 0]), axis=-1
    )  # vee operator (SO3 to R3)
    data["eR_vec"] = (rot.inv() * R.from_euler("xyz", data["cmd_rpy"], degrees=False)).as_rotvec()

    return data


def derivatives_svf(data: dict[str, Array]) -> dict[str, Array]:
    """Apply a State Variable Filter (SVF) to compute smoothed signals and their time derivatives.

    Filters position, attitude (RPY), and command signals with separate corner frequencies (6 Hz for
    translation, 8 Hz for rotation) and computes up to third-order time derivatives.  All output
    keys are prefixed with ``"SVF_"``.

    Args:
        data: Dict produced by [preprocessing][crazyflow.dynamics.utils.data_utils.preprocessing].
            Must contain ``"pos"``, ``"rpy"``, ``"time"``, ``"cmd_f"``, and ``"cmd_rpy"``.

    Returns:
        The same dict with the following ``"SVF_"`` keys added:

        * ``"SVF_pos"``, ``"SVF_vel"``, ``"SVF_acc"``, ``"SVF_jerk"``: Filtered position and its
            first three derivatives.
        * ``"SVF_rpy"``, ``"SVF_drpy"``, ``"SVF_ddrpy"``, ``"SVF_dddrpy"``: Filtered roll/pitch/yaw
            and its first three derivatives.
        * ``"SVF_quat"``: Quaternion computed from ``SVF_rpy``.
        * ``"SVF_z_axis"``: Body z-axis in world frame computed from ``SVF_rpy``.
        * ``"SVF_ang_vel"``, ``"SVF_ang_acc"``, ``"SVF_ang_jerk"``: Angular
            velocity/acceleration/jerk in body frame.
        * ``"SVF_cmd_f"``: Filtered collective thrust command.
        * ``"SVF_cmd_rpy"``: Filtered roll/pitch/yaw command.
        * ``"SVF_eR"``, ``"SVF_eR_vec"``: Rotation error between actual and commanded attitude.
    """
    # Important: Don't mix with unfiltered signals (also for input!)
    if data is None:
        return None

    svf_linear = state_variable_filter(data["pos"].T, data["time"], f_c=6, N_deriv=3)
    data["SVF_pos"] = svf_linear[:, 0].T
    data["SVF_vel"] = svf_linear[:, 1].T
    data["SVF_acc"] = svf_linear[:, 2].T
    data["SVF_jerk"] = svf_linear[:, 3].T

    svf_rotational = state_variable_filter(data["rpy"].T, data["time"], f_c=8, N_deriv=3)
    data["SVF_rpy"] = svf_rotational[:, 0].T
    data["SVF_drpy"] = svf_rotational[:, 1].T
    data["SVF_ddrpy"] = svf_rotational[:, 2].T
    data["SVF_dddrpy"] = svf_rotational[:, 3].T
    rot = R.from_euler("xyz", data["SVF_rpy"])
    data["SVF_quat"] = rot.as_quat()
    data["SVF_z_axis"] = rot.inv().as_matrix()[..., -1, :]
    data["SVF_ang_vel"] = rpy_rates2ang_vel(data["SVF_quat"], data["SVF_drpy"])
    data["SVF_ang_acc"] = rpy_rates2ang_vel(data["SVF_quat"], data["SVF_ddrpy"])
    data["SVF_ang_jerk"] = rpy_rates2ang_vel(data["SVF_quat"], data["SVF_dddrpy"])

    svf_input_f = state_variable_filter(data["cmd_f"], data["time"], f_c=6, N_deriv=3)
    data["SVF_cmd_f"] = svf_input_f[0]
    svf_input_rpy = state_variable_filter(data["cmd_rpy"].T, data["time"], f_c=8, N_deriv=3)
    data["SVF_cmd_rpy"] = svf_input_rpy[:, 0].T

    R_act = rot.as_matrix()
    rot_cmd = R.from_euler("xyz", data["SVF_cmd_rpy"])
    R_des = rot_cmd.as_matrix()
    eRM = np.matmul(np.swapaxes(R_des, -1, -2), R_act) - np.matmul(
        np.swapaxes(R_act, -1, -2), R_des
    )
    data["SVF_eR"] = np.stack(
        (eRM[..., 2, 1], eRM[..., 0, 2], eRM[..., 1, 0]), axis=-1
    )  # vee operator (SO3 to R3)
    data["SVF_eR_vec"] = (rot.inv() * rot_cmd).as_rotvec()

    return data


def state_variable_filter(y: Array, t: Array, f_c: float = 1, N_deriv: int = 2) -> Array:
    """A state variable filter that low pass filters the signal and computes the derivatives.

    Args:
        y: The signal to be filtered. Can be 1D (signal_length) or 2D (batch_size, signal_length).
        t: The time values for the signal. Optimally fixed sampling frequency.
        f_c: Corner frequency of the filter in Hz. Defaults to 1.
        N_deriv: Number of derivatives to be computed. Defaults to 2.

    Returns:
        Array: The filtered signal and its derivatives. Shape (batch_size, N_deriv+1, signal_length)
    """
    if y.ndim == 1:
        y = y[None, :]  # Add batch dimension if single signal
    batch_size, signal_length = y.shape

    # The filter needs to have a minimum of two extra states
    # One for the filtered input signal and one for the actual filter
    N_ord = N_deriv + 2
    omega_c = 2 * np.pi * f_c
    f_s = 1 / np.mean(np.diff(t))

    b, a = butter(N=N_ord, Wn=omega_c, analog=True)
    b_dig, a_dig = bilinear(b, a, fs=f_s)
    a_flipped = np.flip(a)

    def _f(t: Array, x: Array, u: Array) -> Array:
        x_dot = []
        x_dot_last = 0
        # The first states are a simple integrator chain
        for i in np.arange(1, N_ord):
            x_dot.append(x[i])
        # Last state uses the filter coefficients
        for i in np.arange(0, N_ord):
            x_dot_last -= a_flipped[i] * x[i]
        x_dot_last += b[0] * u(t)
        x_dot.append(x_dot_last)

        return x_dot

    results = np.zeros((batch_size, N_deriv + 1, signal_length))

    for i in range(batch_size):
        # Define input
        # Prefilter input backwards to remove time shift
        # Add padding to remove filter oscillations in data
        pad = 100
        y_backwards = np.flip(y[i], axis=-1)
        y_backwards_padded = np.concatenate([np.ones(pad) * y_backwards[0], y_backwards])
        zi = lfiltic(
            b_dig, a_dig, y_backwards_padded, x=y_backwards_padded
        )  # initial filter conditions
        y_backwards, _ = lfilter(b_dig, a_dig, y_backwards_padded, axis=-1, zi=zi)
        u = interp1d(
            t, np.flip(y_backwards[pad:], axis=-1), kind="linear", fill_value="extrapolate"
        )

        # Solve system with initial conditions
        x0 = np.zeros(N_ord)
        x0[0] = y[i, 0]
        sol = solve_ivp(_f, [t[0], t[-1]], x0, t_eval=t, args=(u,))

        results[i] = sol.y[:-1]  # Last state is not of interest

    return results.squeeze()  # Remove batch dim if not needed
