"""Example of a simulated IMU as a plugin.

The IMU measures the specific force and the body rates with noise and a drifting gyroscope bias
while the drone flies a figure-eight. The plot compares the readings with the true state.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

os.environ["SCIPY_ARRAY_API"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.sim import Sim
from crazyflow.sim.pipeline import prepend_fn

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData

ACCEL_STD = 0.03  # m/s^2
GYRO_STD = 0.0025  # rad/s
GYRO_BIAS_WALK_STD = 0.002  # rad/s/sqrt(s), inflated such that the bias walk is visible
GYRO_BIAS_MAX = 0.01  # rad/s, the initial bias is uniform in [-max, max]
DURATION = 20.0  # s, one loop of the figure-eight


def simulate_imu(data: SimData) -> SimData:
    """Sample the IMU from the true state."""
    key, accel_key, gyro_key, walk_key = jax.random.split(data.core.rng_key, 4)
    dt = 1 / data.core.freq
    bias = data.plugins["imu_gyro_bias"]
    bias = bias + jax.random.normal(walk_key, bias.shape) * GYRO_BIAS_WALK_STD * dt**0.5
    # Accelerometers measure the specific force in the body frame, f_b = R_wb^T (a_w - g_w)
    acc = (data.states.vel - data.plugins["imu_prev_vel"]) * data.core.freq
    accel = R.from_quat(data.states.quat).apply(acc - data.params.gravity_vec, inverse=True)
    accel = accel + jax.random.normal(accel_key, acc.shape) * ACCEL_STD
    gyro = data.states.ang_vel + bias + jax.random.normal(gyro_key, acc.shape) * GYRO_STD
    imu = {
        "imu_accel": accel,
        "imu_gyro": gyro,
        "imu_gyro_bias": bias,
        "imu_prev_vel": data.states.vel,
    }
    return data.replace(plugins=data.plugins | imu, core=data.core.replace(rng_key=key))


def trajectory(t: float) -> np.ndarray:
    """Return a figure-eight position command."""
    omega = 2 * np.pi / DURATION
    cmd = np.zeros((1, 1, 16))
    cmd[..., :3] = [np.sin(omega * t), 0.75 * np.sin(2 * omega * t), 1.0]
    cmd[..., 9:13] = R.from_euler("z", 0.0).as_quat()
    return cmd


def main(plot: bool = True):
    sim = Sim(control="state")
    states = sim.data.states.replace(pos=trajectory(0.0)[..., :3])
    key, bias_key = jax.random.split(sim.data.core.rng_key)
    zeros = jnp.zeros_like(states.vel)
    bias = jax.random.uniform(bias_key, zeros.shape, minval=-GYRO_BIAS_MAX, maxval=GYRO_BIAS_MAX)
    imu = {"imu_accel": zeros, "imu_gyro": zeros, "imu_gyro_bias": bias, "imu_prev_vel": states.vel}
    sim.data = sim.data.replace(
        states=states, plugins=sim.data.plugins | imu, core=sim.data.core.replace(rng_key=key)
    )
    prepend_fn(sim.step_pipeline, simulate_imu)
    sim.build_default_data()
    sim.build_step_fn()

    log = {key: [] for key in ("vel", "quat", "ang_vel", "accel", "gyro")}
    for i in range(int(2 * DURATION * sim.control_freq)):
        sim.state_control(trajectory(i / sim.control_freq))
        sim.step(sim.freq // sim.control_freq)
        for key in ("vel", "quat", "ang_vel"):
            log[key].append(np.asarray(getattr(sim.data.states, key)[0, 0]))
        log["accel"].append(np.asarray(sim.data.plugins["imu_accel"][0, 0]))
        log["gyro"].append(np.asarray(sim.data.plugins["imu_gyro"][0, 0]))
    sim.close()
    log = {key: np.array(value) for key, value in log.items()}
    # Compare the accelerations without gravity, a_b = R_wb^T a_w, so that the noise is more visible
    rot = R.from_quat(log["quat"])
    acc = np.gradient(log["vel"], 1 / sim.control_freq, axis=0)
    log["acc"] = np.asarray(rot.apply(acc, inverse=True))
    log["accel"] = log["accel"] + np.asarray(rot.apply(sim.data.params.gravity_vec, inverse=True))
    log = {key: value[len(value) // 2 :] for key, value in log.items()}  # Drop the first loop

    accel_error = np.sqrt(np.mean(np.sum((log["accel"] - log["acc"]) ** 2, axis=-1)))
    gyro_error = np.sqrt(np.mean(np.sum((log["gyro"] - log["ang_vel"]) ** 2, axis=-1)))
    print(
        f"Accelerometer error RMS {accel_error:.3f} m/s^2, "
        f"gyroscope error RMS {gyro_error:.4f} rad/s"
    )
    if plot:
        plot_results(log)


def plot_results(log: dict[str, np.ndarray]):
    import matplotlib.pyplot as plt

    fig, (ax_accel, ax_gyro) = plt.subplots(1, 2, figsize=(11, 4.5))
    t = np.linspace(0, DURATION, len(log["gyro"]))
    sensors = (
        (ax_accel, log["acc"], log["accel"], "Accelerometer without gravity", "[m/s$^2$]"),
        (ax_gyro, log["ang_vel"], log["gyro"], "Gyroscope", "[rad/s]"),
    )
    for ax, true, reading, title, unit in sensors:
        readings = [
            ax.plot(t, reading[:, i], lw=0.6, label=f"{x} reading")[0] for i, x in enumerate("xyz")
        ]
        true_lines = ax.plot(t, true, "k", lw=0.8, label="true")
        ax.legend(handles=[true_lines[0], *readings])
        ax.set(title=title, xlabel="Time [s]", ylabel=unit)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
