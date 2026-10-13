"""Simulate an IMU as a plugin.

The accelerometer measures the specific force and the gyroscope the body rates, both with noise and
drifting biases, while the drone flies two loops of a figure-eight. The plot compares the readings
with the true state, skipping the first loop to remove initial transients.

Run with `--plot` to plot the results, which requires matplotlib from the `tests` environment.
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from typing import TYPE_CHECKING, Callable

os.environ["SCIPY_ARRAY_API"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass, field
from scipy.spatial.transform import Rotation as R

from crazyflow.sim import Sim
from crazyflow.sim.pipeline import prepend_fn
from crazyflow.sim.sim import select_dynamics_fn
from crazyflow.utils import CORE_NDIM_KEY

if TYPE_CHECKING:
    from jax import Array, Device

    from crazyflow.sim.data import SimData, SimStateDeriv

ACCEL_STD = 0.03  # m/s^2
GYRO_STD = 0.0025  # rad/s
ACCEL_BIAS_WALK_STD = 1e-3  # m/s^2/sqrt(s)
GYRO_BIAS_WALK_STD = 0.002  # rad/s/sqrt(s), inflated such that the bias walk is visible
GYRO_BIAS_MAX = 0.01  # rad/s, the initial bias is uniform in [-max, max]
PERIOD = 20.0  # s, one loop of the figure-eight
DURATION = 2 * PERIOD  # s, the first loop is dropped to remove initial transients


@dataclass
class IMUData:
    """IMU measurements, biases and noise parameters."""

    accel: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Latest accelerometer measurement as specific force in the body frame."""
    gyro: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Latest gyroscope measurement as angular velocity in the body frame."""
    accel_bias: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Accelerometer bias."""
    gyro_bias: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3)
    """Gyroscope bias."""
    accel_std: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 1)
    """Standard deviation of the accelerometer noise."""
    gyro_std: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 1)
    """Standard deviation of the gyroscope noise."""
    accel_bias_walk_std: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 1)
    """Standard deviation of the accelerometer bias random walk."""
    gyro_bias_walk_std: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 1)
    """Standard deviation of the gyroscope bias random walk."""

    @staticmethod
    def create(
        n_worlds: int,
        n_drones: int,
        accel_std: float,
        gyro_std: float,
        accel_bias_walk_std: float,
        gyro_bias_walk_std: float,
        device: Device,
    ) -> IMUData:
        """Create IMU data with zero measurements and biases."""
        zeros = jnp.zeros((n_worlds, n_drones, 3), device=device)
        std_shape = (n_worlds, n_drones, 1)
        return IMUData(
            accel=zeros,
            gyro=zeros,
            accel_bias=zeros,
            gyro_bias=zeros,
            accel_std=jnp.full(std_shape, accel_std, device=device),
            gyro_std=jnp.full(std_shape, gyro_std, device=device),
            accel_bias_walk_std=jnp.full(std_shape, accel_bias_walk_std, device=device),
            gyro_bias_walk_std=jnp.full(std_shape, gyro_bias_walk_std, device=device),
        )


def simulate_imu(data: SimData, deriv_fn: Callable[[SimData], SimStateDeriv]) -> SimData:
    """Sample the IMU at the current state."""
    imu: IMUData = data.plugins["imu"]
    key, accel_key, gyro_key, accel_walk_key, gyro_walk_key = jax.random.split(data.core.rng_key, 5)
    sqrt_dt = (1 / data.core.freq) ** 0.5
    accel_walk = jax.random.normal(accel_walk_key, imu.accel_bias.shape) * imu.accel_bias_walk_std
    accel_bias = imu.accel_bias + accel_walk * sqrt_dt
    gyro_walk = jax.random.normal(gyro_walk_key, imu.gyro_bias.shape) * imu.gyro_bias_walk_std
    gyro_bias = imu.gyro_bias + gyro_walk * sqrt_dt
    # Accelerometers measure the specific force in the body frame, f_b = R_wb^T (a_w - g_w)
    acc = deriv_fn(data).acc
    accel = R.from_quat(data.states.quat).apply(acc - data.params.gravity_vec, inverse=True)
    accel = accel + accel_bias + jax.random.normal(accel_key, acc.shape) * imu.accel_std
    gyro = data.states.ang_vel + gyro_bias + jax.random.normal(gyro_key, acc.shape) * imu.gyro_std
    imu = imu.replace(accel=accel, gyro=gyro, accel_bias=accel_bias, gyro_bias=gyro_bias)
    return data.replace(plugins=data.plugins | {"imu": imu}, core=data.core.replace(rng_key=key))


def add_imu(
    sim: Sim,
    accel_std: float = ACCEL_STD,
    gyro_std: float = GYRO_STD,
    accel_bias_walk_std: float = ACCEL_BIAS_WALK_STD,
    gyro_bias_walk_std: float = GYRO_BIAS_WALK_STD,
):
    """Add the IMU data to the plugins and sample the IMU at the start of each step.

    The caller rebuilds the default data and the step function.
    """
    imu = IMUData.create(
        sim.n_worlds,
        sim.n_drones,
        accel_std,
        gyro_std,
        accel_bias_walk_std,
        gyro_bias_walk_std,
        sim.device,
    )
    sim.data = sim.data.replace(plugins=sim.data.plugins | {"imu": imu})
    simulate_fn = partial(simulate_imu, deriv_fn=select_dynamics_fn(sim.dynamics))
    prepend_fn(sim.step_pipeline, simulate_fn, name="simulate_imu")


def trajectory(t: float) -> np.ndarray:
    """Return a figure-eight position command."""
    omega = 2 * np.pi / PERIOD
    cmd = np.zeros((1, 1, 16))
    cmd[..., :3] = [np.sin(omega * t), 0.75 * np.sin(2 * omega * t), 1.0]
    cmd[..., 9:13] = R.from_euler("z", 0.0).as_quat()
    return cmd


def main(plot: bool = False):
    sim = Sim(control="state")
    states = sim.data.states.replace(pos=trajectory(0.0)[..., :3])
    add_imu(sim)
    imu: IMUData = sim.data.plugins["imu"]
    key, bias_key = jax.random.split(sim.data.core.rng_key)
    bias = jax.random.uniform(
        bias_key, imu.gyro_bias.shape, minval=-GYRO_BIAS_MAX, maxval=GYRO_BIAS_MAX
    )
    sim.data = sim.data.replace(
        states=states,
        plugins=sim.data.plugins | {"imu": imu.replace(gyro_bias=bias)},
        core=sim.data.core.replace(rng_key=key),
    )
    sim.build_default_data()
    sim.build_step_fn()

    log = {key: [] for key in ("vel", "quat", "ang_vel", "accel", "gyro")}
    for i in range(int(DURATION * sim.control_freq)):
        sim.state_control(trajectory(i / sim.control_freq))
        sim.step(sim.freq // sim.control_freq)
        log["vel"].append(sim.data.states.vel[0, 0])
        log["quat"].append(sim.data.states.quat[0, 0])
        log["ang_vel"].append(sim.data.states.ang_vel[0, 0])
        log["accel"].append(sim.data.plugins["imu"].accel[0, 0])
        log["gyro"].append(sim.data.plugins["imu"].gyro[0, 0])
    sim.close()
    log = {key: np.array(value) for key, value in log.items()}
    # Compare the accelerations without gravity, a_b = R_wb^T a_w, so that the noise is more visible
    rot = R.from_quat(log["quat"])
    acc = np.gradient(log["vel"], 1 / sim.control_freq, axis=0)
    log["acc"] = rot.apply(acc, inverse=True)
    log["accel"] = log["accel"] + rot.apply(sim.data.params.gravity_vec, inverse=True)
    # Drop the first loop
    log = {key: value[int(PERIOD * sim.control_freq) :] for key, value in log.items()}

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
    t = np.linspace(0, PERIOD, len(log["gyro"]))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot the results")
    main(plot=parser.parse_args().plot)
