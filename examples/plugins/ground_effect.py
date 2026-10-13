"""Model the ground effect of a `cf21B_500` drone as an external force.

The plugin implements Eq. (15) of Shi et al., arXiv:1811.08027. It computes the nominal thrust
from the rotor speeds, applies a height-dependent gain, and adds the extra thrust along the drone's
body z-axis as a world-frame external force.

Run with `--plot` to plot the results, which requires matplotlib from the `tests` environment.
"""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

os.environ["SCIPY_ARRAY_API"] = "1"

import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.sim import Sim
from crazyflow.sim.pipeline import insert_fn_before

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData


# Parameters for cf21B_500
PROPELLER_DIAMETER = 55e-3  # m
MU = 2.0
MIN_HEIGHT = 0.02  # m; Eq. (15) is not valid arbitrarily close to the floor
MAX_GAIN = 2.0  # avoid the model's singularity near the floor

# Descent points
HOVER_HEIGHTS = np.linspace(0.50, 0.02, 15)
SETTLE_DURATION = 10.0  # s
SAMPLE_DURATION = 0.2  # s


def ground_effect_fn(data: SimData) -> SimData:
    rpm = data.states.rotor_vel
    c, b, a = (
        data.params.rpm2thrust[..., 0],
        data.params.rpm2thrust[..., 1],
        data.params.rpm2thrust[..., 2],
    )
    nominal_thrust = jnp.sum(c + b * rpm + a * rpm**2, axis=-1)

    height = jnp.maximum(data.states.pos[..., 2], MIN_HEIGHT)
    gain = 1.0 / (1.0 - MU * (PROPELLER_DIAMETER / (8.0 * height)) ** 2)
    gain = jnp.minimum(gain, MAX_GAIN)
    extra_thrust = nominal_thrust * (gain - 1.0)

    # The force follows the body z thrust axis; ``states.force`` needs world coordinates.
    body_force = jnp.zeros_like(data.states.pos).at[..., 2].set(extra_thrust)
    ground_force = R.from_quat(data.states.quat).apply(body_force)
    return data.replace(states=data.states.replace(force=ground_force))


def measure_hover_points(
    sim: Sim, heights: np.ndarray, render: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Hold at each setpoint and return the mean measured height and thrust command."""
    command = np.zeros((sim.n_worlds, sim.n_drones, 16))
    command[..., 9:13] = R.from_euler("z", 0.0).as_quat()

    settle_steps = int(SETTLE_DURATION * sim.control_freq)
    total_steps = settle_steps + int(SAMPLE_DURATION * sim.control_freq)
    fps = 60

    hover_heights, hover_thrusts = [], []

    for height in heights:
        command[..., 2] = height
        height_samples = []
        thrust_samples = []
        for step in range(total_steps):
            sim.state_control(command)
            sim.step(sim.freq // sim.control_freq)

            if step >= settle_steps:
                height_samples.append(float(sim.data.states.pos[0, 0, 2]))
                # This is the collective force command passed to the motor mixer.
                thrust_samples.append(float(sim.data.controls.force_torque.cmd[0, 0, 0]))
            if render:
                if ((step * fps) % sim.control_freq) < fps:
                    sim.render()

        hover_heights.append(np.mean(height_samples))
        hover_thrusts.append(np.mean(thrust_samples))

    return np.asarray(hover_heights), np.asarray(hover_thrusts)


def main(plot: bool = False, render: bool = False) -> None:

    sim = Sim(n_drones=1, drone="cf21B_500", control="state")

    insert_fn_before(sim.step_pipeline, "integration", ground_effect_fn)
    sim.build_step_fn()

    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=jnp.array([[[0.0, 0.0, HOVER_HEIGHTS[0]]]]))
    )

    hover_heights, hover_thrusts = measure_hover_points(sim, HOVER_HEIGHTS, render=render)

    sim.close()

    if plot:
        plot_results(hover_heights, hover_thrusts)


def plot_results(hover_heights: np.ndarray, hover_thrusts: np.ndarray):
    import matplotlib.pyplot as plt

    plt.scatter(hover_heights, hover_thrusts)
    plt.xlabel("Measured hover height (m)")
    plt.ylabel("Commanded collective thrust (N)")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot the results")
    main(plot=parser.parse_args().plot, render=True)
