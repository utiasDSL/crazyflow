"""Fly straight up with rotor velocity control and compare the integrators.

All rotors get the same constant command above hover, so the drone accelerates straight up. The
plot shows the height difference of every integrator to the default one.

Run with `--plot` to plot the results, which requires matplotlib from the `tests` environment.
"""

import argparse

import numpy as np

from crazyflow.control import Control
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.sim import Sim
from crazyflow.sim.integration import Integrator

DURATION = 2.0


def main(plot: bool = False):
    heights = {}
    for integrator in Integrator:
        sim = Sim(control=Control.rotor_vel, integrator=integrator)
        sim.reset()
        thrust = 1.2 * sim.data.params.mass * -sim.data.params.gravity_vec[-1]
        rpm = motor_force2rotor_vel(thrust / 4, sim.data.params.rpm2thrust)
        sim.rotor_vel_control(np.broadcast_to(rpm, (sim.n_worlds, sim.n_drones, 4)))
        z = [float(sim.data.states.pos[0, 0, 2])]
        for _ in range(int(DURATION * sim.freq)):
            sim.step(1)
            z.append(float(sim.data.states.pos[0, 0, 2]))
        sim.close()
        heights[integrator] = np.array(z)
        print(f"{integrator.value}: z = {z[-1]:.4f} m after {DURATION} s")

    if plot:
        plot_results(heights, sim.freq)


def plot_results(heights: dict[Integrator, np.ndarray], freq: int):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    t = np.arange(len(heights[Integrator.default])) / freq
    for integrator, z in heights.items():
        if integrator != Integrator.default:
            ax.plot(t, z - heights[Integrator.default], label=integrator.value)
    ax.set(xlabel="Time [s]", ylabel="Δz [m]")
    ax.set_title(f"Height difference to {Integrator.default.value}")
    ax.legend()
    ax.grid()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot the results")
    main(plot=parser.parse_args().plot)
