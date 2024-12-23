import jax
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from crazyflow.sim import Sim
from crazyflow.sim.structs import SimData


def disturbance_fn(data: SimData) -> SimData:
    key, subkey = jax.random.split(data.core.rng_key)
    states = data.states
    disturbance_force = jax.random.normal(subkey, states.force.shape) * 2e-1  # In world frame
    states = states.replace(force=states.force + disturbance_force)

    key, subkey = jax.random.split(key)
    disturbance_torque = jax.random.normal(subkey, states.torque.shape) * 2e-3
    states = states.replace(torque=states.torque + disturbance_torque)

    return data.replace(states=states, core=data.core.replace(rng_key=key))


def main(plot: bool = False):
    sim = Sim(control="state")
    control = np.zeros((sim.n_worlds, sim.n_drones, 13))
    control[..., :3] = 0.4

    # First run
    pos_unperturbed = []
    rpy_unperturbed = []
    sim.reset()
    for _ in range(3 * sim.control_freq):
        sim.state_control(control)
        sim.step(sim.freq // sim.control_freq)
        pos_unperturbed.append(sim.data.states.pos[0, 0])
        rpy_unperturbed.append(sim.data.states.rpy_rates[0, 0])
        sim.render()

    # Second run
    sim.disturbance_fn = disturbance_fn
    sim.build()
    pos_perturbed = []
    rpy_perturbed = []
    sim.reset()
    for _ in range(3 * sim.control_freq):
        sim.state_control(control)
        sim.step(sim.freq // sim.control_freq)
        pos_perturbed.append(sim.data.states.pos[0, 0])
        rpy_perturbed.append(sim.data.states.rpy_rates[0, 0])
        sim.render()

    sim.close()
    if plot:
        plot_results(pos_unperturbed, pos_perturbed, rpy_unperturbed, rpy_perturbed)


def plot_results(
    pos_unperturbed: list[NDArray],
    pos_perturbed: list[NDArray],
    rpy_unperturbed: list[NDArray],
    rpy_perturbed: list[NDArray],
):
    pos_unperturbed, pos_perturbed = np.array(pos_unperturbed), np.array(pos_perturbed)
    rpy_unperturbed, rpy_perturbed = np.array(rpy_unperturbed), np.array(rpy_perturbed)
    fig, ax = plt.subplots(3, 2)
    t = np.linspace(0, 3, len(pos_unperturbed))
    # XYZ position
    ax[0, 0].plot(t, pos_unperturbed[:, 0], label="x unperturbed", color="r")
    ax[0, 0].plot(t, pos_perturbed[:, 0], label="x perturbed", color="r", linestyle="--")
    ax[1, 0].plot(t, pos_unperturbed[:, 1], label="y unperturbed", color="g")
    ax[1, 0].plot(t, pos_perturbed[:, 1], label="y perturbed", color="g", linestyle="--")
    ax[2, 0].plot(t, pos_unperturbed[:, 2], label="z unperturbed", color="b")
    ax[2, 0].plot(t, pos_perturbed[:, 2], label="z perturbed", color="b", linestyle="--")
    # RPY rates
    ax[0, 1].plot(t, rpy_unperturbed[:, 0], label="roll unperturbed", color="r")
    ax[0, 1].plot(t, rpy_perturbed[:, 0], label="roll perturbed", color="r", linestyle="--")
    ax[1, 1].plot(t, rpy_unperturbed[:, 1], label="pitch unperturbed", color="g")
    ax[1, 1].plot(t, rpy_perturbed[:, 1], label="pitch perturbed", color="g", linestyle="--")
    ax[2, 1].plot(t, rpy_unperturbed[:, 2], label="yaw unperturbed", color="b")
    ax[2, 1].plot(t, rpy_perturbed[:, 2], label="yaw perturbed", color="b", linestyle="--")
    fig.suptitle("Dynamics with disturbance")
    fig.supxlabel("Time (s)")
    ax[0, 0].sharex(ax[1, 0])
    ax[1, 0].sharex(ax[2, 0])
    ax[0, 1].sharex(ax[1, 1])
    ax[1, 1].sharex(ax[2, 1])
    for _ax in ax.flatten():
        _ax.legend()
    plt.show()


if __name__ == "__main__":
    main(plot=True)  # Default is False to disable plotting during testing
