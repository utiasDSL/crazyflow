import jax
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
    disturbance_torque = jax.random.normal(subkey, states.torque.shape) * 2e-4
    states = states.replace(torque=states.torque + disturbance_torque)

    return data.replace(states=states, core=data.core.replace(rng_key=key))


def main(plot: bool = False):
    sim = Sim(control="state")
    control = np.zeros((sim.n_worlds, sim.n_drones, 13))
    control[..., :3] = 0.2

    # First run
    pos, rpy = [], []
    sim.reset()
    for _ in range(3 * sim.control_freq):
        sim.state_control(control)
        sim.step(sim.freq // sim.control_freq)
        pos.append(sim.data.states.pos[0, 0])
        rpy.append(sim.data.states.ang_vel[0, 0])
        sim.render()

    # Second run
    sim.disturbance_fn = disturbance_fn
    sim.build(mjx=False, data=False, step=True)
    pos_disturbed, rpy_disturbed = [], []
    sim.reset()
    for _ in range(3 * sim.control_freq):
        sim.state_control(control)
        sim.step(sim.freq // sim.control_freq)
        pos_disturbed.append(sim.data.states.pos[0, 0])
        rpy_disturbed.append(sim.data.states.ang_vel[0, 0])
        sim.render()

    sim.close()
    if plot:
        plot_results(pos, pos_disturbed, rpy, rpy_disturbed)


def plot_results(
    pos: list[NDArray],
    pos_disturbed: list[NDArray],
    rpy: list[NDArray],
    rpy_disturbed: list[NDArray],
):
    # Only import if plotting is desired to avoid a dependency on matplotlib
    import matplotlib.pyplot as plt  # noqa: F401

    pos, pos_disturbed = np.array(pos), np.array(pos_disturbed)
    rpy, rpy_disturbed = np.array(rpy), np.array(rpy_disturbed)
    fig, ax = plt.subplots(3, 2)
    t = np.linspace(0, 3, len(pos))
    # XYZ position
    ax[0, 0].plot(t, pos[:, 0], label="x undisturbed", color="r")
    ax[0, 0].plot(t, pos_disturbed[:, 0], label="x disturbed", color="r", linestyle="--")
    ax[1, 0].plot(t, pos[:, 1], label="y undisturbed", color="g")
    ax[1, 0].plot(t, pos_disturbed[:, 1], label="y perturbed", color="g", linestyle="--")
    ax[2, 0].plot(t, pos[:, 2], label="z undisturbed", color="b")
    ax[2, 0].plot(t, pos_disturbed[:, 2], label="z disturbed", color="b", linestyle="--")
    # RPY rates
    ax[0, 1].plot(t, rpy[:, 0], label="roll undisturbed", color="r")
    ax[0, 1].plot(t, rpy_disturbed[:, 0], label="roll disturbed", color="r", linestyle="--")
    ax[1, 1].plot(t, rpy[:, 1], label="pitch undisturbed", color="g")
    ax[1, 1].plot(t, rpy_disturbed[:, 1], label="pitch disturbed", color="g", linestyle="--")
    ax[2, 1].plot(t, rpy[:, 2], label="yaw undisturbed", color="b")
    ax[2, 1].plot(t, rpy_disturbed[:, 2], label="yaw disturbed", color="b", linestyle="--")
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
