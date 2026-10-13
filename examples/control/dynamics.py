"""Compare all dynamics models while tracking the same figure-eight with state control.

Each model starts from the same reference state, and the position RMSE to the reference is
printed for every model. The plot shows the x-z trajectories and the lateral positions against the
reference.

Run with `--plot` to plot the results, which requires matplotlib from the `tests` environment.
"""

import argparse
import os

os.environ["SCIPY_ARRAY_API"] = "1"

import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation as R

from crazyflow.control import Control
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.sim import Dynamics, Sim

DURATION = 5.0
FPS = 60
TRAJ_CENTER = np.array([0.0, 0.0, 1.0])
TRAJ_SIZE = np.array([1.0, 0.0, 0.5])


def trajectory(t: float) -> np.ndarray:
    """Return the position, velocity, and acceleration reference at time ``t``."""
    omega = 2.0 * np.pi / DURATION
    phi = omega * t
    cmd = np.zeros((1, 1, 16))
    cmd[..., 9:13] = R.from_euler("z", 0.0).as_quat()
    cmd[..., 0:3] = TRAJ_CENTER + TRAJ_SIZE * np.array([np.sin(phi), 0.0, np.sin(2.0 * phi)])
    cmd[..., 3:6] = TRAJ_SIZE * omega * np.array([np.cos(phi), 0.0, 2.0 * np.cos(2.0 * phi)])
    cmd[..., 6:9] = -TRAJ_SIZE * omega**2 * np.array([np.sin(phi), 0.0, 4.0 * np.sin(2.0 * phi)])
    return cmd


def reset_sim(sim: Sim) -> None:
    """Reset a simulation to the initial reference state and hover rotor speed."""
    sim.reset()
    initial_state = trajectory(0.0)
    hover_thrust = sim.data.params.mass * -sim.data.params.gravity_vec[-1]

    if sim.dynamics == Dynamics.first_principles:
        motor_forces = jnp.broadcast_to(hover_thrust / 4.0, sim.data.states.rotor_vel.shape)
        rotor_vel = motor_force2rotor_vel(motor_forces, sim.data.params.rpm2thrust)
    elif sim.dynamics in (Dynamics.so_rpy_rotor, Dynamics.so_rpy_rotor_drag):
        # These models store collective thrust in the shared rotor velocity state.
        hover_state = (hover_thrust - sim.data.params.acc_coef) / sim.data.params.cmd_f_coef
        rotor_vel = jnp.broadcast_to(hover_state, sim.data.states.rotor_vel.shape)
    else:
        # The rotor state is unused by so_rpy, but keep its reset state nonzero too.
        rotor_vel = jnp.broadcast_to(hover_thrust, sim.data.states.rotor_vel.shape)

    states = sim.data.states.replace(
        pos=sim.data.states.pos.at[...].set(initial_state[..., 0:3]),
        vel=sim.data.states.vel.at[...].set(initial_state[..., 3:6]),
        rotor_vel=sim.data.states.rotor_vel.at[...].set(rotor_vel),
    )
    sim.data = sim.data.replace(states=states)


def main(plot: bool = False, render: bool = False) -> None:
    trajectories: dict[Dynamics, np.ndarray] = {}
    rmses: dict[Dynamics, float] = {}
    reference = np.empty((0, 3))

    for dynamics in Dynamics:
        sim = Sim(dynamics=dynamics, control=Control.state)
        reset_sim(sim)

        n_steps = int(DURATION * sim.control_freq)
        times = np.arange(n_steps + 1) / sim.control_freq
        commands = np.stack([trajectory(t) for t in times])
        reference = commands[:, 0, 0, 0:3]
        positions = [np.asarray(sim.data.states.pos[0, 0])]

        for i, cmd in enumerate(commands[1:]):
            sim.state_control(cmd)
            sim.step(sim.freq // sim.control_freq)
            positions.append(np.asarray(sim.data.states.pos[0, 0]))
            if ((i * FPS) % sim.control_freq) < FPS and render:
                sim.render()

        sim.close()
        positions = np.asarray(positions)
        trajectories[dynamics] = positions
        rmses[dynamics] = np.sqrt(np.mean(np.sum((positions - reference) ** 2, axis=-1)))
        print(f"{dynamics.value}: position RMSE = {rmses[dynamics]:.3f} m")

    if plot:
        plot_results(reference, trajectories, rmses)


def plot_results(
    reference: np.ndarray, trajectories: dict[Dynamics, np.ndarray], rmses: dict[Dynamics, float]
) -> None:
    """Plot the x-z trajectories and lateral positions against the reference."""
    # Only import if plotting is desired to avoid a dependency on matplotlib.
    import matplotlib.pyplot as plt

    fig, (ax_xz, ax_y) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    ax_xz.plot(reference[:, 0], reference[:, 2], "k--", linewidth=2, label="reference")
    times = np.linspace(0.0, DURATION, len(reference))
    ax_y.plot(times, reference[:, 1], "k--", linewidth=2, label="reference")
    for dynamics, positions in trajectories.items():
        label = f"{dynamics.value} (RMSE={rmses[dynamics] * 1_000:.0f}mm)"
        ax_xz.plot(positions[:, 0], positions[:, 2], label=label)
        ax_y.plot(times, positions[:, 1], label=label)

    ax_xz.set_title("Figure-eight trajectory")
    ax_xz.set_xlabel("x (m)")
    ax_xz.set_ylabel("z (m)")
    ax_xz.set_aspect("equal", adjustable="datalim")
    ax_xz.legend()
    ax_y.set_title("Out-of-plane position")
    ax_y.set_xlabel("time (s)")
    ax_y.set_ylabel("y (m)")
    for ax in (ax_xz, ax_y):
        ax.set_box_aspect(1)
        ax.grid()
    fig.suptitle("Figure-eight tracking across dynamics models")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot the results")
    main(plot=parser.parse_args().plot)
