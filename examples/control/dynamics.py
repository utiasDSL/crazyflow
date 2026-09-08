import jax.numpy as jnp
import numpy as np

from crazyflow.control import Control
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.sim import Dynamics, Sim

DURATION = 5.0
FPS = 60
TRAJECTORY_CENTER = np.array([0.0, 0.0, 1.0])
TRAJECTORY_SIZE = np.array([1.0, 0.0, 0.5])


def figure_eight(t: float) -> np.ndarray:
    """Return the position, velocity, and acceleration reference at time ``t``."""
    omega = 2.0 * np.pi / DURATION
    phase = omega * t
    cmd = np.zeros((1, 1, 13))
    cmd[..., 0:3] = TRAJECTORY_CENTER + TRAJECTORY_SIZE * np.array(
        [np.sin(phase), 0.0, np.sin(2.0 * phase)]
    )
    cmd[..., 3:6] = (
        TRAJECTORY_SIZE * omega * np.array([np.cos(phase), 0.0, 2.0 * np.cos(2.0 * phase)])
    )
    cmd[..., 6:9] = (
        -TRAJECTORY_SIZE * omega**2 * np.array([np.sin(phase), 0.0, 4.0 * np.sin(2.0 * phase)])
    )
    return cmd


def reset_sim(sim: Sim) -> None:
    """Reset a simulation to the initial reference state and hover rotor speed."""
    sim.reset()
    initial_state = figure_eight(0.0)
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
        commands = np.stack([figure_eight(t) for t in times])
        reference = commands[:, 0, 0, 0:3]
        trajectory = [np.asarray(sim.data.states.pos[0, 0])]

        for i, cmd in enumerate(commands[1:]):
            sim.state_control(cmd)
            sim.step(sim.freq // sim.control_freq)
            trajectory.append(np.asarray(sim.data.states.pos[0, 0]))
            if ((i * FPS) % sim.control_freq) < FPS and render:
                sim.render()

        sim.close()
        trajectory = np.asarray(trajectory)
        trajectories[dynamics] = trajectory
        rmses[dynamics] = np.sqrt(np.mean(np.sum((trajectory - reference) ** 2, axis=-1)))
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
    for dynamics, trajectory in trajectories.items():
        label = f"{dynamics.value} (RMSE={rmses[dynamics] * 1_000:.0f}mm)"
        ax_xz.plot(trajectory[:, 0], trajectory[:, 2], label=label)
        ax_y.plot(times, trajectory[:, 1], label=label)

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
    main(plot=True, render=False)
