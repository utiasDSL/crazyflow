"""Add the state derivatives, which are not stored by default, to the step pipeline.

We compare two methods for the state derivatives: evaluating the dynamics and finite differences.
The dynamics need the current state and input, so we evaluate them before integration. At the end
of the step, they are the derivative at the previous state. Finite differences are taken after
integration and give the derivative from the previous to the current state. For Euler integration,
both methods are identical up to numerical noise. Other integrators differ, as shown here for RK4.
Both plugins store a `SimStateDeriv` in the plugins dict, and the plot compares them while the
drone flies a figure-eight.

Run with `--plot` to plot the results, which requires matplotlib from the `tests` environment.
"""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

os.environ["SCIPY_ARRAY_API"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.sim import Sim
from crazyflow.sim.data import SimStateDeriv
from crazyflow.sim.dynamics import first_principles_dynamics
from crazyflow.sim.pipeline import append_fn, insert_fn_before

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData

DURATION = 10.0  # s, one loop of the figure-eight


def dynamics_deriv(data: SimData) -> SimData:
    """Evaluate the dynamics."""
    return data.replace(plugins=data.plugins | {"states_deriv": first_principles_dynamics(data)})


def finite_diff_deriv(data: SimData) -> SimData:
    """Differentiate the states over the last step."""
    prev, states, freq = data.plugins["prev_states"], data.states, data.core.freq
    rot = R.from_quat(prev.quat).inv() * R.from_quat(states.quat)
    deriv = SimStateDeriv(
        vel=(states.pos - prev.pos) * freq,
        ang_vel=rot.as_rotvec() * freq,
        acc=(states.vel - prev.vel) * freq,
        ang_acc=(states.ang_vel - prev.ang_vel) * freq,
        rotor_acc=(states.rotor_vel - prev.rotor_vel) * freq,
    )
    return data.replace(plugins=data.plugins | {"fd_states_deriv": deriv, "prev_states": states})


def trajectory(t: float) -> np.ndarray:
    """Return a figure-eight state command."""
    omega = 2 * np.pi / DURATION
    cmd = np.zeros((1, 1, 16))
    cmd[..., :3] = [2 * np.sin(omega * t), np.sin(2 * omega * t), 1.0]
    cmd[..., 9:13] = R.from_euler("z", 0.0).as_quat()
    return cmd


def main(plot: bool = False):
    results = {}
    for integrator in ("euler", "rk4"):
        sim = Sim(dynamics="first_principles", control="state", integrator=integrator)
        pos = jnp.asarray(trajectory(0.0)[..., :3], device=sim.device)
        sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos))
        plugins = {
            "states_deriv": SimStateDeriv.create(sim.n_worlds, sim.n_drones, sim.device),
            "fd_states_deriv": SimStateDeriv.create(sim.n_worlds, sim.n_drones, sim.device),
            "prev_states": sim.data.states,
        }
        sim.data = sim.data.replace(plugins=sim.data.plugins | plugins)

        insert_fn_before(sim.step_pipeline, "integration", dynamics_deriv)
        append_fn(sim.step_pipeline, finite_diff_deriv)
        sim.build_default_data()
        sim.build_step_fn()

        log = {"states_deriv": [], "fd_states_deriv": []}
        for i in range(int(DURATION * sim.control_freq)):
            sim.state_control(trajectory(i / sim.control_freq))
            sim.step(sim.freq // sim.control_freq)
            for key in log:
                log[key].append(jax.tree.map(lambda x: np.asarray(x[0, 0]), sim.data.plugins[key]))
        sim.close()

        dynamics, fd = (jax.tree.map(lambda *x: np.stack(x), *log[key]) for key in log)
        results[integrator] = dynamics, fd
        for name in ("vel", "ang_vel", "acc", "ang_acc", "rotor_acc"):
            x, x_fd = getattr(dynamics, name), getattr(fd, name)
            diff = np.abs(x - x_fd).max() / np.abs(x).max()
            print(f"{integrator} {name}: max relative difference {diff:.1e}")

    if plot:
        plot_results(results, sim.control_freq)


def plot_results(results: dict[str, tuple[SimStateDeriv, SimStateDeriv]], freq: int):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(12, 7))
    quantities = (
        ("acc", "Linear acceleration", "m/s$^2$"),
        ("ang_acc", "Angular acceleration", "rad/s$^2$"),
    )
    for row, (integrator, (dynamics, fd)) in enumerate(results.items()):
        for col, (name, title, unit) in enumerate(quantities):
            x, x_fd = getattr(dynamics, name), getattr(fd, name)
            t = np.arange(len(x)) / freq
            for i, axis in enumerate("xyz"):
                axes[row, col].plot(t, x_fd[:, i] - x[:, i], f"C{i}", lw=0.8, label=axis)
            ylabel = f"Finite differences - dynamics [{unit}]"
            axes[row, col].set(title=f"{title} ({integrator})", ylabel=ylabel)
    for ax in axes[-1]:
        ax.set(xlabel="Time [s]")
    axes[0, 0].legend()
    for ax in axes.flat:
        ax.grid()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot the results")
    main(plot=parser.parse_args().plot)
