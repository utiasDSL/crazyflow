"""Compare gradients through the rotor command clip with straight-through and no clipping.

Hard clips such as the `clip_rotor_vel_cmd` stage zero the gradients while the command is
saturated. This example ramps a motor command beyond the rotor limits and compares the rotor state
and the gradient of the vertical acceleration w.r.t. the command for three options: the default
clip, a straight-through clip (clipped forward pass, unclipped gradients), and no clip. Replacing
the default clip with the straight-through variant can help gradient-based methods such as
trajectory optimization or policy learning, which would otherwise receive zero gradients whenever
the motors saturate.

Run with `--plot` to plot the results, which requires matplotlib from the `tests` environment.
"""

import argparse
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

import crazyflow.sim.functional as F
from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.pipeline import remove_fn, replace_fn
from crazyflow.sim.sim import rotor_vel_limits


def clip_rotor_vel_cmd_nonblocking(data: SimData, lower: float, upper: float) -> SimData:
    # Straight-through estimator: x + stop_gradient(clip(x) - x) evaluates to clip(x) in the
    # forward pass, while its derivative w.r.t. x is 1 in the backward pass
    cmd = data.controls.rotor_vel
    cmd = cmd + jax.lax.stop_gradient(jnp.clip(cmd, lower, upper) - cmd)
    return data.replace(controls=data.controls.replace(rotor_vel=cmd))


def rollout(sim: Sim, cmds: NDArray) -> tuple[NDArray, NDArray]:
    step_fn = sim.build_step_fn()

    # The command only enters the acceleration through the rotor state of the *next* step, so we
    # measure the acceleration with a one-step lookahead while the outer loop advances by a single
    # step per command
    def acc_z(cmd: jax.Array, data: SimData) -> tuple[jax.Array, SimData]:
        data = F.rotor_vel_control(data, jnp.full((1, 1, 4), cmd))
        data = step_fn(data, 1)
        lookahead = step_fn(data, 1)
        acc = (lookahead.states.vel[0, 0, 2] - data.states.vel[0, 0, 2]) * sim.freq
        return acc, data

    grad_fn = jax.jit(jax.value_and_grad(acc_z, has_aux=True))

    data, rotor_vel, grads = sim.data, [], []
    for cmd in cmds:
        (_, data), grad = grad_fn(jnp.float32(cmd), data)
        rotor_vel.append(data.states.rotor_vel[0, 0, 0])
        grads.append(grad)
    return np.array(rotor_vel), np.array(grads)


def main(plot: bool = False):
    sim = Sim(control=Control.rotor_vel)
    lower, upper = rotor_vel_limits(sim.dynamics, sim.drone)
    # Start in the air so that the drone never reaches the floor, where the floor clipping would
    # zero the velocity and kill the gradients (see gradient.py)
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=sim.data.states.pos.at[..., 2].set(2.0))
    )

    # Motor command (RPM): ramp up beyond the upper limit, hold, ramp back down to zero
    ramp = float(upper) + 10_000
    n = 250  # 0.5 s per segment at 500 Hz
    cmds = np.concatenate([np.linspace(0, ramp, n), np.full(n, ramp), np.linspace(ramp, 0, n)])

    # Option 1: keep the clipping as is. The rotor state respects the limits, but the gradient is
    # zero while the command is saturated
    results = {"clip (default)": rollout(sim, cmds)}

    # Option 2: clip the command in the forward pass, but keep the gradients flowing in the
    # backward pass (straight-through estimator)
    clip_fn = partial(clip_rotor_vel_cmd_nonblocking, lower=lower, upper=upper)
    replace_fn(sim.step_pipeline, clip_fn, "clip_rotor_vel_cmd")
    results["nonblocking clip"] = rollout(sim, cmds)

    # Option 3: remove the clipping. Gradients always flow, but the state can leave the limits
    remove_fn(sim.step_pipeline, "clip_rotor_vel_cmd")
    results["no clip"] = rollout(sim, cmds)

    sim.close()
    if plot:
        plot_results(cmds, results, float(lower), float(upper), sim.freq)


def plot_results(
    cmds: NDArray,
    results: dict[str, tuple[NDArray, NDArray]],
    lower: float,
    upper: float,
    freq: int,
):
    # Only import if plotting is desired to avoid a dependency on matplotlib
    import matplotlib.pyplot as plt

    t = np.arange(len(cmds)) / freq
    fig, (ax_state, ax_grad) = plt.subplots(1, 2, sharex="all", figsize=(12, 5))
    ax_state.plot(t, cmds, label="command", color="gray", linestyle=":")
    # The forward pass of the nonblocking clip is identical to the default clip, so we plot it
    # dashed to keep both curves visible
    styles = {"nonblocking clip": {"linestyle": "--"}}
    for name, (rotor_vel, grads) in results.items():
        ax_state.plot(t, rotor_vel, label=name, **styles.get(name, {}))
        ax_grad.plot(t, grads, label=name, **styles.get(name, {}))
    for limit in (lower, upper):
        ax_state.axhline(limit, color="black", linestyle="--", linewidth=0.8)
    ax_state.set_title("Rotor state")
    ax_state.set_ylabel("rotor_vel (RPM)")
    ax_grad.set_title("Gradient d acc_z / d cmd")
    ax_grad.set_ylabel("(m/s$^2$) / RPM")
    for ax in (ax_state, ax_grad):
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.grid(True)
    fig.suptitle("Rotor command clipping and its effect on gradients")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot the results")
    main(plot=parser.parse_args().plot)
