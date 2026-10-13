"""Delay the attitude commands by 0.03 s with a plugin.

The plugin queues the staged attitude commands in the plugins dict and is prepended to the step
pipeline. The drone flies the same trajectory with and without the delay, and the undelayed path
is drawn for comparison.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

os.environ["SCIPY_ARRAY_API"] = "1"

import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation as R

from crazyflow import Sim
from crazyflow.sim.pipeline import prepend_fn
from crazyflow.sim.visualize import draw_points

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData


def trajectory(t: float) -> np.ndarray:
    cmd = np.zeros((1, 1, 16))
    cmd[..., 9:13] = R.from_euler("z", 0.0).as_quat()
    cmd[..., :3] = [np.cos(t) - 1, np.sin(t), 0.2 * t]
    return cmd


def action_delay(data: SimData) -> SimData:
    """Delay state control actions."""
    queued_actions = data.plugins["queued_actions"]
    next_action = queued_actions[0]
    queued_actions = jnp.roll(queued_actions, shift=-1, axis=0)
    queued_actions = queued_actions.at[-1].set(data.controls.attitude.staged_cmd)
    data = data.replace(
        controls=data.controls.replace(
            attitude=data.controls.attitude.replace(staged_cmd=next_action)
        ),
        plugins=data.plugins | {"queued_actions": queued_actions},
    )
    return data


def main():
    sim = Sim(control="state")
    states = []
    steps = 500
    for i in range(steps):
        cmd = trajectory(i / sim.control_freq)
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        sim.render(camera="track_cam:0")
        states.append(sim.data.states.pos[0, 0])

    # Delay settings
    delay: float = 0.03  # seconds
    delay_steps = int(delay * sim.data.controls.attitude.freq)
    sim.reset()

    # Now we add our action delay into the simulation. We first insert the data we need into the
    # plugins dict, then add our plugin function into the step pipeline, and finally rebuild the sim
    # default data and step function to make sure our plugin is included and data persists across
    # resets.
    custom_data = {"queued_actions": jnp.zeros((delay_steps, 1, 1, 4))}
    sim.data = sim.data.replace(plugins=sim.data.plugins | custom_data)
    prepend_fn(sim.step_pipeline, action_delay, name="action_delay")
    sim.build_default_data()
    sim.build_step_fn()

    # Run the simulation again, this time with an action delay. The states should be different
    delayed_states = []
    for i in range(steps):
        cmd = trajectory(i / sim.control_freq)
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        draw_points(sim, states[i], size=0.02)
        sim.render()
        delayed_states.append(sim.data.states.pos[0, 0])

    position_differences = jnp.linalg.norm(jnp.array(states) - jnp.array(delayed_states), axis=-1)
    print(f"Mean position difference: {jnp.mean(position_differences) * 100:.2f}cm")
    sim.close()


if __name__ == "__main__":
    main()
