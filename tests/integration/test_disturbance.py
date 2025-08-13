import jax
import numpy as np
import pytest

from crazyflow.sim import Physics, Sim
from crazyflow.sim.structs import SimData


def disturbance_fn(data: SimData) -> SimData:
    key, subkey = jax.random.split(data.core.rng_key)
    states = data.states
    disturbance_force = jax.random.normal(subkey, states.force.shape) * 5e-1  # In world frame
    states = states.replace(force=states.force + disturbance_force)
    return data.replace(states=states, core=data.core.replace(rng_key=key))


@pytest.mark.parametrize("physics", Physics)
@pytest.mark.integration
def test_disturbance(physics: Physics):
    sim = Sim(n_worlds=2, n_drones=3, control="state", physics=physics)
    control = np.zeros((sim.n_worlds, sim.n_drones, 13))
    control[..., :3] = 1.0
    n_steps = 10

    pos, pos_disturbed = [], []
    for _ in range(n_steps):
        sim.state_control(control)
        sim.step(sim.freq // sim.control_freq)
        pos.append(sim.data.states.pos[0, 0])

    sim.reset()
    sim.step_pipeline = sim.step_pipeline[:2] + (disturbance_fn,) + sim.step_pipeline[2:]
    sim.build_step_fn()
    for _ in range(n_steps):
        sim.state_control(control)
        sim.step(sim.freq // sim.control_freq)
        pos_disturbed.append(sim.data.states.pos[0, 0])

    # Disturbed positions should be different from unperturbed positions
    assert np.all(np.array(pos) != np.array(pos_disturbed)), "Disturbance has no effect on dynamics"
