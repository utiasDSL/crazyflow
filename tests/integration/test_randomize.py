import jax
import numpy as np
import pytest

from crazyflow.control import Control
from crazyflow.randomize import randomize_inertia, randomize_mass
from crazyflow.sim import Physics, Sim


@pytest.mark.integration
def test_randomize_mass():
    sim = Sim(n_worlds=2, n_drones=4, control=Control.state, physics=Physics.first_principles)

    control = np.zeros((sim.n_worlds, sim.n_drones, 13))
    control[:, :, 2] = 0.5

    sim.state_control(control)
    sim.step(sim.freq // sim.control_freq)
    pos = sim.data.states.pos

    sim.reset()
    mass_disturbance = jax.random.normal(jax.random.key(0), sim.data.params.mass.shape) * 1e-4
    randomize_mass(sim, sim.data.params.mass + mass_disturbance)

    sim.state_control(control)
    sim.step(sim.freq // sim.control_freq)
    pos_random = sim.data.states.pos
    assert not np.all(pos == pos_random), "Mass randomization has no effect on dynamics"


@pytest.mark.integration
def test_randomize_inertia():
    sim = Sim(n_worlds=2, n_drones=4, control=Control.state, physics=Physics.first_principles)

    control = np.zeros((sim.n_worlds, sim.n_drones, 13))
    control[:, :, :2] = 0.1  # Sideways motion to force tilt for inertia to have an effect
    control[:, :, 2] = 0.5

    sim.state_control(control)
    sim.step(50)
    pos = sim.data.states.pos

    sim.reset()
    J = sim.data.params.J + jax.random.normal(jax.random.key(0), sim.data.params.J.shape) * 1e-5
    randomize_inertia(sim, J)

    sim.state_control(control)
    sim.step(50)
    pos_random = sim.data.states.pos
    assert not np.all(pos == pos_random), "Inertia randomization has no effect on dynamics"
