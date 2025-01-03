import numpy as np
import pytest

from crazyflow.constants import J
from crazyflow.control import Control
from crazyflow.randomize import randomize_inertia, randomize_mass
from crazyflow.sim import Physics, Sim


@pytest.mark.integration
@pytest.mark.parametrize("physics", [Physics.analytical, Physics.mujoco])
def test_randomize_mass(physics: Physics):
    if physics == Physics.mujoco:  # TODO: Add mujoco when implemented
        pytest.skip("MuJoCo randomization not implemented yet")

    sim = Sim(n_worlds=2, n_drones=4, control=Control.state, physics=physics)

    add_on_mass = np.random.uniform(-0.005, 0.005, size=(sim.n_worlds, sim.n_drones))
    masses = np.ones((sim.n_worlds, sim.n_drones)) * 0.025
    randomized_masses = masses + add_on_mass
    control = np.zeros((sim.n_worlds, sim.n_drones, 13))
    control[:, :, 2] = 0.5

    sim.state_control(control)
    sim.step(sim.freq // sim.control_freq)
    pos = sim.data.states.pos
    sim.reset()

    randomize_mass(sim, randomized_masses)

    sim.state_control(control)
    sim.step(sim.freq // sim.control_freq)
    pos_random = sim.data.states.pos
    assert not np.all(pos == pos_random), "Inertia randomization has no effect on dynamics"


@pytest.mark.integration
@pytest.mark.parametrize("physics", [Physics.analytical, Physics.mujoco])
def test_randomize_inertia(physics: Physics):
    if physics == Physics.mujoco:  # TODO: Add mujoco when implemented
        pytest.skip("MuJoCo randomization not implemented yet")

    sim = Sim(n_worlds=2, n_drones=4, control=Control.state, physics=physics)

    add_on_j = np.random.uniform(-1.5e-5, 1.5e-5, size=(sim.n_worlds, sim.n_drones, 3, 3))
    randomized_j = J + add_on_j
    control = np.zeros((sim.n_worlds, sim.n_drones, 13))
    control[:, :, 2] = 0.5

    sim.state_control(control)
    sim.step(sim.freq // sim.control_freq)
    pos = sim.data.states.pos
    sim.reset()

    randomize_inertia(sim, randomized_j)

    sim.state_control(control)
    sim.step(sim.freq // sim.control_freq)
    pos_random = sim.data.states.pos
    assert not np.all(pos == pos_random), "Inertia randomization has no effect on dynamics"
