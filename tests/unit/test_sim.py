import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.serialization import to_state_dict

from crazyflow.control.controller import Control, Controller
from crazyflow.exception import ConfigError
from crazyflow.sim.core import Sim
from crazyflow.sim.physics import Physics


@pytest.mark.unit
@pytest.mark.parametrize("physics", Physics)
@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("control", Control)
@pytest.mark.parametrize("controller", Controller)
@pytest.mark.parametrize("n_worlds", [1, 2])
def test_sim_creation(
    physics: Physics, device: str, control: Control, controller: Controller, n_worlds: int
):
    n_drones = 1

    def create_sim():
        return Sim(
            n_worlds=n_worlds,
            n_drones=n_drones,
            physics=physics,
            device=device,
            control=control,
            controller=controller,
        )

    if n_drones * n_worlds > 1 and controller == Controller.pycffirmware:
        with pytest.raises(ConfigError):
            create_sim()
        return
    if physics == Physics.sys_id and control == Control.state:
        with pytest.raises(ConfigError):
            create_sim()
        return
    sim = create_sim()
    assert sim.n_worlds == n_worlds
    assert sim.n_drones == n_drones
    assert sim.device == jax.devices(device)[0]
    assert sim.physics == physics

    # Test state buffer shapes
    assert sim.states.pos.shape == (n_worlds, n_drones, 3)
    assert sim.states.pos.device == jax.devices(device)[0]
    assert sim.states.quat.shape == (n_worlds, n_drones, 4)
    assert sim.states.vel.shape == (n_worlds, n_drones, 3)
    assert sim.states.ang_vel.shape == (n_worlds, n_drones, 3)

    # Test control buffer shapes
    assert sim._controls.attitude.shape == (n_worlds, n_drones, 4)
    assert sim._controls.thrust.shape == (n_worlds, n_drones, 4)
    assert sim._controls.state.shape == (n_worlds, n_drones, 13)
    assert sim._controls.state.device == jax.devices(device)[0]


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_setup(device: str):
    sim = Sim(n_worlds=2, n_drones=3, device=device)
    sim.setup()


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("physics", Physics)
@pytest.mark.parametrize("n_worlds", [1, 2])
@pytest.mark.parametrize("n_drones", [1, 3])
def test_reset(device: str, physics: Physics, n_worlds: int, n_drones: int):
    """Test that reset without mask resets all worlds to default state."""
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=physics, device=device)

    # Modify states
    sim.states = sim.states.replace(pos=sim.states.pos.at[:, :, 2].set(1.0))
    sim._controls = sim._controls.replace(attitude=sim._controls.attitude.at[:, :, 2].set(1.0))
    sim._params = sim._params.replace(mass=sim._params.mass.at[:, n_drones - 1].set(1.0))

    sim.reset()

    for k, v in to_state_dict(sim.states).items():
        default = getattr(sim.defaults["states"], k)
        if not isinstance(v, jnp.ndarray) or not isinstance(default, jnp.ndarray):
            continue
        assert v.shape == default.shape, f"{k} shape mismatch"
        assert v.device == default.device, f"{k} device mismatch"
        assert jnp.all(v == default), f"{k} value mismatch"
    for k, v in to_state_dict(sim._controls).items():
        default = getattr(sim.defaults["controls"], k)
        if not isinstance(v, jnp.ndarray) or not isinstance(default, jnp.ndarray):
            continue
        assert v.shape == default.shape, f"{k} shape mismatch"
        assert v.device == default.device, f"{k} device mismatch"
        assert jnp.all(v == default), f"{k} value mismatch"
    for k, v in to_state_dict(sim._params).items():
        default = getattr(sim.defaults["params"], k)
        if not isinstance(v, jnp.ndarray) or not isinstance(default, jnp.ndarray):
            continue
        assert v.shape == default.shape, f"{k} shape mismatch"
        assert v.device == default.device, f"{k} device mismatch"
        assert jnp.all(v == default), f"{k} value mismatch"


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("physics", Physics)
def test_reset_masked(device: str, physics: Physics):
    """Test that reset with mask only resets specified worlds."""
    sim = Sim(n_worlds=2, n_drones=1, physics=physics, device=device)

    # Modify states
    sim.states = sim.states.replace(pos=sim.states.pos.at[:, :, 2].set(1.0))
    sim._controls = sim._controls.replace(attitude=sim._controls.attitude.at[:, :, 2].set(1.0))
    sim._params = sim._params.replace(mass=sim._params.mass.at[:, 0].set(1.0))
    sim.states = sim.states.replace(step=sim.states.step + 100)

    # Reset only first world
    mask = jnp.array([True, False])
    sim.reset(mask)

    # Check world 1 was reset to defaults
    for k, v in to_state_dict(sim.states).items():
        default = getattr(sim.defaults["states"], k)
        if not isinstance(v, jnp.ndarray) or not isinstance(default, jnp.ndarray):
            continue
        assert v.shape == default.shape, f"{k} shape mismatch"
        assert v.device == default.device, f"{k} device mismatch"
        assert jnp.all(v[0] == default[0]), f"{k} value mismatch"
    for k, v in to_state_dict(sim._controls).items():
        default = getattr(sim.defaults["controls"], k)
        if not isinstance(v, jnp.ndarray) or not isinstance(default, jnp.ndarray):
            continue
        assert v.shape == default.shape, f"{k} shape mismatch"
        assert v.device == default.device, f"{k} device mismatch"
        assert jnp.all(v[0] == default[0]), f"{k} value mismatch"
    for k, v in to_state_dict(sim._params).items():
        default = getattr(sim.defaults["params"], k)
        if not isinstance(v, jnp.ndarray) or not isinstance(default, jnp.ndarray):
            continue
        assert v.shape == default.shape, f"{k} shape mismatch"
        assert v.device == default.device, f"{k} device mismatch"
        assert jnp.all(v[0] == default[0]), f"{k} value mismatch"

    # Check world 2 kept modifications
    assert jnp.all(sim.states.pos[1, :, 2] == 1.0)
    assert jnp.all(sim._controls.attitude[1, :, 2] == 1.0)
    assert jnp.all(sim._params.mass[1, :, 0] == 1.0)


@pytest.mark.unit
@pytest.mark.parametrize("physics", Physics)
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_sim_step(physics: Physics, device: str):
    sim = Sim(n_worlds=2, n_drones=3, physics=physics, device=device)
    try:
        for _ in range(3):
            sim.step()
    except NotImplementedError:
        pytest.skip("Physics not implemented")


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_sim_state_control(device: str):
    sim = Sim(n_worlds=2, n_drones=3, control=Control.state, device=device)
    cmd = np.random.rand(sim.n_worlds, sim.n_drones, 13)
    sim.state_control(cmd)
    assert isinstance(sim._controls.state, jnp.ndarray), "Buffers must remain JAX arrays"
    assert jnp.allclose(sim._controls.state, cmd), "Buffers must match command"


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_sim_attitude_control(device: str):
    sim = Sim(n_worlds=2, n_drones=3, control=Control.attitude, device=device)
    cmd = np.random.rand(sim.n_worlds, sim.n_drones, 4)
    sim.attitude_control(cmd)
    assert isinstance(sim._controls.attitude, jnp.ndarray), "Buffers must remain JAX arrays"
    assert jnp.allclose(sim._controls.attitude, cmd), "Buffers must match command"


@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.render
def test_render(device: str):
    sim = Sim(device=device)
    sim.render()
    sim.viewer.close()


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_device(device: str):
    sim = Sim(n_worlds=2, physics=Physics.sys_id, device=device)
    sim.step()
    assert sim.states.pos.device == jax.devices(device)[0]
    assert sim._mjx_data.qpos.device == jax.devices(device)[0]


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("n_worlds", [1, 2])
@pytest.mark.parametrize("n_drones", [1, 3])
def test_shape_consistency(device: str, n_drones: int, n_worlds: int):
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=Physics.sys_id, device=device)
    qpos_shape, qvel_shape = sim._mjx_data.qpos.shape, sim._mjx_data.qvel.shape
    sim.step()
    assert sim._mjx_data.qpos.shape == qpos_shape, "step() should not change qpos shape"
    assert sim._mjx_data.qvel.shape == qvel_shape, "step() should not change qvel shape"
