import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.serialization import to_state_dict

from crazyflow.control.controller import Control, Controller
from crazyflow.exception import ConfigError
from crazyflow.sim.core import Sim
from crazyflow.sim.physics import Physics


def available_backends() -> list[str]:
    """Return list of available JAX backends."""
    backends = []
    for backend in ["tpu", "gpu", "cpu"]:
        try:
            jax.devices(backend)
        except RuntimeError:
            pass
        else:
            backends.append(backend)
    return backends


def skip_unavailable_device(device: str):
    if device not in available_backends():
        pytest.skip(f"{device} device not available")


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
    skip_unavailable_device(device)

    def create_sim() -> Sim:
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
    assert sim.controls.attitude.shape == (n_worlds, n_drones, 4)
    assert sim.controls.thrust.shape == (n_worlds, n_drones, 4)
    assert sim.controls.state.shape == (n_worlds, n_drones, 13)
    assert sim.controls.state.device == jax.devices(device)[0]


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("physics", Physics)
@pytest.mark.parametrize("n_worlds", [1, 2])
@pytest.mark.parametrize("n_drones", [1, 3])
def test_reset(device: str, physics: Physics, n_worlds: int, n_drones: int):
    """Test that reset without mask resets all worlds to default state."""
    skip_unavailable_device(device)
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=physics, device=device)
    if physics == Physics.mujoco:
        return  # MuJoCo is not yet supported. TODO: Enable once supported

    # Modify states
    sim.steps = sim.steps + 100
    sim.last_ctrl_steps = sim.last_ctrl_steps + 100
    sim.states = sim.states.replace(pos=sim.states.pos.at[:, :, 2].set(1.0))
    sim.controls = sim.controls.replace(attitude=sim.controls.attitude.at[:, :, 2].set(1.0))
    sim.params = sim.params.replace(mass=sim.params.mass.at[:, n_drones - 1].set(1.0))
    sim.controls = sim.controls.replace(staged_attitude=jnp.ones_like(sim.controls.staged_attitude))
    sim.controls = sim.controls.replace(state=jnp.ones_like(sim.controls.state))
    sim.reset()

    for k, v in to_state_dict(sim.states).items():
        default = getattr(sim.defaults["states"], k)
        if not isinstance(v, jnp.ndarray) or not isinstance(default, jnp.ndarray):
            continue
        assert v.shape == default.shape, f"{k} shape mismatch"
        assert v.device == default.device, f"{k} device mismatch"
        assert jnp.all(v == default), f"{k} value mismatch"
    for k, v in to_state_dict(sim.controls).items():
        default = getattr(sim.defaults["controls"], k)
        if not isinstance(v, jnp.ndarray) or not isinstance(default, jnp.ndarray):
            continue
        assert v.shape == default.shape, f"{k} shape mismatch"
        assert v.device == default.device, f"{k} device mismatch"
        assert jnp.all(v == default), f"{k} value mismatch"
    for k, v in to_state_dict(sim.params).items():
        default = getattr(sim.defaults["params"], k)
        if not isinstance(v, jnp.ndarray) or not isinstance(default, jnp.ndarray):
            continue
        assert v.shape == default.shape, f"{k} shape mismatch"
        assert v.device == default.device, f"{k} device mismatch"
        assert jnp.all(v == default), f"{k} value mismatch"
    assert jnp.all(sim.steps == 0), "Steps must be reset to 0"
    assert jnp.all(sim.last_ctrl_steps == -sim.freq), "Last control steps must be reset to -freq"


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("physics", Physics)
def test_reset_masked(device: str, physics: Physics):
    """Test that reset with mask only resets specified worlds."""
    skip_unavailable_device(device)
    sim = Sim(n_worlds=2, n_drones=1, physics=physics, device=device)

    # Modify states
    sim.states = sim.states.replace(pos=sim.states.pos.at[:, :, 2].set(1.0))
    sim.controls = sim.controls.replace(attitude=sim.controls.attitude.at[:, :, 2].set(1.0))
    sim.params = sim.params.replace(mass=sim.params.mass.at[:, 0].set(1.0))
    sim.steps = sim.steps + 100

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
    for k, v in to_state_dict(sim.controls).items():
        default = getattr(sim.defaults["controls"], k)
        if not isinstance(v, jnp.ndarray) or not isinstance(default, jnp.ndarray):
            continue
        assert v.shape == default.shape, f"{k} shape mismatch"
        assert v.device == default.device, f"{k} device mismatch"
        assert jnp.all(v[0] == default[0]), f"{k} value mismatch"
    for k, v in to_state_dict(sim.params).items():
        default = getattr(sim.defaults["params"], k)
        if not isinstance(v, jnp.ndarray) or not isinstance(default, jnp.ndarray):
            continue
        assert v.shape == default.shape, f"{k} shape mismatch"
        assert v.device == default.device, f"{k} device mismatch"
        assert jnp.all(v[0] == default[0]), f"{k} value mismatch"

    # Check world 2 kept modifications
    assert jnp.all(sim.states.pos[1, :, 2] == 1.0)
    assert jnp.all(sim.controls.attitude[1, :, 2] == 1.0)
    assert jnp.all(sim.params.mass[1, :, 0] == 1.0)


@pytest.mark.unit
@pytest.mark.parametrize("n_worlds", [1, 2])
@pytest.mark.parametrize("n_drones", [1, 3])
@pytest.mark.parametrize("physics", Physics)
@pytest.mark.parametrize("control", Control)
@pytest.mark.parametrize("controller", Controller)
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_sim_step(
    n_worlds: int,
    n_drones: int,
    physics: Physics,
    control: Control,
    controller: Controller,
    device: str,
):
    skip_unavailable_device(device)
    if n_drones * n_worlds > 1 and controller == Controller.pycffirmware:
        return  # PyCFFirmware does not support multiple drones
    sim = Sim(
        n_worlds=n_worlds,
        n_drones=n_drones,
        physics=physics,
        device=device,
        control=control,
        controller=controller,
    )
    try:
        for _ in range(2):
            sim.step()
    except NotImplementedError:
        pytest.skip("Physics not implemented")  # TODO: Remove once MuJoCo is supported


@pytest.mark.unit
@pytest.mark.parametrize("control_freq", [33, 50, 100, 200])
def test_sim_attitude_control(control_freq: int):
    sim = Sim(n_worlds=2, n_drones=3, control=Control.attitude, freq=100, control_freq=control_freq)
    can_control_1 = np.arange(6) * control_freq % sim.freq < control_freq
    can_control_2 = np.array([0, 0, 1, 2, 3, 4]) * control_freq % sim.freq < control_freq
    for i in range(6):
        cmd = np.random.rand(sim.n_worlds, sim.n_drones, 4)
        assert jnp.all(sim.controllable[0] == can_control_1[i]), f"Controllable 1 mismatch at t={i}"
        assert jnp.all(sim.controllable[1] == can_control_2[i]), f"Controllable 2 mismatch at t={i}"
        sim.attitude_control(cmd)
        sim.step()
        sim_cmd = sim.controls.attitude[0]
        if can_control_1[i]:
            assert jnp.all(sim_cmd == cmd[0]), f"Controls do not match at t={i}"
        else:
            assert not jnp.all(sim_cmd == cmd[0]), f"Controls shouldn't match at t={i}"
        sim_cmd = sim.controls.attitude[1]
        if can_control_2[i]:
            assert jnp.all(sim_cmd == cmd[1]), f"Controls do not match at t={i}"
        else:
            assert not jnp.all(sim_cmd == cmd[1]), f"Controls shouldn't match at t={i}"
        if i == 0:
            sim.reset(np.array([False, True]))  # Make world 2 asynchronous


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_sim_attitude_control_device(device: str):
    skip_unavailable_device(device)
    sim = Sim(n_worlds=2, n_drones=3, control=Control.attitude, device=device)
    cmd = np.random.rand(sim.n_worlds, sim.n_drones, 4)
    sim.attitude_control(cmd)
    assert isinstance(sim.controls.staged_attitude, jnp.ndarray), "Buffers must remain JAX arrays"
    assert jnp.all(sim.controls.staged_attitude == cmd), "Buffers must match command"


@pytest.mark.unit
@pytest.mark.parametrize("control_freq", [33, 50, 100, 200])
def test_sim_state_control(control_freq: int):
    sim = Sim(n_worlds=2, n_drones=3, control=Control.state, freq=100, control_freq=control_freq)
    can_control_1 = np.arange(6) * control_freq % sim.freq < control_freq
    can_control_2 = np.array([0, 0, 1, 2, 3, 4]) * control_freq % sim.freq < control_freq
    for i in range(6):
        cmd = np.random.rand(sim.n_worlds, sim.n_drones, 13)
        assert jnp.all(sim.controllable[0] == can_control_1[i]), f"Controllable 1 mismatch at t={i}"
        assert jnp.all(sim.controllable[1] == can_control_2[i]), f"Controllable 2 mismatch at t={i}"
        sim.state_control(cmd)
        last_attitude = sim.controls.staged_attitude
        sim.step()
        attitude = sim.controls.staged_attitude
        last_att, att = last_attitude[0], attitude[0]
        if can_control_1[i]:
            assert not jnp.all(att == last_att), f"Controls haven't been applied at t={i}"
        else:
            assert jnp.all(att == last_att), f"Controls should be unchanged at t={i}"
        last_att, att = last_attitude[1], attitude[1]
        if can_control_2[i]:
            assert not jnp.all(att == last_att), f"Controls haven't been applied at t={i}"
        else:
            assert jnp.all(att == last_att), f"Controls should be unchanged at t={i}"
        if i == 0:
            sim.reset(np.array([False, True]))  # Make world 2 asynchronous


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_sim_state_control_device(device: str):
    skip_unavailable_device(device)
    sim = Sim(n_worlds=2, n_drones=3, control=Control.state, device=device)
    cmd = np.random.rand(sim.n_worlds, sim.n_drones, 13)
    sim.state_control(cmd)
    assert isinstance(sim.controls.state, jnp.ndarray), "Buffers must remain JAX arrays"
    assert jnp.all(sim.controls.state == cmd), "Buffers must match command"


@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.render
def test_render(device: str):
    skip_unavailable_device(device)
    sim = Sim(device=device)
    sim.render()
    sim.viewer.close()


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_device(device: str):
    skip_unavailable_device(device)
    sim = Sim(n_worlds=2, physics=Physics.sys_id, device=device)
    sim.step()
    assert sim.states.pos.device == jax.devices(device)[0]
    assert sim._mjx_data.qpos.device == jax.devices(device)[0]


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("n_worlds", [1, 2])
@pytest.mark.parametrize("n_drones", [1, 3])
def test_shape_consistency(device: str, n_drones: int, n_worlds: int):
    skip_unavailable_device(device)
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=Physics.sys_id, device=device)
    qpos_shape, qvel_shape = sim._mjx_data.qpos.shape, sim._mjx_data.qvel.shape
    sim.step()
    assert sim._mjx_data.qpos.shape == qpos_shape, "step() should not change qpos shape"
    assert sim._mjx_data.qvel.shape == qvel_shape, "step() should not change qvel shape"
