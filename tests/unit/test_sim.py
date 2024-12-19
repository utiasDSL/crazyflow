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
def test_sim_init(
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
    if physics != Physics.analytical and control == Control.thrust:
        with pytest.raises(ConfigError):  # TODO: Remove when supported with sys_id
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
def test_setup(device: str):
    skip_unavailable_device(device)
    sim = Sim(n_worlds=2, n_drones=3, device=device)
    sim.setup()


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
    sim.controls = sim.controls.replace(staged_state=jnp.ones_like(sim.controls.staged_state))
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
    if physics != Physics.analytical and control == Control.thrust:
        return  # TODO: Remove when supported with sys_id
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
    except NotImplementedError:  # TODO: Remove once MuJoCo is supported
        pytest.skip("Physics not implemented")


@pytest.mark.unit
@pytest.mark.parametrize("control", [Control.state, Control.attitude])
@pytest.mark.parametrize("control_freq", [33, 50, 100, 200])
def test_sim_control(control: Control, control_freq: int):
    sim = Sim(n_worlds=2, n_drones=3, control=control, freq=100, control_freq=control_freq)
    cmd_dim = 13 if control == Control.state else 4
    can_control_1 = np.arange(6) * control_freq % sim.freq < control_freq
    can_control_2 = np.array([0, 0, 1, 2, 3, 4]) * control_freq % sim.freq < control_freq
    cmd_fn = sim.state_control if control == Control.state else sim.attitude_control
    for i in range(6):
        cmd = np.random.rand(sim.n_worlds, sim.n_drones, cmd_dim)
        assert jnp.all(sim.controllable[0] == can_control_1[i]), f"Controllable 1 mismatch at t={i}"
        assert jnp.all(sim.controllable[1] == can_control_2[i]), f"Controllable 2 mismatch at t={i}"
        cmd_fn(cmd)
        sim.step()
        sim_cmd = getattr(sim.controls, control)[0]
        if can_control_1[i]:
            assert jnp.all(sim_cmd == cmd[0]), f"Controls do not match at t={i}"
        else:
            assert not jnp.all(sim_cmd == cmd[0]), f"Controls shouldn't match at t={i}"
        sim_cmd = getattr(sim.controls, control)[1]
        if can_control_2[i]:
            assert jnp.all(sim_cmd == cmd[1]), f"Controls do not match at t={i}"
        else:
            assert not jnp.all(sim_cmd == cmd[1]), f"Controls shouldn't match at t={i}"
        if i == 0:
            sim.reset(np.array([False, True]))  # Make world 2 asynchronous


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_sim_state_control(device: str):
    skip_unavailable_device(device)
    sim = Sim(n_worlds=2, n_drones=3, control=Control.state, device=device)
    cmd = np.random.rand(sim.n_worlds, sim.n_drones, 13)
    sim.state_control(cmd)
    assert isinstance(sim.controls.staged_state, jnp.ndarray), "Buffers must remain JAX arrays"
    assert jnp.all(sim.controls.staged_state == cmd), "Buffers must match command"


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_sim_attitude_control(device: str):
    skip_unavailable_device(device)
    sim = Sim(n_worlds=2, n_drones=3, control=Control.attitude, device=device)
    cmd = np.random.rand(sim.n_worlds, sim.n_drones, 4)
    sim.attitude_control(cmd)
    assert isinstance(sim.controls.staged_attitude, jnp.ndarray), "Buffers must remain JAX arrays"
    assert jnp.all(sim.controls.staged_attitude == cmd), "Buffer must match command"


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


@pytest.mark.unit
@pytest.mark.parametrize("physics", [Physics.sys_id, Physics.analytical])
def test_control_frequency(physics: Physics):
    # Create two sims with different frequencies
    sim_500 = Sim(freq=500, physics=physics, control="state")
    sim_1000 = Sim(freq=1000, physics=physics, control="state")

    # Set same initial state and controls
    cmd = np.zeros((1, 1, 13))  # Single world, single drone, state control
    # Target position of (1, 1, 1). Needs to be off-center to check attitude integration error
    cmd[..., :3] = 1.0

    # Run both sims for one control cycle
    sim_500.state_control(cmd)
    sim_500.step()

    sim_1000.state_control(cmd)
    sim_1000.step()
    sim_1000.step()

    # Check that the controls are the same
    assert np.all(sim_500.controls.rpms == sim_1000.controls.rpms), "Control mismatch"
    assert np.all(sim_500.controls.thrust == sim_1000.controls.thrust), "Control mismatch"
    assert np.all(sim_500.controls.attitude == sim_1000.controls.attitude), "Control mismatch"
    assert np.all(sim_500.controls.pos_err_i == sim_1000.controls.pos_err_i), "Control mismatch"
    assert np.all(sim_500.controls.rpy_err_i == sim_1000.controls.rpy_err_i), "Control mismatch"

    sim_500.close()
    sim_1000.close()
