import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from crazyflow.control import Control
from crazyflow.exception import ConfigError
from crazyflow.sim import Physics, Sim


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


def array_meta_assert(
    x: Array,
    shape: tuple[int, ...] | None = None,
    device: str | None = None,
    name: str | None = None,
):
    """Assert that the array has the correct metadata (shape and device)."""
    prefix = f"{name}: " if name is not None else ""
    assert isinstance(x, jnp.ndarray), f"{prefix}x must be a JAX array, is {type(x)}"
    if shape is not None:
        assert x.shape == shape, f"{prefix}Shape mismatch {x.shape} {shape}"
    if device is not None:
        device = jax.devices(device)[0]
        assert x.device == device, f"{prefix}Device mismatch {x.device} {device}"


def array_compare_assert(x: Array, y: Array, value: bool = True, name: str | None = None):
    """Assert that the arrays are comparable (shape and device must match, value is optional)."""
    prefix = f"{name}: " if name is not None else ""
    assert type(x) is type(y), f"{prefix}Types mismatch {type(x)} {type(y)}"
    assert x.shape == y.shape, f"{prefix}Shape mismatch {x.shape} {y.shape}"
    assert x.device == y.device, f"{prefix}Device mismatch {x.device} {y.device}"
    if value:
        assert jnp.all(x == y), f"{prefix}Value mismatch {x} {y}"


@pytest.mark.unit
@pytest.mark.parametrize("physics", Physics)
@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("control", Control)
@pytest.mark.parametrize("n_worlds", [1, 2])
def test_sim_init(physics: Physics, device: str, control: Control, n_worlds: int):
    n_drones = 1
    skip_unavailable_device(device)

    if physics == Physics.sys_id and control == Control.thrust:
        with pytest.raises(ConfigError):
            Sim(n_worlds=n_worlds, physics=physics, device=device, control=control)
        return
    sim = Sim(n_worlds=n_worlds, physics=physics, device=device, control=control)
    assert sim.n_worlds == n_worlds
    assert sim.n_drones == n_drones
    assert sim.device == jax.devices(device)[0]
    assert sim.physics == physics

    # Test state buffer shapes
    array_meta_assert(sim.data.states.pos, (n_worlds, n_drones, 3), device, "pos")
    array_meta_assert(sim.data.states.quat, (n_worlds, n_drones, 4), device, "quat")
    array_meta_assert(sim.data.states.vel, (n_worlds, n_drones, 3), device, "vel")
    array_meta_assert(sim.data.states.ang_vel, (n_worlds, n_drones, 3), device, "ang_vel")

    # Test control buffer shapes
    array_meta_assert(sim.data.controls.attitude, (n_worlds, n_drones, 4), device)
    array_meta_assert(sim.data.controls.thrust, (n_worlds, n_drones, 4), device)
    array_meta_assert(sim.data.controls.state, (n_worlds, n_drones, 13), device)


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("physics", Physics)
@pytest.mark.parametrize("n_worlds", [1, 2])
@pytest.mark.parametrize("n_drones", [1, 3])
def test_reset(device: str, physics: Physics, n_worlds: int, n_drones: int):
    """Test that reset without mask resets all worlds to default state."""
    skip_unavailable_device(device)
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=physics, device=device)

    # Modify states
    data = sim.data
    states, controls, params, core = data.states, data.controls, data.params, data.core
    core = core.replace(steps=core.steps + 100)
    attitude = controls.attitude.at[:, :, 2].set(1.0)
    staged_attitude = jnp.ones_like(controls.staged_attitude)
    controls = controls.replace(staged_attitude=staged_attitude, attitude=attitude)
    states = states.replace(pos=states.pos.at[:, :, 2].set(1.0))
    params = params.replace(mass=params.mass.at[:, n_drones - 1].set(1.0))
    sim.data = data.replace(states=states, controls=controls, params=params, core=core)
    sim.reset()

    data = jax.tree.flatten_with_path(sim.data)[0]
    default_data = jax.tree.flatten(sim.default_data)[0]
    for i, (path, value) in enumerate(data):
        default_value = default_data[i]
        if isinstance(value, jnp.ndarray):
            array_compare_assert(value, default_value, name=path)
        else:
            assert value == default_value, f"{path} value mismatch"

    assert jnp.all(sim.data.core.steps == 0), "Steps must be reset to 0"
    assert jnp.all(sim.data.controls.attitude_steps == -1), "Control steps not reset to -1"


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("physics", Physics)
def test_reset_masked(device: str, physics: Physics):
    """Test that reset with mask only resets specified worlds."""
    skip_unavailable_device(device)
    sim = Sim(n_worlds=2, n_drones=1, physics=physics, device=device)

    # Modify states
    data = sim.data
    states, controls, params, core = data.states, data.controls, data.params, data.core
    core = core.replace(steps=core.steps + 100)
    attitude = controls.attitude.at[:, :, 2].set(1.0)
    staged_attitude = jnp.ones_like(controls.staged_attitude)
    controls = controls.replace(staged_attitude=staged_attitude, attitude=attitude)
    states = states.replace(pos=states.pos.at[:, :, 2].set(1.0))
    params = params.replace(mass=params.mass.at[:, :, 0].set(1.0))
    sim.data = data.replace(states=states, controls=controls, params=params, core=core)

    # Reset only first world
    mask = jnp.array([True, False])
    sim.reset(mask)

    # Check world 1 was reset to defaults
    data = jax.tree.flatten_with_path(sim.data)[0]
    default_data = jax.tree.flatten(sim.default_data)[0]
    for i, (path, value) in enumerate(data):
        default_value = default_data[i]
        if isinstance(value, jnp.ndarray):
            array_compare_assert(value, default_value, name=path, value=False)
            # Do not check zero-shaped arrays common to all worlds
            if value.ndim >= 1 and default_value.shape[0] > 0:
                # Only check values for the first world
                assert jnp.all(value[0] == default_value[0]), f"{path} value mismatch"
        else:
            assert value == default_value, f"{path} value mismatch"

    # Check world 2 kept modifications
    data = sim.data
    assert jnp.all(data.states.pos[1, :, 2] == 1.0), "World 2 pos should be unchanged"
    assert jnp.all(data.controls.attitude[1, :, 2] == 1.0), "World 2 attitude should be unchanged"
    assert jnp.all(data.params.mass[1, :, 0] == 1.0), "World 2 mass should be unchanged"


@pytest.mark.unit
@pytest.mark.parametrize("n_worlds", [1, 2])
@pytest.mark.parametrize("n_drones", [1, 3])
@pytest.mark.parametrize("physics", Physics)
@pytest.mark.parametrize("control", Control)
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_sim_step(n_worlds: int, n_drones: int, physics: Physics, control: Control, device: str):
    skip_unavailable_device(device)
    if physics == Physics.sys_id and control == Control.thrust:
        return
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=physics, device=device, control=control)
    sim.step(2)


@pytest.mark.unit
@pytest.mark.parametrize("attitude_freq", [33, 50, 100, 200])
def test_sim_attitude_control(attitude_freq: int):
    sim = Sim(n_worlds=2, n_drones=3, control="attitude", freq=100, attitude_freq=attitude_freq)

    can_control_1 = np.arange(6) * attitude_freq % sim.freq < attitude_freq
    can_control_2 = np.array([0, 0, 1, 2, 3, 4]) * attitude_freq % sim.freq < attitude_freq
    for i in range(6):
        cmd = np.random.rand(sim.n_worlds, sim.n_drones, 4)
        assert jnp.all(sim.controllable[0] == can_control_1[i]), f"Controllable 1 mismatch at t={i}"
        assert jnp.all(sim.controllable[1] == can_control_2[i]), f"Controllable 2 mismatch at t={i}"
        sim.attitude_control(cmd)
        sim.step()
        sim_cmd = sim.data.controls.attitude[0]
        if can_control_1[i]:
            assert jnp.all(sim_cmd == cmd[0]), f"Controls do not match at t={i}"
        else:
            assert not jnp.all(sim_cmd == cmd[0]), f"Controls shouldn't match at t={i}"
        sim_cmd = sim.data.controls.attitude[1]
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
    controls = sim.data.controls
    assert isinstance(controls.staged_attitude, jnp.ndarray), "Buffers must remain JAX arrays"
    assert jnp.all(controls.staged_attitude == cmd), "Buffers must match command"


@pytest.mark.unit
@pytest.mark.parametrize("state_freq", [33, 50, 100, 200])
def test_sim_state_control(state_freq: int):
    sim = Sim(n_worlds=2, n_drones=3, control=Control.state, freq=100, state_freq=state_freq)
    can_control_1 = np.arange(6) * state_freq % sim.freq < state_freq
    can_control_2 = np.array([0, 0, 1, 2, 3, 4]) * state_freq % sim.freq < state_freq
    for i in range(6):
        cmd = np.random.rand(sim.n_worlds, sim.n_drones, 13)
        assert jnp.all(sim.controllable[0] == can_control_1[i]), f"Controllable 1 mismatch at t={i}"
        assert jnp.all(sim.controllable[1] == can_control_2[i]), f"Controllable 2 mismatch at t={i}"
        sim.state_control(cmd)
        last_attitude = sim.data.controls.staged_attitude
        sim.step()
        attitude = sim.data.controls.staged_attitude
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
    controls = sim.data.controls
    assert isinstance(controls.state, jnp.ndarray), "Buffers must remain JAX arrays"
    assert jnp.all(controls.state == cmd), "Buffers must match command"


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
    assert sim.data.states.pos.device == jax.devices(device)[0]
    assert sim.data.mjx_data.qpos.device == jax.devices(device)[0]


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("n_worlds", [1, 2])
@pytest.mark.parametrize("n_drones", [1, 3])
def test_shape_consistency(device: str, n_drones: int, n_worlds: int):
    skip_unavailable_device(device)
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=Physics.sys_id, device=device)
    qpos_shape, qvel_shape = sim.data.mjx_data.qpos.shape, sim.data.mjx_data.qvel.shape
    sim.step()
    assert sim.data.mjx_data.qpos.shape == qpos_shape, "step() should not change qpos shape"
    assert sim.data.mjx_data.qvel.shape == qvel_shape, "step() should not change qvel shape"


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
    sim_1000.step(2)

    # Check that the controls are the same
    assert np.all(sim_500.data.controls.rpms == sim_1000.data.controls.rpms)
    assert np.all(sim_500.data.controls.thrust == sim_1000.data.controls.thrust)
    assert np.all(sim_500.data.controls.attitude == sim_1000.data.controls.attitude)
    assert np.all(sim_500.data.controls.pos_err_i == sim_1000.data.controls.pos_err_i)
    assert np.all(sim_500.data.controls.rpy_err_i == sim_1000.data.controls.rpy_err_i)

    sim_500.close()
    sim_1000.close()


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_seed(device: str):
    skip_unavailable_device(device)
    sim = Sim(rng_key=42, device=device)
    assert (jax.random.key_data(sim.data.core.rng_key)[1] == 42).all(), "rng_key not set correctly"
    assert sim.data.core.rng_key.device == sim.device, "__init__() must set device of rng_key"
    # Test seed() method
    sim.seed(43)
    assert (jax.random.key_data(sim.data.core.rng_key)[1] == 43).all(), "seed() doesn't set rng_key"
    assert sim.data.core.rng_key.device == sim.device, "seed() changes device of rng_key"
    sim.close()


@pytest.mark.unit
def test_seed_reset():
    sim = Sim(rng_key=42)
    sim.seed(43)
    sim.reset()
    rng_key = jax.random.key_data(sim.data.core.rng_key)[1]
    assert (rng_key == 43).all(), "rng_key was overwritten by reset()"
