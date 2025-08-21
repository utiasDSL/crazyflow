import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from crazyflow.control import Control
from crazyflow.exception import ConfigError
from crazyflow.sim import Physics, Sim
from crazyflow.sim.sim import sync_sim2mjx
from crazyflow.sim.structs import ControlData


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


def skip_headless():
    if os.environ.get("DISPLAY") is None:
        pytest.skip("DISPLAY is not set, skipping test in headless environment")


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

    if physics == Physics.sys_id and control == Control.force_torque:
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
    if control == Control.state:
        assert isinstance(sim.data.controls.state, ControlData)
        array_meta_assert(sim.data.controls.state.staged_cmd, (n_worlds, n_drones, 13), device)
        array_meta_assert(sim.data.controls.state.cmd, (n_worlds, n_drones, 13), device)
    else:
        assert sim.data.controls.state is None
    # Test attitude buffer shapes
    if control in (Control.attitude, Control.state):
        assert isinstance(sim.data.controls.attitude, ControlData)
        array_meta_assert(sim.data.controls.attitude.staged_cmd, (n_worlds, n_drones, 4), device)
        array_meta_assert(sim.data.controls.attitude.cmd, (n_worlds, n_drones, 4), device)
    else:
        assert sim.data.controls.attitude is None

    # Test force torque buffer shapes
    ft_ctrl = sim.data.controls.force_torque
    assert isinstance(ft_ctrl, ControlData)
    array_meta_assert(ft_ctrl.cmd, (n_worlds, n_drones, 4), device)
    array_meta_assert(ft_ctrl.staged_cmd, (n_worlds, n_drones, 4), device)


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
    controls = controls.replace(state=None)
    controls = controls.replace(
        force_torque=controls.force_torque.replace(cmd=jnp.ones((n_worlds, n_drones, 4)))
    )
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
    assert jnp.all(sim.data.controls.force_torque.steps == -1), "Control steps not reset to -1"
    if sim.control in (Control.state, Control.attitude):
        assert jnp.all(sim.data.controls.attitude.steps == -1), "Control steps not reset to -1"
    if sim.control == Control.state:
        assert jnp.all(sim.data.controls.state.steps == -1), "Control steps not reset to -1"


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
    controls = controls.replace(state=None)
    controls = controls.replace(force_torque=controls.force_torque.replace(cmd=jnp.ones((2, 1, 4))))
    controls = controls.replace(force_torque=controls.force_torque.replace(steps=jnp.ones((2, 1))))
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
    assert jnp.all(data.states.pos[1, :, 2] == 1.0), "World 2 pos was reset"
    assert jnp.all(data.controls.force_torque.cmd[1, ...] == 1.0), "World 2 cmd was reset"
    assert jnp.all(data.params.mass[1, 0] == 1.0), "World 2 mass was reset"
    assert jnp.all(data.core.steps[1] == 100), "World 2 steps were reset"
    assert data.controls.force_torque.steps[1] == 1, "World 2 force torque steps were reset"


@pytest.mark.unit
@pytest.mark.parametrize("n_worlds", [1, 2])
@pytest.mark.parametrize("n_drones", [1, 3])
@pytest.mark.parametrize("physics", Physics)
@pytest.mark.parametrize("control", Control)
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_sim_step(n_worlds: int, n_drones: int, physics: Physics, control: Control, device: str):
    skip_unavailable_device(device)
    if physics == Physics.sys_id and control == Control.force_torque:
        pytest.skip("Force-torque control is not supported with sys_id physics")
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
        sim_cmd = sim.data.controls.attitude.cmd[0]
        if can_control_1[i]:
            assert jnp.all(sim_cmd == cmd[0]), f"Controls do not match at t={i}"
        else:
            assert not jnp.all(sim_cmd == cmd[0]), f"Controls shouldn't match at t={i}"
        sim_cmd = sim.data.controls.attitude.cmd[1]
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
    controls = sim.data.controls.attitude
    assert isinstance(controls.staged_cmd, jnp.ndarray), "Buffers must remain JAX arrays"
    assert jnp.all(controls.staged_cmd == cmd), "Buffers must match command"


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
        last_attitude = sim.data.controls.attitude.staged_cmd
        sim.step()
        attitude = sim.data.controls.attitude.staged_cmd
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
    controls = sim.data.controls.state
    assert isinstance(controls.cmd, jnp.ndarray), "Buffers must remain JAX arrays"
    assert isinstance(controls.staged_cmd, jnp.ndarray), "Buffers must remain JAX arrays"
    assert jnp.all(controls.staged_cmd == cmd), "Buffers must match command"


@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.render
def test_render_human(device: str):
    skip_unavailable_device(device)
    sim = Sim(device=device)
    sim.render()
    sim.viewer.close()


# Do not mark as render to ensure it runs by default. This function will not open a viewer.
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_render_rgb_array(device: str):
    skip_unavailable_device(device)
    skip_headless()
    sim = Sim(n_worlds=2, device=device)
    img = sim.render(mode="rgb_array", width=1024, height=1024)
    assert isinstance(img, np.ndarray), "Image must be a numpy array"
    assert img.shape == (1024, 1024, 3), f"Unexpected image shape {img.shape}"
    # Check if mj_model.vis.global_.offwidth is set correctly
    assert not all(img[0, 0, :] == 0), "Image contains black patches"
    assert not all(img[-1, -1, :] == 0), "Image contains black patches"


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_device(device: str):
    skip_unavailable_device(device)
    sim = Sim(n_worlds=2, physics=Physics.sys_id, device=device)
    sim.step()
    assert sim.data.states.pos.device == jax.devices(device)[0]


@pytest.mark.unit
@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("n_worlds", [1, 2])
@pytest.mark.parametrize("n_drones", [1, 3])
def test_sync_shape_consistency(device: str, n_drones: int, n_worlds: int):
    skip_unavailable_device(device)
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, physics=Physics.sys_id, device=device)
    qpos_shape, qvel_shape = sim.mjx_data.qpos.shape, sim.mjx_data.qvel.shape
    _, mjx_data = sync_sim2mjx(sim.data, sim.mjx_data, sim.mjx_model)
    assert mjx_data.qpos.shape == qpos_shape, "sync_sim2mjx() should not change qpos shape"
    assert mjx_data.qvel.shape == qvel_shape, "sync_sim2mjx() should not change qvel shape"


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

    # Check that the controls are the same for state
    state_ctrl_500 = sim_500.data.controls.state
    state_ctrl_1000 = sim_1000.data.controls.state
    assert np.all(state_ctrl_500.cmd == state_ctrl_1000.cmd)
    assert np.all(state_ctrl_500.staged_cmd == state_ctrl_1000.staged_cmd)
    assert np.all(state_ctrl_500.pos_err_i == state_ctrl_1000.pos_err_i)
    # attitude
    att_ctrl_500 = sim_500.data.controls.attitude
    att_ctrl_1000 = sim_1000.data.controls.attitude
    assert np.all(att_ctrl_500.cmd == att_ctrl_1000.cmd)
    assert np.all(att_ctrl_500.staged_cmd == att_ctrl_1000.staged_cmd)
    assert np.all(att_ctrl_500.r_int_error == att_ctrl_1000.r_int_error)
    # and force torque
    ft_ctrl_500 = sim_500.data.controls.force_torque
    ft_ctrl_1000 = sim_1000.data.controls.force_torque
    assert np.all(ft_ctrl_500.cmd == ft_ctrl_1000.cmd)
    assert np.all(ft_ctrl_500.staged_cmd == ft_ctrl_1000.staged_cmd)
    assert np.all(sim_500.data.controls.rotor_vel == sim_1000.data.controls.rotor_vel)
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


@pytest.mark.unit
@pytest.mark.parametrize("physics", [Physics.analytical, Physics.sys_id])
def test_floor_penetration(physics: Physics):
    """Test that drones cannot penetrate the floor (z < 0.01).

    We don't test for mujoco, as mujoco uses collisions by default and will let the drone bounce on
    the floor.
    """
    sim = Sim(physics=physics, control=Control.attitude, freq=500, device="cpu")
    sim.reset()
    # Command to fall: zero thrust and attitude that points downward
    attitude_cmd = np.zeros((1, 1, 4))  # [roll, pitch, yaw, thrust]
    attitude_cmd[..., 0] = 0.0  # Zero thrust to fall
    sim.attitude_control(attitude_cmd)
    # Run simulation for short duration to let drone fall
    for _ in range(5):  # 0.1 seconds at 500Hz
        sim.step(10)
        # Check that drone never goes below floor
        z_pos = sim.data.states.pos[..., 2]
        assert jnp.all(z_pos >= -0.001), f"Drone penetrated floor: z={z_pos.min()}"
    # Check that the drone ended up on the floor (very close to z=0)
    final_z_pos = sim.data.states.pos[..., 2]
    assert jnp.all(final_z_pos == -0.001), f"Drone should be on floor but z={final_z_pos}"
    sim.close()


@pytest.mark.unit
@pytest.mark.parametrize("physics", [Physics.sys_id, Physics.analytical])
def test_contacts(physics: Physics):
    sim = Sim(physics=physics, control=Control.attitude, freq=500, device="cpu")
    sim.reset()
    sim.step(10)  # Make sure the drone is on the ground
    contacts = sim.contacts()
    assert jnp.all(contacts), "Drone should be in contact with the floor"
    sim.close()


@pytest.mark.unit
@pytest.mark.parametrize("physics", [Physics.sys_id, Physics.analytical])
def test_compile(physics: Physics):
    sim = Sim(physics=physics, control=Control.attitude, freq=500, device="cpu")
    # Make sure we don't recompile the step function after the first call
    sim.step(1)
    sim.step(1)
    assert sim._step._cache_size() == 1, "Step function should not be recompiled"


@pytest.mark.unit
@pytest.mark.parametrize("physics", Physics)
def test_scan_results(physics: Physics):
    sim = Sim(n_worlds=2, n_drones=3, physics=physics, control=Control.state, device="cpu")
    sim.reset()
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
    cmd[..., :3] = sim.data.states.pos + np.array([0.3, 0.3, 0.3])
    sim.state_control(cmd)
    n_steps, n_iters = sim.freq // sim.control_freq, 100  # 1 second at 100Hz
    for _ in range(n_iters):
        sim.step(n_steps)
    pos_loop_steps = sim.data.states.pos
    sim.reset()
    sim.state_control(cmd)
    sim.step(n_steps * n_iters)
    pos_scan_steps = sim.data.states.pos
    assert np.all(pos_loop_steps[..., 2] > 0.1), "Drones should have moved"
    assert np.allclose(pos_scan_steps, pos_loop_steps), "Scan results should be identical"
    sim.close()
