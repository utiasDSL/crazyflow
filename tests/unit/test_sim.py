"""Unit tests for the simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from conftest import skip_if_headless
from jax import Array

from crazyflow.control import Control
from crazyflow.exception import ConfigError
from crazyflow.sim import Dynamics, Sim
from crazyflow.sim.data import ControlData, SimData
from crazyflow.sim.integration import Integrator
from crazyflow.sim.sim import rotor_vel_limits, sync_sim2mjx, use_box_collision
from crazyflow.sim.visualize import change_material

if TYPE_CHECKING:
    from typing import Any


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
@pytest.mark.parametrize("dynamics", Dynamics)
@pytest.mark.parametrize("control", Control)
@pytest.mark.parametrize("n_worlds", [1, 2])
def test_sim_init(dynamics: Dynamics, device: str, control: Control, n_worlds: int):
    n_drones = 1

    if dynamics != Dynamics.first_principles:
        if control in (Control.force_torque, Control.rotor_vel):
            with pytest.raises(ConfigError):
                Sim(n_worlds=n_worlds, dynamics=dynamics, device=device, control=control)
            return

    sim = Sim(n_worlds=n_worlds, dynamics=dynamics, device=device, control=control)
    assert sim.n_worlds == n_worlds
    assert sim.n_drones == n_drones
    assert sim.device == jax.devices(device)[0]
    assert sim.dynamics == dynamics

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
    if control in (Control.state, Control.attitude, Control.force_torque):
        ft_ctrl = sim.data.controls.force_torque
        assert isinstance(ft_ctrl, ControlData)
        array_meta_assert(ft_ctrl.cmd, (n_worlds, n_drones, 4), device)
        array_meta_assert(ft_ctrl.staged_cmd, (n_worlds, n_drones, 4), device)


@pytest.mark.unit
@pytest.mark.parametrize("dynamics", Dynamics)
@pytest.mark.parametrize("n_worlds", [1, 2])
@pytest.mark.parametrize("n_drones", [1, 3])
def test_reset(device: str, dynamics: Dynamics, n_worlds: int, n_drones: int):
    """Test that reset without mask resets all worlds to default state."""
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, dynamics=dynamics, device=device)

    # Modify states
    data = sim.data
    states, controls, params, core = data.states, data.controls, data.params, data.core
    core = core.replace(steps=core.steps + 100)
    controls = controls.replace(state=None)
    controls = controls.replace(
        force_torque=controls.force_torque.replace(cmd=jnp.ones((n_worlds, n_drones, 4)))
    )
    states = states.replace(pos=states.pos.at[:, :, 2].set(1.0))
    mass = jnp.broadcast_to(params.mass, (n_worlds, n_drones, 1)).at[:, n_drones - 1].set(1.0)
    params = params.replace(mass=mass)  # Per-drone masses
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
@pytest.mark.parametrize("dynamics", Dynamics)
def test_reset_masked(device: str, dynamics: Dynamics):
    """Test that reset with mask only resets specified worlds."""
    sim = Sim(n_worlds=2, n_drones=1, dynamics=dynamics, device=device)

    # Modify states
    data = sim.data
    states, controls, params, core = data.states, data.controls, data.params, data.core
    # Use per-drone masses to verify per-world params resets
    params = params.replace(mass=jnp.broadcast_to(params.mass, (sim.n_worlds, sim.n_drones, 1)))
    sim.data = data = data.replace(params=params)
    sim.build_default_data()
    core = core.replace(steps=core.steps + 100)
    controls = controls.replace(state=None)
    controls = controls.replace(force_torque=controls.force_torque.replace(cmd=jnp.ones((2, 1, 4))))
    controls = controls.replace(force_torque=controls.force_torque.replace(steps=jnp.ones((2, 1))))
    states = states.replace(pos=states.pos.at[:, :, 2].set(1.0))
    params = params.replace(mass=params.mass.at[..., 0].set(1.0))
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
    assert jnp.all(data.params.mass[1, 0, 0] == 1.0), "World 2 mass was reset"
    assert jnp.all(data.core.steps[1] == 100), "World 2 steps were reset"
    assert data.controls.force_torque.steps[1] == 1, "World 2 force torque steps were reset"


@pytest.mark.unit
@pytest.mark.parametrize("n_worlds", [1, 2])
@pytest.mark.parametrize("n_drones", [1, 3])
@pytest.mark.parametrize("dynamics", Dynamics)
@pytest.mark.parametrize("control", Control)
def test_sim_step(n_worlds: int, n_drones: int, dynamics: Dynamics, control: Control, device: str):
    if dynamics != Dynamics.first_principles:
        if control in (Control.force_torque, Control.rotor_vel):
            pytest.skip(f"{control} is not supported with non-first-principles dynamics")

    sim = Sim(
        n_worlds=n_worlds, n_drones=n_drones, dynamics=dynamics, device=device, control=control
    )
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
def test_sim_attitude_control_device(device: str):
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
def test_sim_state_control_device(device: str):
    sim = Sim(n_worlds=2, n_drones=3, control=Control.state, device=device)
    cmd = np.random.rand(sim.n_worlds, sim.n_drones, 13)
    sim.state_control(cmd)
    controls = sim.data.controls.state
    assert isinstance(controls.cmd, jnp.ndarray), "Buffers must remain JAX arrays"
    assert isinstance(controls.staged_cmd, jnp.ndarray), "Buffers must remain JAX arrays"
    assert jnp.all(controls.staged_cmd == cmd), "Buffers must match command"


@pytest.mark.render
@skip_if_headless
def test_render_human(device: str):
    sim = Sim(device=device)
    sim.render()
    sim.viewer.close()


@skip_if_headless
def test_render_rgb_array(device: str):
    sim = Sim(n_worlds=2, device=device)
    img = sim.render(mode="rgb_array", width=1024, height=1024)
    assert isinstance(img, np.ndarray), "Image must be a numpy array"
    assert img.shape == (1024, 1024, 3), f"Unexpected image shape {img.shape}"
    # Check if mj_model.vis.global_.offwidth is set correctly
    assert not all(img[0, 0, :] == 0), "Image contains black patches"
    assert not all(img[-1, -1, :] == 0), "Image contains black patches"


@pytest.mark.unit
def test_device(device: str):
    sim = Sim(n_worlds=2, dynamics=Dynamics.so_rpy, device=device)
    sim.step()
    assert sim.data.states.pos.device == jax.devices(device)[0]


@pytest.mark.unit
@pytest.mark.parametrize("n_worlds", [1, 2])
@pytest.mark.parametrize("n_drones", [1, 3])
def test_sync_shape_consistency(device: str, n_drones: int, n_worlds: int):
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, dynamics=Dynamics.so_rpy, device=device)
    qpos_shape, qvel_shape = sim.mjx_data.qpos.shape, sim.mjx_data.qvel.shape
    _, mjx_data = sync_sim2mjx(sim.data, sim.mjx_data, sim.mjx_model)
    assert mjx_data.qpos.shape == qpos_shape, "sync_sim2mjx() should not change qpos shape"
    assert mjx_data.qvel.shape == qvel_shape, "sync_sim2mjx() should not change qvel shape"


@pytest.mark.unit
@pytest.mark.parametrize("dynamics", Dynamics)
def test_control_frequency(dynamics: Dynamics):
    # Create two sims with different frequencies
    sim_500 = Sim(freq=500, dynamics=dynamics, control="state")
    sim_1000 = Sim(freq=1000, dynamics=dynamics, control="state")

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
def test_seed(device: str):
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
@pytest.mark.parametrize("dynamics", Dynamics)
def test_floor_penetration(dynamics: Dynamics):
    """Test that drones cannot penetrate the floor (z < 0.01).

    We don't test for mujoco, as mujoco uses collisions by default and will let the drone bounce on
    the floor.
    """
    sim = Sim(dynamics=dynamics, control=Control.attitude, freq=500, device="cpu")
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
@pytest.mark.parametrize("integrator", Integrator)
def test_rotor_vel_clip(integrator: Integrator):
    """Test that the first-principles rotor state saturates at its physical limits."""
    sim = Sim(
        dynamics=Dynamics.first_principles,
        control=Control.rotor_vel,
        integrator=integrator,
        device="cpu",
    )
    lower, upper = rotor_vel_limits(Dynamics.first_principles, sim.drone)
    assert 0.0 < lower < upper

    # States outside the limits are clipped after the integration
    for value, target in ((2 * upper, upper), (-upper, lower)):
        states = sim.data.states.replace(rotor_vel=jnp.full_like(sim.data.states.rotor_vel, value))
        sim.rotor_vel_control(np.full((1, 1, 4), target))
        sim.data = sim.data.replace(states=states)
        sim.step()
        assert jnp.all(sim.data.states.rotor_vel >= lower)
        assert jnp.all(sim.data.states.rotor_vel <= upper)
    sim.close()


@pytest.mark.unit
@pytest.mark.parametrize(
    "dynamics", [Dynamics.so_rpy, Dynamics.so_rpy_rotor, Dynamics.so_rpy_rotor_drag]
)
@pytest.mark.parametrize("integrator", Integrator)
def test_thrust_clip(dynamics: Dynamics, integrator: Integrator):
    """Test that the thrust state of every so_rpy model is clipped to its physical limits."""
    sim = Sim(dynamics=dynamics, control=Control.attitude, integrator=integrator, device="cpu")
    lower, upper = rotor_vel_limits(dynamics, sim.drone)
    assert 0.0 < lower < upper

    for value, target in ((2 * upper, upper), (-upper, lower)):
        states = sim.data.states.replace(rotor_vel=jnp.full_like(sim.data.states.rotor_vel, value))
        sim.attitude_control(np.array([[[0.0, 0.0, 0.0, target]]]))
        sim.data = sim.data.replace(states=states)
        sim.step()
        assert jnp.all(sim.data.states.rotor_vel >= lower)
        assert jnp.all(sim.data.states.rotor_vel <= upper)
    sim.close()


@pytest.mark.unit
@pytest.mark.parametrize("dynamics", Dynamics)
def test_contacts(dynamics: Dynamics):
    sim = Sim(dynamics=dynamics, control=Control.attitude, freq=500, device="cpu")
    sim.reset()
    sim.step(10)  # Make sure the drone is on the ground
    contacts = sim.contacts()
    # The contact buffer must contain the drone-floor contact
    assert contacts.shape[-1] > 0, "Contact buffer should not be empty"
    assert jnp.all(contacts), "Drone should be in contact with the floor"
    sim.close()


@pytest.mark.unit
@pytest.mark.parametrize("box_collision", [False, True])
def test_contacts_between_drones(box_collision: bool):
    """Two overlapping drones must register a contact.

    Drones are mocap bodies that are all welded to the world. MuJoCo/MJX filter out collisions
    between geoms welded into the same kinematic tree, so without explicit contact pairs no
    drone-drone contact is ever generated.
    """
    sim = Sim(n_drones=2, control=Control.attitude, freq=500, device="cpu")
    if box_collision:
        use_box_collision(sim, True)
    sim.reset()
    states = sim.data.states
    # Place both drones at the same position above the floor
    pos = states.pos.at[:, 1, :].set(states.pos[:, 0, :]).at[..., 2].set(1.0)
    sim.data = sim.data.replace(states=states.replace(pos=pos))
    sim.data = sim.data.replace(core=sim.data.core.replace(mjx_synced=False))
    assert jnp.any(sim.contacts("drone:0")), "Overlapping drones should be in contact"
    assert jnp.any(sim.contacts("drone:1")), "Overlapping drones should be in contact"
    sim.close()


def _attach_obstacle(sim: Sim, pos: list[float], mocap: bool):
    """Attach the minimal obstacle to the sim at ``pos``, mirroring downstream track loading."""
    OBSTACLE_XML = """
    <mujoco>
    <worldbody>
        <body name="obstacle">
        <geom name="obstacle" type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
    </mujoco>
    """
    obstacle_spec = mujoco.MjSpec.from_string(OBSTACLE_XML)
    frame = sim.spec.worldbody.add_frame()
    obstacle = frame.attach_body(obstacle_spec.body("obstacle"), "", ":0")
    obstacle.pos = pos
    obstacle.mocap = mocap  # Mocap bodies (like the gates) are still welded to the world
    sim.build_mjx()


@pytest.mark.unit
@pytest.mark.parametrize("mocap", [True, False])
def test_contacts_with_obstacle(mocap: bool):
    """A drone placed inside an obstacle must register a contact.

    Covers both a mocap obstacle (attached like the gates) and a static one. Both are welded to the
    world just like the drones, so the collision is only detected if explicit contact pairs are
    generated.
    """
    obstacle_pos = [0.0, 0.0, 1.0]
    sim = Sim(control=Control.attitude, freq=500, device="cpu")
    _attach_obstacle(sim, obstacle_pos, mocap=mocap)
    sim.reset()
    states = sim.data.states
    pos = states.pos.at[..., :].set(jnp.array(obstacle_pos))
    sim.data = sim.data.replace(states=states.replace(pos=pos))
    sim.data = sim.data.replace(core=sim.data.core.replace(mjx_synced=False))
    assert jnp.any(sim.contacts("drone:0")), "Drone should collide with the obstacle"
    sim.close()


@pytest.mark.unit
def test_spec_copy_during_attach():
    """sim.spec must copy bodies on attach so the same source body can be attached repeatedly."""
    sim = Sim(n_drones=1, device="cpu")
    box = mujoco.MjSpec.from_string('<mujoco><worldbody><body name="box"/></worldbody></mujoco>')
    frame = sim.spec.worldbody.add_frame()
    for i in range(2):  # copy_during_attach=False would move "box" out, returning None next lookup
        frame.attach_body(box.body("box"), "", f":{i}")
    sim.close()


@pytest.mark.unit
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("n_drones", [3, 17])
def test_drone_meshes_shared(n_drones: int, fused: bool):
    """The drone's visual meshes are shared across instances, not copied per drone.

    Regression test for per-drone mesh duplication: ``attach_body`` copies the drone meshes for
    every instance, which makes ``compile()`` memory and time grow linearly with ``n_drones``. With
    sharing, the compiled mesh count is independent of the number of drones.
    """
    single = Sim(n_drones=1, fused_mjx_model=fused, device="cpu")
    multi = Sim(n_drones=n_drones, fused_mjx_model=fused, device="cpu")
    n_single, n_multi = single.mj_model.nmesh, multi.mj_model.nmesh
    assert n_single > 0, "expected the drone to define visual meshes"
    assert n_multi == n_single, f"mismatched mesh count: {single=} vs {multi=} for {n_drones=}"
    # Every drone is still present as a distinct mocap body with the ``{body}:{i}`` naming.
    assert multi.mj_model.nmocap == n_drones
    body = "drone_fused" if fused else "drone"
    ids = {multi.mj_model.body(f"{body}:{i}").id for i in range(n_drones)}
    assert len(ids) == n_drones, "each drone must be a distinct body"
    # Materials, unlike meshes, are kept per drone so each drone owns a distinct ``{mat}:{i}``
    mat_ids = {
        mujoco.mj_name2id(multi.mj_model, mujoco.mjtObj.mjOBJ_MATERIAL, f"led_bot:{i}")
        for i in range(n_drones)
    }
    assert -1 not in mat_ids and len(mat_ids) == n_drones, "each drone needs its own material"
    single.close()
    multi.close()


@pytest.mark.unit
@pytest.mark.parametrize("control", Control)
def test_data_committed(control: Control, device: str):
    # Check that the data is committed to the device we chose
    sim = Sim(dynamics=Dynamics.first_principles, control=control, freq=500, device=device)

    def assert_committed(obj0: Array | Any, path: str = "data"):
        if isinstance(obj0, jnp.ndarray):
            assert obj0.committed, f"{path} is not committed"
        elif isinstance(obj0, (int, float, bool, str, type(None))):
            pass  # Primitive types are always "committed"
        elif hasattr(obj0, "__dict__"):  # Dataclass
            for attr_name in obj0.__dict__:
                assert_committed(getattr(obj0, attr_name), f"{path}.{attr_name}")
        elif isinstance(obj0, (list, tuple)):  # Handle sequences
            for i, item0 in enumerate(obj0):
                assert_committed(item0, f"{path}[{i}]")
        elif isinstance(obj0, type(sim.device)):  # Device objects
            pass  # Devices themselves don't have committed attribute
        elif isinstance(obj0, dict):
            for key, value0 in obj0.items():
                assert_committed(value0, f"{path}[{key}]")
        else:
            raise TypeError(f"Could not handle type {type(obj0)} at {path}")

    assert_committed(sim.data)


@pytest.mark.unit
@pytest.mark.parametrize("dynamics", Dynamics)
def test_compile(dynamics: Dynamics, device: str):
    sim = Sim(dynamics=dynamics, control=Control.state, freq=500, device=device)
    # Make sure we don't recompile the step function after the first call
    sim.step(1)
    sim.step(1)
    assert sim._step._cache_size() == 1, "Step function should not be recompiled"
    sim.close()


@pytest.mark.unit
@pytest.mark.parametrize("dynamics", Dynamics)
def test_scan_results(dynamics: Dynamics):
    sim = Sim(n_worlds=2, n_drones=3, dynamics=dynamics, control=Control.state, device="cpu")
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


@pytest.mark.unit
@pytest.mark.parametrize("drone", ["cf2x_L250", "cf2x_P250", "cf2x_T350", "cf21B_500"])
@pytest.mark.parametrize("mat_name", ["led_top", "led_bot"])
def test_change_material(device: str, drone: str, mat_name: str):
    """change_material should broadcast RGBA/emission and update MuJoCo materials appropriately."""
    n_drones = 2

    sim = Sim(n_drones=n_drones, drone=drone, device=device)

    drone_ids = np.array([0, 1], dtype=int)
    # Distinct values per drone to assert that each material gets its own value
    rgba = np.stack([0.42 * np.ones(4), 0.73 * np.ones(4)])
    emission = np.array([0.42, 0.73])

    change_material(sim, mat_name=mat_name, drone_ids=drone_ids, rgba=rgba, emission=emission)

    mj_model = sim.mj_model
    mat0 = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_MATERIAL, f"{mat_name}:0")
    mat1 = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_MATERIAL, f"{mat_name}:1")
    assert mat0 != mat1, "drones must not share the same material asset"
    np.testing.assert_allclose(mj_model.mat_rgba[mat0], rgba[0])
    np.testing.assert_allclose(mj_model.mat_rgba[mat1], rgba[1])
    assert mj_model.mat_emission[mat0] == pytest.approx(emission[0])
    assert mj_model.mat_emission[mat1] == pytest.approx(emission[1])


@pytest.mark.unit
def test_change_material_errors(device: str):
    """Test that change_material raises the expected errors for bad inputs."""
    n_drones = 2
    sim = Sim(n_drones=n_drones, device=device)

    drone_ids = np.array([0, 1], dtype=int)
    rgba = np.ones((n_drones, 4), dtype=float)
    emission = np.ones((n_drones,), dtype=float)

    with pytest.raises(ValueError):
        change_material(sim, mat_name="bad_mat", drone_ids=drone_ids, rgba=rgba, emission=emission)

    with pytest.raises(ValueError, match="drone_ids must be 1D array"):
        change_material(
            sim, mat_name="led_top", drone_ids=np.array(2, dtype=int), rgba=rgba, emission=emission
        )

    with pytest.raises(ValueError, match=r"drone_ids must be in range \[0, 1\]"):
        change_material(
            sim, mat_name="led_top", drone_ids=np.arange(3, dtype=int), rgba=rgba, emission=emission
        )


@pytest.mark.unit
@pytest.mark.parametrize("control", Control)
def test_build_data(control: Control):
    sim = Sim(control=control)
    data = sim.build_data()
    assert isinstance(data, SimData), "build_data() must return a SimData instance"
    default_data = sim.build_default_data()
    assert isinstance(default_data, SimData), "build_default_data() must return a SimData instance"


@pytest.mark.unit
@pytest.mark.parametrize("drone", ["cf2x_L250", "cf2x_P250", "cf2x_T350", "cf21B_500"])
def test_fused_model(device: str, drone: str):
    sim = Sim(drone=drone, fused_mjx_model=True, device=device)
    sim.reset()
    sim.step(1)
    sim.close()


@pytest.mark.unit
def test_partial_reset_keeps_shared_arrays():
    # A partial reset with a world mask must leave shared arrays intact
    sim = Sim(n_worlds=3)
    gravity = jnp.array([1.0, 2.0, 3.0])
    sim.data = sim.data.replace(params=sim.data.params.replace(gravity_vec=gravity))
    sim.reset(jnp.array([True, False, False]))
    assert jnp.array_equal(sim.data.params.gravity_vec, gravity)


@pytest.mark.unit
def test_full_reset_restores_shared_arrays():
    sim = Sim(n_worlds=3)
    default_gravity, rng_key = sim.default_data.params.gravity_vec, sim.data.core.rng_key
    sim.data = sim.data.replace(
        params=sim.data.params.replace(gravity_vec=jnp.array([1.0, 2.0, 3.0]))
    )
    sim.reset()
    assert jnp.array_equal(sim.data.params.gravity_vec, default_gravity)
    # The random key is the only thing that does not reset
    assert jnp.array_equal(jax.random.key_data(sim.data.core.rng_key), jax.random.key_data(rng_key))
