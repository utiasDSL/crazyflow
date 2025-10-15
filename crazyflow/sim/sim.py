from __future__ import annotations

from functools import partial, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

import drone_models
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
from drone_controllers.mellinger import (
    attitude2force_torque,
    force_torque2rotor_vel,
    state2attitude,
)
from einops import rearrange
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from jax import Array, Device

from crazyflow.control.control import Control, controllable
from crazyflow.exception import ConfigError, NotInitializedError
from crazyflow.sim.data import SimControls, SimCore, SimData, SimParams, SimState, SimStateDeriv
from crazyflow.sim.integration import Integrator, euler, rk4, symplectic_euler
from crazyflow.sim.physics import (
    Physics,
    first_principles_physics,
    so_rpy_physics,
    so_rpy_rotor_drag_physics,
    so_rpy_rotor_physics,
)
from crazyflow.utils import grid_2d, leaf_replace, pytree_replace, to_device

if TYPE_CHECKING:
    from mujoco.mjx import Data, Model
    from numpy.typing import NDArray

    from crazyflow.control.mellinger import (
        MellingerAttitudeData,
        MellingerForceTorqueData,
        MellingerStateData,
    )

Params = ParamSpec("Params")  # Represents arbitrary parameters
Return = TypeVar("Return")  # Represents the return type


def requires_mujoco_sync(fn: Callable[Params, Return]) -> Callable[Params, Return]:
    """Decorator to ensure that the simulation data is synchronized with the MuJoCo mjx data."""

    @wraps(fn)
    def wrapper(sim: Sim, *args: Any, **kwargs: Any) -> SimData:
        if not sim.data.core.mjx_synced:
            sim.data, sim.mjx_data = sync_sim2mjx(sim.data, sim.mjx_data, sim.mjx_model)
        return fn(sim, *args, **kwargs)

    return wrapper


class Sim:
    def __init__(
        self,
        n_worlds: int = 1,
        n_drones: int = 1,
        drone_model: str = "cf2x_L250",
        physics: Physics = Physics.default,
        control: Control = Control.default,
        integrator: Integrator = Integrator.default,
        freq: int = 500,
        state_freq: int = 100,
        attitude_freq: int = 500,
        force_torque_freq: int = 500,
        device: str = "cpu",
        xml_path: Path | None = None,
        rng_key: int = 0,
    ):
        assert Physics(physics) in Physics, f"Physics mode {physics} not implemented"
        assert Control(control) in Control, f"Control mode {control} not implemented"
        if physics != Physics.first_principles and control == Control.force_torque:
            raise ConfigError("Force-torque control requires first principles physics")
        if freq > 10_000 and not jax.config.jax_enable_x64:
            raise ConfigError("High frequency simulations require double precision mode")
        self.physics = physics
        self.control = control
        self.drone_model = drone_model
        self.integrator = integrator
        self.device = jax.devices(device)[0]
        self.n_worlds = n_worlds
        self.n_drones = n_drones
        self.freq = freq
        self.max_visual_geom = 1000

        # Initialize MuJoCo world and data
        self._xml_path = xml_path or Path(__file__).parents[1] / "scene.xml"
        self.drone_path = Path(drone_models.__file__).parent / "data" / f"{drone_model}.xml"
        self.spec = self.build_mjx_spec()
        self.mj_model, self.mj_data, self.mjx_model, self.mjx_data = self.build_mjx_model(self.spec)
        self.viewer: MujocoRenderer | None = None

        self.data = self.init_data(state_freq, attitude_freq, force_torque_freq, rng_key)
        self.default_data: SimData
        self.build_default_data()

        # Build the simulation pipeline and overwrite the default _step implementation with it
        self.reset_pipeline: tuple[Callable[[SimData, Array[bool] | None], SimData], ...] = tuple()
        self.step_pipeline: tuple[Callable[[SimData], SimData], ...] = tuple()
        # The ``select_xxx_fn`` methods return functions, not the results of calling those
        # functions. They act as factories that produce building blocks for the construction of our
        # simulation pipeline.
        self.step_pipeline += build_control_fns(self.control, self.physics)
        physics_fn = select_physics_fn(self.physics)
        self.step_pipeline += (select_integrate_fn(self.integrator, physics_fn),)
        self.step_pipeline += (increment_steps,)
        # We never drop below -0.001 (drones can't pass through the floor). We use -0.001 to
        # enable checks for negative z sign
        self.step_pipeline += (clip_floor_pos,)

        self.build_reset_fn()
        self.build_step_fn()

    def reset(self, mask: Array | None = None):
        """Reset the simulation to the initial state.

        Args:
            mask: Boolean array of shape (n_worlds, ) that indicates which worlds to reset. If None,
                all worlds are reset.
        """
        assert mask is None or mask.shape == (self.n_worlds,), f"Mask shape mismatch {mask.shape}"
        self.data = self._reset(self.data, self.default_data, mask)

    def step(self, n_steps: int = 1):
        """Simulate all drones in all worlds for n time steps."""
        assert n_steps > 0, "Number of steps must be positive"
        self.data = self._step(self.data, n_steps=n_steps)

    def state_control(self, controls: Array):
        """Set the desired state for all drones in all worlds."""
        assert controls.shape == (self.n_worlds, self.n_drones, 13), "controls shape mismatch"
        assert self.control == Control.state, "State control is not enabled by the sim config"
        controls = to_device(controls, self.device)
        self.data = self.data.replace(
            controls=self.data.controls.replace(
                state=self.data.controls.state.replace(staged_cmd=controls)
            )
        )

    def attitude_control(self, controls: Array):
        """Set the desired attitude for all drones in all worlds.

        We need to stage the attitude controls because the sys_id physics mode operates directly on
        the attitude controls. If we were to directly update the controls, this would effectively
        bypass the control frequency and run the attitude controller at the physics update rate. By
        staging the controls, we ensure that the physics module sees the old controls until the
        controller updates at its correct frequency.
        """
        assert controls.shape == (self.n_worlds, self.n_drones, 4), "controls shape mismatch"
        assert self.control == Control.attitude, "Attitude control is not enabled by the sim config"
        controls = to_device(controls, self.device)
        self.data = self.data.replace(
            controls=self.data.controls.replace(
                attitude=self.data.controls.attitude.replace(staged_cmd=controls)
            )
        )

    def force_torque_control(self, cmd: Array):
        """Set the desired force and torque for all drones in all worlds."""
        assert cmd.shape == (self.n_worlds, self.n_drones, 4), "Command shape mismatch"
        assert self.control == Control.force_torque, (
            "Force-torque control is not enabled by the sim config"
        )
        controls = to_device(cmd, self.device)
        self.data = self.data.replace(
            controls=self.data.controls.replace(
                force_torque=self.data.controls.force_torque.replace(staged_cmd=controls)
            )
        )

    @requires_mujoco_sync
    def render(
        self,
        mode: str | None = "human",
        world: int = 0,
        default_cam_config: dict | None = None,
        width: int = 1920,
        height: int = 1080,
    ) -> NDArray | None:
        if self.viewer is None:
            self.mj_model.vis.global_.offwidth = width
            self.mj_model.vis.global_.offheight = height
            self.viewer = MujocoRenderer(
                self.mj_model,
                self.mj_data,
                max_geom=self.max_visual_geom,
                default_cam_config=default_cam_config,
                height=height,
                width=width,
            )
        self.mj_data.qpos[:] = self.mjx_data.qpos[world, :]
        self.mj_data.mocap_pos[:] = self.mjx_data.mocap_pos[world, :]
        self.mj_data.mocap_quat[:] = self.mjx_data.mocap_quat[world, :]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.viewer.render(mode)

    def seed(self, seed: int):
        """Set the JAX rng key for the simulation.

        Args:
            seed: The seed for the JAX rng.
        """
        self.data: SimData = seed_sim(self.data, seed, self.device)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def build_mjx_spec(self) -> mujoco.MjSpec:
        """Build the MuJoCo model specification for the simulation."""
        assert self._xml_path.exists(), f"Model file {self._xml_path} does not exist"
        spec = mujoco.MjSpec.from_file(str(self._xml_path))
        spec.option.timestep = 1 / self.freq
        spec.copy_during_attach = True
        drone_spec = mujoco.MjSpec.from_file(str(self.drone_path))
        frame = spec.worldbody.add_frame(name="world")
        # Add drones and their actuators
        for i in range(self.n_drones):
            drone_body = drone_spec.body("drone")
            if drone_body is None:
                raise ValueError("Drone body not found in drone spec")
            drone = frame.attach_body(drone_body, "", f":{i}")
            drone.add_freejoint()
        return spec

    def build_mjx_model(self, spec: mujoco.MjSpec) -> tuple[Any, Any, Model, Data]:
        """Build the MuJoCo model and data structures for the simulation."""
        mj_model = spec.compile()
        mj_data = mujoco.MjData(mj_model)
        mjx_model = mjx.put_model(mj_model, device=self.device)
        mjx_data = mjx.put_data(mj_model, mj_data, device=self.device)
        mjx_data = jax.vmap(lambda _: mjx_data)(jnp.arange(self.n_worlds))
        return mj_model, mj_data, mjx_model, mjx_data

    def build_step_fn(self):
        """Setup the chain of functions that are called in Sim.step().

        We know all the functions that are called in succession since the simulation is configured
        at initialization time. Instead of branching through options at runtime, we construct a step
        function at initialization that selects the correct functions based on the settings.

        Warning:
            If any settings change, the pipeline of functions needs to be reconstructed.
        """
        pipeline = self.step_pipeline

        # None is required by jax.lax.scan to unpack the tuple returned by single_step.
        def single_step(data: SimData, _: None) -> tuple[SimData, None]:
            for fn in pipeline:
                data = fn(data)
            return data, None

        # ``scan`` allows us control over loop unrolling for single steps from a single WhileOp to
        # complete unrolling, reducing either compilation times or fusing the loops to give XLA
        # maximum freedom to reorder operations and jointly optimize the pipeline. This is especially
        # relevant for the common use case of running multiple sim steps in an outer loop, e.g. in
        # gym environments.
        # Having n_steps as a static argument is fine, since patterns with n_steps > 1 will almost
        # always use the same n_steps value for successive calls.
        @partial(jax.jit, static_argnames="n_steps")
        def step(data: SimData, n_steps: int = 1) -> SimData:
            data, _ = jax.lax.scan(single_step, data, length=n_steps, unroll=1)
            data = data.replace(core=data.core.replace(mjx_synced=False))  # Flag mjx data as stale
            return data

        self._step = step

    def build_reset_fn(self):
        """Build the reset function for the current simulation configuration."""
        pipeline = self.reset_pipeline

        @jax.jit
        def reset(data: SimData, default_data: SimData, mask: Array | None = None) -> SimData:
            data = pytree_replace(data, default_data, mask)  # Does not overwrite rng_key
            for fn in pipeline:
                data = fn(data, mask)
            data = data.replace(core=data.core.replace(mjx_synced=False))  # Flag mjx data as stale
            return data

        self._reset = reset

    def build_data(self):
        self.data = self.init_data(
            self.data.controls.state_freq,
            self.data.controls.attitude_freq,
            self.data.controls.force_torque_freq,
            self.data.core.rng_key,
        )

    def build_default_data(self):
        """Initialize the default data for the simulation."""
        self.default_data = self.data.replace()

    def build_mjx(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.mj_model, self.mj_data, self.mjx_model, self.mjx_data = self.build_mjx_model(self.spec)

    def init_data(
        self, state_freq: int, attitude_freq: int, force_torque_freq: int, rng_key: Array
    ) -> SimData:
        """Initialize the simulation data."""
        drone_ids = [self.mj_model.body(f"drone:{i}").id for i in range(self.n_drones)]
        N, D = self.n_worlds, self.n_drones
        data = SimData(
            states=SimState.create(N, D, self.device),
            states_deriv=SimStateDeriv.create(N, D, self.device),
            controls=SimControls.create(
                N,
                D,
                self.control,
                self.drone_model,
                state_freq,
                attitude_freq,
                force_torque_freq,
                self.device,
            ),
            params=SimParams.create(N, D, self.physics, self.drone_model, self.device),
            core=SimCore.create(self.freq, N, D, drone_ids, rng_key, self.device),
        )
        if D > 1:  # If multiple drones, arrange them in a grid
            grid = grid_2d(D)
            states = data.states.replace(pos=data.states.pos.at[..., :2].set(grid))
            data = data.replace(states=states)
        return data

    @property
    def time(self) -> Array:
        return self.data.core.steps / self.data.core.freq

    @property
    def control_freq(self) -> int:
        if self.control == Control.state:
            return self.data.controls.state.freq
        if self.control == Control.attitude:
            return self.data.controls.attitude.freq
        if self.control == Control.force_torque:
            return self.data.controls.force_torque.freq
        raise NotImplementedError(f"Control mode {self.control} not implemented")

    @property
    def controllable(self) -> Array:
        """Boolean array of shape (n_worlds,) that indicates which worlds are controllable.

        A world is controllable if the last control step was more than 1/control_freq seconds ago.
        Desired controls get stashed in the staged control buffers and are applied in `step`
        as soon as the controller frequency allows for an update. Successive control updates that
        happen before the staged buffers are applied overwrite the desired values.
        """
        controls = self.data.controls
        match self.control:
            case Control.state:
                control_steps, control_freq = controls.state.steps, controls.state.freq
            case Control.attitude:
                control_steps, control_freq = controls.attitude.steps, controls.attitude.freq
            case Control.force_torque:
                control_steps = controls.force_torque.steps
                control_freq = controls.force_torque.freq
            case _:
                raise NotImplementedError(f"Control mode {self.control} not implemented")
        return controllable(self.data.core.steps, self.data.core.freq, control_steps, control_freq)

    @requires_mujoco_sync
    def contacts(self, body: str | None = None) -> Array:
        """Get contact information from the simulation.

        Args:
            body: Optional body name to filter contacts for. If None, returns flags for all bodies.

        Returns:
            An boolean array of shape (n_worlds,) that is True if any contact is present.
        """
        if body is None:
            return self.mjx_data._impl.contact.dist < 0
        body_id = self.mj_model.body(body).id
        geom_start = self.mj_model.body_geomadr[body_id]
        geom_count = self.mj_model.body_geomnum[body_id]
        return contacts(geom_start, geom_count, self.mjx_data)

    @staticmethod
    def _reset(data: SimData, default_data: SimData, mask: Array | None = None) -> SimData:
        raise NotInitializedError("_reset call before building the simulation pipeline.")

    @staticmethod
    def _step(data: SimData, n_steps: int) -> SimData:
        raise NotInitializedError("_step call before building the simulation pipeline.")


def build_control_fns(
    control: Control, physics: Physics
) -> tuple[Callable[[SimData], SimData], ...]:
    """Select the control functions for the given control mode.

    Note:
        This function returns a tuple of functions, not a single function. The returned functions
        are called in succession in the simulation pipeline.
    """
    match control:
        case Control.state:
            control_pipeline = (step_state_controller, step_attitude_controller)
            if physics == Physics.first_principles:
                control_pipeline = control_pipeline + (step_force_torque_controller,)
        case Control.attitude:
            if physics == Physics.first_principles:
                control_pipeline = (step_attitude_controller, step_force_torque_controller)
            elif physics in (Physics.so_rpy, Physics.so_rpy_rotor, Physics.so_rpy_rotor_drag):
                control_pipeline = (commit_attitude_controller,)
            else:
                raise NotImplementedError(f"Control mode {control} not implemented for {physics}")
        case Control.force_torque:
            control_pipeline = (step_force_torque_controller,)
        case _:
            raise NotImplementedError(f"Control mode {control} not implemented")

    return control_pipeline


def select_physics_fn(physics: Physics) -> Callable[[SimData], SimData]:
    """Select the physics function for the given physics mode."""
    match physics:
        case Physics.first_principles:
            return first_principles_physics
        case Physics.so_rpy:
            return so_rpy_physics
        case Physics.so_rpy_rotor:
            return so_rpy_rotor_physics
        case Physics.so_rpy_rotor_drag:
            return so_rpy_rotor_drag_physics
        case _:
            raise NotImplementedError(f"Physics mode {physics} not implemented")


def select_integrate_fn(
    integrator: Integrator, physics_fn: Callable[[SimData], SimData]
) -> Callable[[SimData], SimData]:
    """Select the integration function for the given physics and integrator mode."""
    match integrator:
        case Integrator.euler:
            integrate_fn = euler
        case Integrator.rk4:
            integrate_fn = rk4
        case Integrator.symplectic_euler:
            integrate_fn = symplectic_euler
        case _:
            raise NotImplementedError(f"Integrator {integrator} not implemented")

    return partial(integrate_fn, deriv_fn=physics_fn)


def increment_steps(data: SimData) -> SimData:
    """Increment the simulation steps."""
    return data.replace(core=data.core.replace(steps=data.core.steps + 1))


@jax.jit
def contacts(geom_start: int, geom_count: int, data: Data) -> Array:
    """Filter contacts from MuJoCo data."""
    geom1_valid = data.contact.geom1 >= geom_start
    geom1_valid &= data.contact.geom1 < geom_start + geom_count
    geom2_valid = data.contact.geom2 >= geom_start
    geom2_valid &= data.contact.geom2 < geom_start + geom_count
    return (data.contact.dist < 0) & (geom1_valid | geom2_valid)


@jax.jit
def sync_sim2mjx(data: SimData, mjx_data: Data, mjx_model: Model) -> tuple[SimData, Data]:
    """Synchronize the simulation data with the MuJoCo model."""
    states = data.states
    pos, quat, vel, ang_vel = states.pos, states.quat, states.vel, states.ang_vel
    quat = jnp.roll(quat, 1, axis=-1)  # MuJoCo quat is [w, x, y, z], ours is [x, y, z, w]
    qpos = rearrange(jnp.concat([pos, quat], axis=-1), "w d qpos -> w (d qpos)")
    qvel = rearrange(jnp.concat([vel, ang_vel], axis=-1), "w d qvel -> w (d qvel)")
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    mjx_data = jax.vmap(mjx.kinematics, in_axes=(None, 0))(mjx_model, mjx_data)
    mjx_data = jax.vmap(mjx.collision, in_axes=(None, 0))(mjx_model, mjx_data)
    data = data.replace(core=data.core.replace(mjx_synced=True))
    return data, mjx_data


def step_state_controller(data: SimData) -> SimData:
    """Compute the updated controls for the state controller."""
    states = data.states
    state_ctrl: MellingerStateData = data.controls.state
    assert state_ctrl is not None, "Using state controller without initialized data"
    mask = controllable(data.core.steps, data.core.freq, state_ctrl.steps, state_ctrl.freq)
    state_ctrl = leaf_replace(state_ctrl, mask, cmd=state_ctrl.staged_cmd)
    rpyt, pos_err_i = state2attitude(
        states.pos,
        states.quat,
        states.vel,
        states.ang_vel,
        state_ctrl.cmd,
        ctrl_errors=(state_ctrl.pos_err_i,),
        ctrl_freq=state_ctrl.freq,
        **state_ctrl.params._asdict(),
    )
    state_ctrl = leaf_replace(state_ctrl, mask, steps=data.core.steps, pos_err_i=pos_err_i)
    attitude_ctrl = leaf_replace(data.controls.attitude, mask, staged_cmd=rpyt)
    return data.replace(controls=data.controls.replace(state=state_ctrl, attitude=attitude_ctrl))


def step_attitude_controller(data: SimData) -> SimData:
    """Compute the updated controls for the attitude controller."""
    states = data.states
    attitude_ctrl: MellingerAttitudeData = data.controls.attitude
    assert attitude_ctrl is not None, "Using attitude controller without initialized data"
    mask = controllable(data.core.steps, data.core.freq, attitude_ctrl.steps, attitude_ctrl.freq)
    attitude_ctrl = leaf_replace(attitude_ctrl, mask, cmd=attitude_ctrl.staged_cmd)
    force, torque, r_int_error = attitude2force_torque(
        states.pos,
        states.quat,
        states.vel,
        states.ang_vel,
        attitude_ctrl.cmd,
        ctrl_errors=(attitude_ctrl.r_int_error,),
        ctrl_freq=attitude_ctrl.freq,
        **attitude_ctrl.params._asdict(),
    )
    attitude_ctrl = leaf_replace(
        attitude_ctrl,
        mask,
        r_int_error=r_int_error,
        last_ang_vel=states.ang_vel,
        steps=data.core.steps,
    )
    ft_ctrl = leaf_replace(
        data.controls.force_torque, mask, staged_cmd=jnp.concat([force, torque], axis=-1)
    )
    return data.replace(
        states=states, controls=data.controls.replace(attitude=attitude_ctrl, force_torque=ft_ctrl)
    )


def commit_attitude_controller(data: SimData) -> SimData:
    """Commit the staged attitude command to the controller setpoint."""
    attitude_ctrl: MellingerAttitudeData = data.controls.attitude
    mask = controllable(data.core.steps, data.core.freq, attitude_ctrl.steps, attitude_ctrl.freq)
    attitude_ctrl = leaf_replace(attitude_ctrl, mask, cmd=attitude_ctrl.staged_cmd)
    return data.replace(controls=data.controls.replace(attitude=attitude_ctrl))


def step_force_torque_controller(data: SimData) -> SimData:
    """Compute the updated controls for the thrust controller."""
    ft_ctrl: MellingerForceTorqueData = data.controls.force_torque
    assert ft_ctrl is not None, "Using force torque controller without initialized data"
    mask = controllable(data.core.steps, data.core.freq, ft_ctrl.steps, ft_ctrl.freq)
    ft_ctrl = leaf_replace(ft_ctrl, mask, cmd=ft_ctrl.staged_cmd)
    rotor_vel = force_torque2rotor_vel(
        ft_ctrl.cmd[..., [0]], ft_ctrl.cmd[..., 1:], **ft_ctrl.params._asdict()
    )
    ft_ctrl = leaf_replace(ft_ctrl, mask, steps=data.core.steps)
    return data.replace(controls=data.controls.replace(rotor_vel=rotor_vel, force_torque=ft_ctrl))


def clip_floor_pos(data: SimData) -> SimData:
    """Clip the position of the drone to the floor."""
    clip = data.states.pos[..., 2] < -0.001
    clip_pos = data.states.pos.at[..., 2].set(jnp.where(clip, -0.001, data.states.pos[..., 2]))
    clip_vel = data.states.vel.at[..., :3].set(
        jnp.where(clip[..., None], 0, data.states.vel[..., :3])
    )
    return data.replace(states=data.states.replace(pos=clip_pos, vel=clip_vel))


@partial(jax.jit, static_argnames="device")
def seed_sim(data: SimData, seed: int, device: Device) -> SimData:
    """JIT-compiled seeding function."""
    rng_key = jax.device_put(jax.random.key(seed), device)
    return data.replace(core=data.core.replace(rng_key=rng_key))


def use_box_collision(sim: Sim, enable: bool = True):
    """Changes the collision geometry to use boxes or spheres (default).

    Args:
        sim: The simulation instance.
        enable: If True, use box collision geometry. If False, use sphere collision geometry.

    Warning:
        Using box collision geometry is more computationally expensive than sphere collision
        geometry, especially for larger swarms. It is recommended to only enable box collision
        geometry for small swarms or when high accuracy is required.
    """
    for geom in sim.spec.geoms:
        if geom.name.startswith("col_sphere"):
            geom.contype = 1 * (not enable)
            geom.conaffinity = 1 * (not enable)
            geom.rgba[3] = 1 * (not enable)
        if geom.name.startswith("col_box"):
            geom.contype = 1 * enable
            geom.conaffinity = 1 * enable
            geom.rgba[3] = 1 * enable

    sim.build_mjx()
