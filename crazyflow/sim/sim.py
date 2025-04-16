from functools import partial
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
from einops import rearrange
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from jax import Array, Device
from jax.scipy.spatial.transform import Rotation as R
from mujoco.mjx import Data, Model

from crazyflow.constants import J_INV, MASS, SIGN_MIX_MATRIX, J
from crazyflow.control.control import Control, attitude2rpm, pwm2rpm, state2attitude, thrust2pwm
from crazyflow.exception import ConfigError, NotInitializedError
from crazyflow.sim.integration import Integrator, euler, rk4
from crazyflow.sim.physics import (
    Physics,
    collective_force2acceleration,
    collective_torque2ang_vel_deriv,
    rpms2collective_wrench,
    rpms2motor_forces,
    rpms2motor_torques,
    surrogate_identified_collective_wrench,
)
from crazyflow.sim.structs import SimControls, SimCore, SimData, SimParams, SimState, SimStateDeriv
from crazyflow.utils import grid_2d, leaf_replace, patch_viewer, pytree_replace, to_device


class Sim:
    default_path = Path(__file__).parents[1] / "models/cf2/scene.xml"
    drone_path = Path(__file__).parents[1] / "models/cf2/cf2.xml"

    def __init__(
        self,
        n_worlds: int = 1,
        n_drones: int = 1,
        physics: Physics = Physics.default,
        control: Control = Control.default,
        integrator: Integrator = Integrator.default,
        freq: int = 500,
        state_freq: int = 100,
        attitude_freq: int = 500,
        thrust_freq: int = 500,
        device: str = "cpu",
        xml_path: Path | None = None,
        rng_key: int = 0,
    ):
        assert Physics(physics) in Physics, f"Physics mode {physics} not implemented"
        assert Control(control) in Control, f"Control mode {control} not implemented"
        if physics == Physics.sys_id and control == Control.thrust:
            raise ConfigError("Thrust control is not supported with sys_id physics")
        if freq > 10_000 and not jax.config.jax_enable_x64:
            raise ConfigError("Double precision mode is required for high frequency simulations")
        self.physics = physics
        self.control = control
        self.integrator = integrator
        self.device = jax.devices(device)[0]
        self.n_worlds = n_worlds
        self.n_drones = n_drones
        self.freq = freq
        self.max_visual_geom = 1000

        # Initialize MuJoCo world and data
        self._xml_path = xml_path or self.default_path
        self.spec = self.build_mjx_spec()
        self.mj_model, self.mj_data, self.mjx_model, mjx_data = self.build_mjx_model(self.spec)
        self.viewer: MujocoRenderer | None = None

        self.data = self.init_data(state_freq, attitude_freq, thrust_freq, rng_key, mjx_data)
        self.default_data = self.init_default_data()

        # Default functions for the simulation pipeline
        self.disturbance_fn: Callable[[SimData], SimData] = identity
        self.reset_hook: Callable[[SimData, Array[bool] | None], SimData] = identity

        # Build the simulation pipeline and overwrite the default _step implementation with it
        self.build_reset_fn()
        self.build_step_fn()

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
        mjx_data = jax.vmap(lambda _: mjx_data)(range(self.n_worlds))
        return mj_model, mj_data, mjx_model, mjx_data

    def build_step_fn(self):
        """Setup the chain of functions that are called in Sim.step().

        We know all the functions that are called in succession since the simulation is configured
        at initialization time. Instead of branching through options at runtime, we construct a step
        function at initialization that selects the correct functions based on the settings.

        Warning:
            If any settings change, the pipeline of functions needs to be reconstructed.
        """
        # The ``generate_xxx_fn`` methods return functions, not the results of calling those
        # functions. They act as factories that produce building blocks for the construction of our
        # simulation pipeline.
        ctrl_fn = generate_control_fn(self.control)
        wrench_fn = generate_wrench_fn(self.physics)
        disturbance_fn = self.disturbance_fn
        physics_fn = generate_physics_fn(self.physics, self.integrator)
        sync_fn = generate_sync_fn(self.physics)

        # None is required by jax.lax.scan to unpack the tuple returned by single_step.
        def single_step(data: SimData, _: None) -> tuple[SimData, None]:
            data = ctrl_fn(data)
            data = wrench_fn(data)
            data = disturbance_fn(data)
            data = physics_fn(data)
            data = data.replace(core=data.core.replace(steps=data.core.steps + 1))
            # MuJoCo needs to sync after every physics step, so that the next step control, wrench
            # and disturbance functions see the correct state.
            data = sync_fn(data)
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
            # Performance optimization: When step is called, jax checks if it can reuse a previously
            # compiled version of the function. This check flattens the sim.data PyTree and compares
            # the metadata of each leaf with the cached metadata. The more leaves contained in
            # sim.data, the more time is spent on the cache lookup even if the function has already
            # been compiled. Since mjx_model contains many PyTree nodes and it is not used by
            # physics modes other than mujoco with domain randomization, we set it to None and
            # capture the current sim.mjx_model in the step function's closure. Changes to the
            # params are synced to mjx_model at the start of the step function.
            if optimize_mjx_model := (data.mjx_model is None):
                data = data.replace(mjx_model=self.mjx_model)
            data = self.sync_sim2mjx(data)
            data, _ = jax.lax.scan(single_step, data, length=n_steps, unroll=10)
            if optimize_mjx_model:
                data = data.replace(mjx_model=None)
            return data

        self._step = step

    def init_data(
        self, state_freq: int, attitude_freq: int, thrust_freq: int, rng_key: Array, mjx_data: Data
    ) -> tuple[SimData, SimData]:
        """Initialize the simulation data."""
        drone_ids = [self.mj_model.body(f"drone:{i}").id for i in range(self.n_drones)]
        N, D = self.n_worlds, self.n_drones
        data = SimData(
            states=SimState.create(N, D, self.device),
            states_deriv=SimStateDeriv.create(N, D, self.device),
            controls=SimControls.create(N, D, state_freq, attitude_freq, thrust_freq, self.device),
            params=SimParams.create(N, D, MASS, J, J_INV, self.device),
            core=SimCore.create(self.freq, N, D, drone_ids, rng_key, self.device),
            mjx_data=mjx_data,
            mjx_model=None,
        )
        if D > 1:  # If multiple drones, arrange them in a grid
            grid = grid_2d(D)
            states = data.states.replace(pos=data.states.pos.at[..., :2].set(grid))
            data = data.replace(states=states)
        data = self.sync_sim2mjx(data, self.mjx_model)
        return data

    def init_default_data(self) -> SimData:
        """Initialize the default data for the simulation.

        Todo:
            Only save the data of one world.
        """
        return self.data.replace()

    def build_reset_fn(self):
        """Build the reset function for the current simulation configuration."""
        reset_hook = self.reset_hook

        @jax.jit
        def reset(data: SimData, default_data: SimData, mask: Array | None = None) -> SimData:
            data = pytree_replace(data, default_data, mask)  # Does not overwrite rng_key
            data = reset_hook(data, mask)
            data = self.sync_sim2mjx(data, self.mjx_model)
            return data

        self._reset = reset

    def build(
        self,
        *,
        mjx: bool = True,
        data: bool = True,
        default_data: bool = True,
        reset: bool = True,
        step: bool = True,
    ):
        """Build the simulation pipeline.

        This method is used to (re)build the simulation pipeline after changing the MuJoCo
        model specification or any of the default functions that are used in the compiled step
        function.

        Warning:
            Depending on what you build, you reset the simulation state. For example, rebuilding the
            simulation data will reset the drone states.

        Args:
            mjx: Flag to (re)build the MuJoCo model and data structures.
            data: Flag to (re)build the simulation data.
            default_data: Flag to (re)build the default data. Useful for setting the reset state to
                the current state.
            reset: Flag to (re)build the reset function.
            step: Flag to (re)build the simulation step function.
        """
        # TODO: Write tests for all options
        if mjx:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            self.mj_model, self.mj_data, self.mjx_model, mjx_data = self.build_mjx_model(self.spec)
            self.data = self.data.replace(mjx_data=mjx_data)
            self.data = self.sync_sim2mjx(self.data, self.mjx_model)
            self.default_data = self.default_data.replace(mjx_data=mjx_data)
        if data:
            self.data = self.init_data(
                self.data.controls.state_freq,
                self.data.controls.attitude_freq,
                self.data.controls.thrust_freq,
                self.data.core.rng_key,
                self.data.mjx_data if not mjx else mjx_data,
            )
        if default_data:
            self.default_data = self.init_default_data()
        if reset:
            self.build_reset_fn()
        if step:
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
        self.data = self.data.replace(controls=self.data.controls.replace(staged_attitude=controls))

    def state_control(self, controls: Array):
        """Set the desired state for all drones in all worlds."""
        assert controls.shape == (self.n_worlds, self.n_drones, 13), "controls shape mismatch"
        assert self.control == Control.state, "State control is not enabled by the sim config"
        controls = to_device(controls, self.device)
        self.data = self.data.replace(controls=self.data.controls.replace(state=controls))

    def thrust_control(self, cmd: Array):
        """Set the desired thrust for all drones in all worlds."""
        assert cmd.shape == (self.n_worlds, self.n_drones, 4), "Command shape mismatch"
        assert self.control == Control.thrust, "Thrust control is not enabled by the sim config"
        controls = to_device(cmd, self.device)
        self.data = self.data.replace(controls=self.data.controls.replace(thrust=controls))

    def render(self):
        if self.viewer is None:
            patch_viewer()
            self.viewer = MujocoRenderer(self.mj_model, self.mj_data, max_geom=self.max_visual_geom)
        self.mj_data.qpos[:] = self.data.mjx_data.qpos[0, :]
        self.mj_data.mocap_pos[:] = self.data.mjx_data.mocap_pos[0, :]
        self.mj_data.mocap_quat[:] = self.data.mjx_data.mocap_quat[0, :]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.viewer.render("human")

    def seed(self, seed: int):
        """Set the JAX rng key for the simulation.

        Args:
            seed: The seed for the JAX rng.
        """
        self.data = seed_sim(self.data, seed, self.device)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    @property
    def time(self) -> Array:
        return self.data.core.steps / self.data.core.freq

    @property
    def control_freq(self) -> int:
        if self.control == Control.state:
            return self.data.controls.state_freq
        if self.control == Control.attitude:
            return self.data.controls.attitude_freq
        if self.control == Control.thrust:
            return self.data.controls.thrust_freq
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
                control_steps, control_freq = controls.state_steps, controls.state_freq
            case Control.attitude:
                control_steps, control_freq = (controls.attitude_steps, controls.attitude_freq)
            case Control.thrust:
                control_steps, control_freq = (controls.thrust_steps, controls.thrust_freq)
            case _:
                raise NotImplementedError(f"Control mode {self.control} not implemented")
        return controllable(self.data.core.steps, self.data.core.freq, control_steps, control_freq)

    def contacts(self, body: str | None = None) -> Array:
        """Get contact information from the simulation.

        Args:
            body: Optional body name to filter contacts for. If None, returns flags for all bodies.

        Returns:
            An boolean array of shape (n_worlds,) that is True if any contact is present.
        """
        if body is None:
            return self.data.mjx_data.contact.dist < 0
        body_id = self.mj_model.body(body).id
        geom_start = self.mj_model.body_geomadr[body_id]
        geom_count = self.mj_model.body_geomnum[body_id]
        return contacts(geom_start, geom_count, self.data.mjx_data)

    @staticmethod
    @jax.jit
    def sync_sim2mjx(data: SimData, mjx_model: Model | None = None) -> SimData:
        states = data.states
        pos, quat, vel, ang_vel = states.pos, states.quat, states.vel, states.ang_vel
        quat = quat[..., [3, 0, 1, 2]]  # MuJoCo quat is [w, x, y, z], ours is [x, y, z, w]
        qpos = rearrange(jnp.concat([pos, quat], axis=-1), "w d qpos -> w (d qpos)")
        qvel = rearrange(jnp.concat([vel, ang_vel], axis=-1), "w d qvel -> w (d qvel)")
        mjx_data = data.mjx_data
        mjx_model = data.mjx_model if mjx_model is None else mjx_model
        assert mjx_model is not None, "MuJoCo model is not initialized"
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        mjx_data = jax.vmap(mjx.kinematics, in_axes=(None, 0))(mjx_model, mjx_data)
        mjx_data = jax.vmap(mjx.collision, in_axes=(None, 0))(mjx_model, mjx_data)
        data = data.replace(mjx_data=mjx_data)
        if data.mjx_model is None:  # Only modify model if it is part of data
            return data
        # Sync model parameters such as mass and inertia for domain randomization
        # This is currently not supported. See https://github.com/google-deepmind/mujoco/issues/1607
        # TODO: Implement once mjx supports batching single model fields.
        return data

    @staticmethod
    @jax.jit
    def sync_mjx2sim(data: SimData) -> SimData:
        mjx_data = data.mjx_data
        qpos = mjx_data.qpos.reshape(data.core.n_worlds, data.core.n_drones, 7)
        qvel = mjx_data.qvel.reshape(data.core.n_worlds, data.core.n_drones, 6)
        pos, quat = jnp.split(qpos, [3], axis=-1)
        vel, ang_vel = jnp.split(qvel, [3], axis=-1)
        quat = quat[..., [1, 2, 3, 0]]  # MuJoCo quat is [w, x, y, z], ours is [x, y, z, w]
        states = data.states.replace(pos=pos, quat=quat, vel=vel, ang_vel=ang_vel)
        return data.replace(states=states)

    @staticmethod
    def _reset(data: SimData, default_data: SimData, mask: Array | None = None) -> SimData:
        raise NotInitializedError("_reset call before building the simulation pipeline.")

    @staticmethod
    def _step(data: SimData, n_steps: int) -> SimData:
        raise NotInitializedError("_step call before building the simulation pipeline.")


def generate_control_fn(control: Control) -> Callable[[SimData], SimData]:
    """Generate the control function for the given control mode."""
    match control:
        case Control.state:
            return lambda data: step_attitude_controller(step_state_controller(data))
        case Control.attitude:
            return step_attitude_controller
        case Control.thrust:
            return step_thrust_controller
        case _:
            raise NotImplementedError(f"Control mode {control} not implemented")


def generate_wrench_fn(physics: Physics) -> Callable[[SimData], SimData]:
    """Generate the wrench function for the given physics mode."""
    match physics:
        case Physics.analytical:
            return analytical_wrench
        case Physics.sys_id:
            return identified_wrench
        case Physics.mujoco:
            return mujoco_wrench
        case _:
            raise NotImplementedError(f"Physics mode {physics} not implemented")


def generate_derivative_fn(physics: Physics) -> Callable[[SimData], SimData]:
    """Generate the derivative function for the given physics mode."""
    match physics:
        case Physics.analytical:
            return analytical_derivative
        case Physics.sys_id:
            return identified_derivative
        case _:
            raise NotImplementedError(f"Physics mode {physics} not implemented")


def generate_integrator_fn(
    integrator: Integrator,
) -> Callable[[SimData, Callable[[SimData], SimData]], SimData]:
    """Generate the integrator function for the given integrator mode."""
    match integrator:
        case Integrator.euler:
            return euler
        case Integrator.rk4:
            return rk4
        case _:
            raise NotImplementedError(f"Integrator {integrator} not implemented")


def generate_physics_fn(physics: Physics, integrator: Integrator) -> Callable[[SimData], SimData]:
    """Generate the physics function for the given physics mode."""
    match physics:
        case Physics.sys_id | Physics.analytical:
            integrator_fn = generate_integrator_fn(integrator)
            derivative_fn = generate_derivative_fn(physics)
            return lambda data: integrator_fn(data, derivative_fn)
        case Physics.mujoco:
            return mjx_physics_fn
        case _:
            raise NotImplementedError(f"Physics mode {physics} not implemented")


def generate_sync_fn(physics: Physics) -> Callable[[SimData], SimData]:
    """Generate the sync function for the given physics mode."""
    match physics:
        case Physics.sys_id | Physics.analytical:
            return Sim.sync_sim2mjx
        case Physics.mujoco:
            return Sim.sync_mjx2sim
        case _:
            raise NotImplementedError(f"Physics mode {physics} not implemented")


@jax.jit
def controllable(step: Array, freq: int, control_steps: Array, control_freq: int) -> Array:
    """Check which worlds can currently update their controllers.

    Args:
        step: The current step of the simulation.
        freq: The frequency of the simulation.
        control_steps: The steps at which the controllers were last updated.
        control_freq: The frequency of the controllers.

    Returns:
        A boolean mask of shape (n_worlds,) that is True at the worlds where the controllers can be
        updated.
    """
    return ((step - control_steps) >= (freq / control_freq)) | (control_steps == -1)


@jax.jit
def contacts(geom_start: int, geom_count: int, data: Data) -> Array:
    """Filter contacts from MuJoCo data."""
    geom1_valid = data.contact.geom1 >= geom_start
    geom1_valid &= data.contact.geom1 < geom_start + geom_count
    geom2_valid = data.contact.geom2 >= geom_start
    geom2_valid &= data.contact.geom2 < geom_start + geom_count
    return (data.contact.dist < 0) & (geom1_valid | geom2_valid)


def step_state_controller(data: SimData) -> SimData:
    """Compute the updated controls for the state controller."""
    states, controls = data.states, data.controls
    mask = controllable(data.core.steps, data.core.freq, controls.state_steps, controls.state_freq)
    des_pos, des_vel = controls.state[..., :3], controls.state[..., 3:6]
    des_yaw = controls.state[..., [9]]  # Keep (N, M, 1) shape for broadcasting
    dt = 1 / data.controls.state_freq
    attitude, pos_err_i = state2attitude(
        states.pos, states.vel, states.quat, des_pos, des_vel, des_yaw, controls.pos_err_i, dt
    )
    controls = leaf_replace(
        controls, mask, state_steps=data.core.steps, staged_attitude=attitude, pos_err_i=pos_err_i
    )
    return data.replace(controls=controls)


def step_attitude_controller(data: SimData) -> SimData:
    """Compute the updated controls for the attitude controller."""
    controls = data.controls
    steps, freq = data.core.steps, data.core.freq
    mask = controllable(steps, freq, controls.attitude_steps, controls.attitude_freq)
    # Commit the staged attitude controls
    staged_attitude = controls.staged_attitude
    controls = leaf_replace(controls, mask, attitude_steps=steps, attitude=staged_attitude)
    # Compute the new rpm values from the committed attitude controls
    quat, attitude = data.states.quat, controls.attitude
    dt = 1 / controls.attitude_freq
    rpms, rpy_err_i = attitude2rpm(attitude, quat, controls.last_rpy, controls.rpy_err_i, dt)
    rpy = R.from_quat(quat).as_euler("xyz")
    controls = leaf_replace(controls, mask, rpms=rpms, rpy_err_i=rpy_err_i, last_rpy=rpy)
    return data.replace(controls=controls)


def step_thrust_controller(data: SimData) -> SimData:
    """Compute the updated controls for the thrust controller."""
    controls = data.controls
    steps = data.core.steps
    mask = controllable(steps, data.core.freq, controls.thrust_steps, controls.thrust_freq)
    rpms = pwm2rpm(thrust2pwm(controls.thrust))
    controls = leaf_replace(controls, mask, thrust_steps=steps, rpms=rpms)
    return data.replace(controls=controls)


def analytical_wrench(data: SimData) -> SimData:
    """Compute the wrench from the analytical dynamics model."""
    states, controls, params = data.states, data.controls, data.params
    force, torque = rpms2collective_wrench(controls.rpms, states.quat, states.ang_vel, params.J)
    return data.replace(states=data.states.replace(force=force, torque=torque))


def analytical_derivative(data: SimData) -> SimData:
    """Compute the derivative of the states."""
    quat, mass, J_inv = data.states.quat, data.params.mass, data.params.J_INV
    acc = collective_force2acceleration(data.states.force, mass)
    ang_vel_deriv = collective_torque2ang_vel_deriv(data.states.torque, quat, J_inv)
    vel, ang_vel = (data.states.vel, data.states.ang_vel)  # Already given in the states
    deriv = data.states_deriv
    deriv = deriv.replace(dpos=vel, drot=ang_vel, dvel=acc, dang_vel=ang_vel_deriv)
    return data.replace(states_deriv=deriv)


def identified_wrench(data: SimData) -> SimData:
    """Compute the wrench from the identified dynamics model."""
    states, controls = data.states, data.controls
    mass, J = data.params.mass, data.params.J
    force, torque = surrogate_identified_collective_wrench(
        controls.attitude, states.quat, states.ang_vel, mass, J, 1 / data.core.freq
    )
    return data.replace(states=data.states.replace(force=force, torque=torque))


identified_derivative = analytical_derivative  # We can use the same derivative function for both


def mujoco_wrench(data: SimData) -> SimData:
    """Compute the wrench from the MuJoCo dynamics model."""
    forces = rpms2motor_forces(data.controls.rpms)
    torques = SIGN_MIX_MATRIX[..., 2] * rpms2motor_torques(data.controls.rpms)
    # Zero out external forces and torques to avoid summation over multiple steps
    states = data.states
    force, torque = jnp.zeros_like(states.force), jnp.zeros_like(states.torque)
    states = states.replace(motor_forces=forces, motor_torques=torques, force=force, torque=torque)
    return data.replace(states=states)


batched_mjx_step = jax.vmap(mjx.step, in_axes=(None, 0))


def mjx_physics_fn(data: SimData) -> SimData:
    """Step the MuJoCo simulation."""
    force_torques = jnp.concatenate([data.states.motor_forces, data.states.motor_torques], axis=-1)
    force_torques = rearrange(force_torques, "w d ft -> w (d ft)")
    mjx_data = data.mjx_data.replace(ctrl=force_torques)
    # Add disturbances from data.states.force/torque with mjx_data.xfrc_applied
    xfrc = jnp.concatenate([data.states.force, data.states.torque], axis=-1)
    xfrc_applied = data.mjx_data.xfrc_applied.at[:, data.core.drone_ids, :].set(xfrc)
    mjx_data = mjx_data.replace(xfrc_applied=xfrc_applied)
    mjx_data = batched_mjx_step(data.mjx_model, mjx_data)
    return data.replace(mjx_data=mjx_data)


def identity(data: SimData, *args: Any, **kwargs: Any) -> SimData:
    """Identity function for the simulation pipeline.

    Used as default function for optional pipeline steps.
    """
    return data


@partial(jax.jit, static_argnames="device")
def seed_sim(data: SimData, seed: int, device: Device) -> SimData:
    """JIT-compiled seeding function."""
    rng_key = jax.device_put(jax.random.key(seed), device)
    return data.replace(core=data.core.replace(rng_key=rng_key))
