from functools import partial
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
from einops import rearrange
from flax.struct import dataclass
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from jax import Array
from jax.scipy.spatial.transform import Rotation as R
from mujoco.mjx import Data, Model

from crazyflow.constants import J_INV, MASS, SIGN_MIX_MATRIX, J
from crazyflow.control.control import Control, attitude2rpm, pwm2rpm, state2attitude, thrust2pwm
from crazyflow.exception import ConfigError, NotInitializedError
from crazyflow.sim.integration import Integrator, euler, rk4
from crazyflow.sim.physics import (
    Physics,
    ang_vel2rpy_rates,
    collective_force2acceleration,
    collective_torque2rpy_rates_deriv,
    rpms2collective_wrench,
    rpms2motor_forces,
    rpms2motor_torques,
    virtual_identified_collective_wrench,
)
from crazyflow.sim.structs import (
    SimData,
    default_controls,
    default_core,
    default_params,
    default_state,
    default_state_deriv,
)
from crazyflow.utils import grid_2d, leaf_replace, pytree_replace


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
        if physics != Physics.analytical and control == Control.thrust:  # TODO: Implement
            raise ConfigError("Thrust control is not supported with sys_id physics")
        if freq > 10_000 and not jax.config.jax_enable_x64:
            raise ConfigError("Double precision mode is required for high frequency simulations")
        self.physics = physics
        self.control = control
        self.integrator = integrator
        self.device = jax.devices(device)[0]
        self.n_worlds = n_worlds
        self.n_drones = n_drones

        # Initialize MuJoCo world and data
        self._xml_path = xml_path or self.default_path
        self.spec, self._mj_model, self._mj_data, self.mjx_model, self.mjx_data = self.setup_mj()
        self.viewer: MujocoRenderer | None = None

        # Allocate internal states and controls
        states = default_state(n_worlds, n_drones, self.device)
        states_deriv = default_state_deriv(n_worlds, n_drones, self.device)
        controls = default_controls(
            n_worlds, n_drones, state_freq, attitude_freq, thrust_freq, self.device
        )
        params = default_params(n_worlds, n_drones, MASS, J, J_INV, self.device)
        core = default_core(freq, n_worlds, n_drones, rng_key, self.device)
        self.data = SimData(
            states=states, states_deriv=states_deriv, controls=controls, params=params, core=core
        )
        if self.n_drones > 1:  # If multiple drones, arrange them in a grid
            grid = grid_2d(self.n_drones)
            states = self.data.states.replace(pos=self.data.states.pos.at[..., :2].set(grid))
            self.data: SimData = self.data.replace(states=states)
        self.default_data = self.data.replace()

        # Default functions for the simulation pipeline
        self.disturbance_fn: Callable[[SimData], SimData] | None = None

        # Build the simulation pipeline and overwrite the default _step implementation with it
        self.build()

    def setup_mj(self) -> tuple[Any, Any, Any, Model, Data]:
        assert self._xml_path.exists(), f"Model file {self._xml_path} does not exist"
        spec = mujoco.MjSpec.from_file(str(self._xml_path))
        drone_spec = mujoco.MjSpec.from_file(str(self.drone_path))
        frame = spec.worldbody.add_frame()
        # Add drones and their actuators
        for i in range(self.n_drones):
            drone = frame.attach_body(drone_spec.find_body("drone"), "", f":{i}")
            drone.add_freejoint()
        # Compile and create data structures
        mj_model = spec.compile()
        mj_data = mujoco.MjData(mj_model)
        mjx_model = mjx.put_model(mj_model, device=self.device)
        mjx_data = mjx.put_data(mj_model, mj_data, device=self.device)
        mjx_data = jax.vmap(lambda _: mjx_data)(jnp.arange(self.n_worlds))
        # Avoid recompilation on the second call due to time being a weak type. See e.g.
        # https://github.com/jax-ml/jax/issues/4274#issuecomment-692406759
        # Tracking issue: https://github.com/google-deepmind/mujoco/issues/2306
        mjx_data = mjx_data.replace(time=jnp.float32(mjx_data.time))
        return spec, mj_model, mj_data, mjx_model, mjx_data

    def build(self):
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
        disturbance_fn = identity if self.disturbance_fn is None else self.disturbance_fn
        physics_fn = generate_physics_fn(self.physics, self.integrator)
        sync_fn = generate_sync_fn(self.physics)

        # None is required by jax.lax.scan to unpack the tuple returned by single_step.
        def single_step(data: SimData, _: None) -> tuple[SimData, None]:
            data = ctrl_fn(data)
            data = wrench_fn(data)
            data = disturbance_fn(data)
            data = physics_fn(data)
            data = data.replace(core=data.core.replace(steps=data.core.steps + 1))
            return data, None

        @dataclass
        class ScanData:
            """jax.lax.scan requires that the carry-over data is a single Array or PyTree."""

            data: SimData
            mjx_data: Data
            mjx_model: Model

        def single_mujoco_step(scan_data: ScanData, _: None) -> tuple[ScanData, None]:
            data, mjx_data, mjx_model = scan_data.data, scan_data.mjx_data, scan_data.mjx_model
            data = ctrl_fn(data)
            data = wrench_fn(data)
            data = disturbance_fn(data)
            mjx_data = mjx_physics_fn(data, mjx_data, mjx_model)
            data = data.replace(core=data.core.replace(steps=data.core.steps + 1))
            data = sync_fn(data, mjx_data)
            return ScanData(data=data, mjx_data=mjx_data, mjx_model=mjx_model), None

        # ``scan`` can be lowered to a single WhileOp, reducing compilation times while still fusing
        # the loops and giving XLA maximum freedom to reorder operations and jointly optimize the
        # pipeline. This is especially relevant for the common use case of running multiple sim
        # steps in an outer loop, e.g. in gym environments.
        # Having n_steps as a static argument is fine, since patterns with n_steps > 1 will almost
        # always use the same n_steps value for successive calls.
        @partial(jax.jit, static_argnames="n_steps")
        def _step(data: SimData, mjx_data: Data, n_steps: int = 1) -> SimData:
            data, _ = jax.lax.scan(single_step, data, length=n_steps)
            mjx_data = sync_fn(data, mjx_data, self.mjx_model)
            return data, mjx_data

        # We wrap the _step function to remove the unused Model argument. This significantly
        # improves performance, because the Model struct does not have to be flattened and checked
        # for consistency with the jitted _step function.
        def step(data: SimData, mjx_data: Data, _: Model, n_steps: int = 1) -> SimData:
            return _step(data, mjx_data, n_steps)

        @partial(jax.jit, static_argnames="n_steps")
        def mujoco_step(
            data: SimData, mjx_data: Data, mjx_model: Model, n_steps: int = 1
        ) -> SimData:
            scan_data = ScanData(data=data, mjx_data=mjx_data, mjx_model=mjx_model)
            scan_data, _ = jax.lax.scan(single_mujoco_step, scan_data, length=n_steps)
            return scan_data.data, scan_data.mjx_data

        self._step = mujoco_step if self.physics == Physics.mujoco else step

    def reset(self, mask: Array | None = None):
        """Reset the simulation to the initial state.

        Args:
            mask: Boolean array of shape (n_worlds, ) that indicates which worlds to reset. If None,
                all worlds are reset.
        """
        assert mask is None or mask.shape == (self.n_worlds,), f"Mask shape mismatch {mask.shape}"
        self.data = self._reset(self.data, self.default_data, mask)
        self.mjx_data = self.sync_sim2mjx(self.data, self.mjx_data, self.mjx_model)

    def step(self, n_steps: int = 1):
        """Simulate all drones in all worlds for n time steps."""
        assert n_steps > 0, "Number of steps must be positive"
        self.data, self.mjx_data = self._step(self.data, self.mjx_data, self.mjx_model, n_steps)

    def attitude_control(self, controls: Array):
        """Set the desired attitude for all drones in all worlds."""
        assert controls.shape == (self.n_worlds, self.n_drones, 4), "controls shape mismatch"
        assert self.control == Control.attitude, "Attitude control is not enabled by the sim config"
        self.data = attitude_control(controls, self.data, self.device)

    def state_control(self, controls: Array):
        """Set the desired state for all drones in all worlds."""
        assert controls.shape == (self.n_worlds, self.n_drones, 13), "controls shape mismatch"
        assert self.control == Control.state, "State control is not enabled by the sim config"
        self.data = state_control(controls, self.data, self.device)

    def thrust_control(self, cmd: Array):
        """Set the desired thrust for all drones in all worlds."""
        assert cmd.shape == (self.n_worlds, self.n_drones, 4), "Command shape mismatch"
        assert self.control == Control.thrust, "Thrust control is not enabled by the sim config"
        self.data = thrust_control(cmd, self.data, self.device)

    def render(self):
        if self.viewer is None:
            self.viewer = MujocoRenderer(self._mj_model, self._mj_data)
        self._mj_data.qpos[:] = self.mjx_data.qpos[0, :]
        mujoco.mj_forward(self._mj_model, self._mj_data)
        self.viewer.render("human")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    @property
    def time(self) -> Array:
        return self.data.core.steps / self.data.core.freq

    @property
    def freq(self) -> int:
        return self.data.core.freq

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
                control_steps, control_freq = controls.attitude_steps, controls.attitude_freq
            case Control.thrust:
                control_steps, control_freq = controls.thrust_steps, controls.thrust_freq
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
            return self.mjx_data.contact.dist < 0
        body_id = self._mj_model.body(body).id
        geom_start = self._mj_model.body_geomadr[body_id]
        geom_count = self._mj_model.body_geomnum[body_id]
        return contacts(geom_start, geom_count, self.mjx_data)

    @staticmethod
    @jax.jit
    def sync_sim2mjx(data: SimData, mjx_data: Data, mjx_model: Model) -> Data:
        states = data.states
        pos, quat, vel, rpy_rates = states.pos, states.quat, states.vel, states.rpy_rates
        quat = quat[..., [3, 0, 1, 2]]  # MuJoCo quat is [w, x, y, z], ours is [x, y, z, w]
        qpos = rearrange(jnp.concat([pos, quat], axis=-1), "w d qpos -> w (d qpos)")
        # TODO: rpy_rates should be ang_vel instead. Fix with conversion from rpy_rates to ang_vel
        qvel = rearrange(jnp.concat([vel, rpy_rates], axis=-1), "w d qvel -> w (d qvel)")
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        mjx_data = mjx_kinematics(mjx_model, mjx_data)
        mjx_data = mjx_collision(mjx_model, mjx_data)
        return mjx_data

    @staticmethod
    @jax.jit
    def sync_mjx2sim(data: SimData, mjx_data: Data) -> SimData:
        qpos = mjx_data.qpos.reshape(data.core.n_worlds, data.core.n_drones, 7)
        qvel = mjx_data.qvel.reshape(data.core.n_worlds, data.core.n_drones, 6)
        pos, quat = jnp.split(qpos, [3], axis=-1)
        quat = quat[..., [1, 2, 3, 0]]  # MuJoCo quat is [w, x, y, z], ours is [x, y, z, w]
        vel, local_ang_vel = jnp.split(qvel, [3], axis=-1)
        rpy_rates = R.from_quat(quat).apply(ang_vel2rpy_rates(local_ang_vel, quat))
        states = data.states.replace(pos=pos, quat=quat, vel=vel, rpy_rates=rpy_rates)
        return data.replace(states=states)

    @staticmethod
    @jax.jit
    def _reset(data: SimData, default_data: SimData, mask: Array | None = None) -> SimData:
        rng_key = data.core.rng_key
        data = pytree_replace(data, default_data, mask)
        return data.replace(core=data.core.replace(rng_key=rng_key))  # Don't reset the rng_key

    def _step(self, data: SimData, mjx_data: Data, mjx_model: Model, n_steps: int) -> SimData:
        raise NotInitializedError("_step call before compiling the simulation pipeline.")


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


@partial(jax.jit, static_argnames="device")
def state_control(controls: Array, data: SimData, device: str) -> SimData:
    """Set the desired state for all drones in all worlds."""
    controls = jnp.array(controls, device=device)
    return data.replace(controls=data.controls.replace(state=controls))


@partial(jax.jit, static_argnames="device")
def attitude_control(controls: Array, data: SimData, device: str) -> SimData:
    """Stage the desired attitude for all drones in all worlds.

    We need to stage the attitude controls because the sys_id physics mode operates directly on
    the attitude controls. If we were to directly update the controls, this would effectively
    bypass the control frequency and run the attitude controller at the physics update rate. By
    staging the controls, we ensure that the physics module sees the old controls until the
    controller updates at its correct frequency.
    """
    controls = jnp.array(controls, device=device)
    return data.replace(controls=data.controls.replace(staged_attitude=controls))


@partial(jax.jit, static_argnames="device")
def thrust_control(controls: Array, data: SimData, device: str) -> SimData:
    """Set the desired thrust for all drones in all worlds."""
    controls = jnp.array(controls, device=device)
    return data.replace(controls=data.controls.replace(thrust=controls))


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
    return data.contact.dist < 0 & (geom1_valid | geom2_valid)


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
    force, torque = rpms2collective_wrench(controls.rpms, states.quat, states.rpy_rates, params.J)
    return data.replace(states=data.states.replace(force=force, torque=torque))


def analytical_derivative(data: SimData) -> SimData:
    """Compute the derivative of the states."""
    quat, mass, J_inv = data.states.quat, data.params.mass, data.params.J_INV
    acc = collective_force2acceleration(data.states.force, mass)
    rpy_rates_deriv = collective_torque2rpy_rates_deriv(data.states.torque, quat, J_inv)
    vel, rpy_rates = data.states.vel, data.states.rpy_rates  # Already given in the states
    deriv = data.states_deriv
    deriv = deriv.replace(dpos=vel, drot=rpy_rates, dvel=acc, drpy_rates=rpy_rates_deriv)
    return data.replace(states_deriv=deriv)


def identified_wrench(data: SimData) -> SimData:
    """Compute the wrench from the identified dynamics model."""
    states, controls = data.states, data.controls
    mass, J = data.params.mass, data.params.J
    force, torque = virtual_identified_collective_wrench(
        controls.attitude, states.quat, states.rpy_rates, mass, J, 1 / data.core.freq
    )
    return data.replace(states=data.states.replace(force=force, torque=torque))


identified_derivative = analytical_derivative  # We can use the same derivative function for both


def mujoco_wrench(data: SimData) -> SimData:
    """Compute the wrench from the MuJoCo dynamics model."""
    forces = rpms2motor_forces(data.controls.rpms)
    torques = SIGN_MIX_MATRIX[..., 3] * rpms2motor_torques(data.controls.rpms)
    return data.replace(states=data.states.replace(motor_forces=forces, motor_torques=torques))


batched_mjx_step = jax.vmap(mjx.step, in_axes=(None, 0))


def mjx_physics_fn(data: SimData, mjx_data: Data, mjx_model: Model) -> SimData:
    """Step the MuJoCo simulation."""
    force_torques = jnp.concatenate([data.states.motor_forces, data.states.motor_torques], axis=-1)
    force_torques = rearrange(force_torques, "w d ft -> w (d ft)")
    mjx_data = mjx_data.replace(ctrl=force_torques)
    # TODO: Add disturbances from data.states.force/torque with mjx_data.xfrc_applied
    mjx_data = batched_mjx_step(mjx_model, mjx_data)
    return mjx_data


def identity(data: SimData) -> SimData:
    """Identity function for the simulation pipeline.

    Used as default function for optional pipeline steps.
    """
    return data


mjx_kinematics = jax.vmap(mjx.kinematics, in_axes=(None, 0))
mjx_collision = jax.vmap(mjx.collision, in_axes=(None, 0))
