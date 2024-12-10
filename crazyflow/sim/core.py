from functools import partial
from pathlib import Path
from typing import Any, Callable, TypeVar

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
from einops import rearrange
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from jax import Array
from mujoco.mjx import Data, Model

from crazyflow.constants import J_INV, J
from crazyflow.control.controller import Control, attitude2rpm
from crazyflow.exception import NotInitializedError
from crazyflow.sim.fused import fused_analytical_dynamics, fused_identified_dynamics, state2attitude
from crazyflow.sim.integration import Integrator
from crazyflow.sim.physics import Physics
from crazyflow.sim.structs import (
    SimControls,
    SimData,
    SimParams,
    SimState,
    default_controls,
    default_core,
    default_params,
    default_state,
)
from crazyflow.utils import clone_body, grid_2d


class Sim:
    default_path = Path(__file__).parents[1] / "models/cf2/scene.xml"

    def __init__(
        self,
        n_worlds: int = 1,
        n_drones: int = 1,
        physics: Physics = Physics.default,
        control: Control = Control.default,
        integrator: Integrator = Integrator.default,
        freq: int = 500,
        control_freq: int = 500,
        device: str = "cpu",
        xml_path: Path | None = None,
    ):
        assert Physics(physics) in Physics, f"Physics mode {physics} not implemented"
        assert Control(control) in Control, f"Control mode {control} not implemented"
        self.physics = physics
        self.control = control
        self.integrator = integrator
        self.device = jax.devices(device)[0]
        self.n_worlds = n_worlds
        self.n_drones = n_drones
        # Allocate internal states and controls
        states = default_state(n_worlds, n_drones, self.device)
        controls = default_controls(n_worlds, n_drones, control_freq, control_freq, self.device)
        params = default_params(n_worlds, n_drones, 0.025, J, J_INV, self.device)
        sim = default_core(freq, jnp.zeros((n_worlds, 1), dtype=jnp.int32, device=self.device))
        self.data = SimData(states=states, controls=controls, params=params, sim=sim)
        self.default_data: SimData | None = None  # Populated at the end of self.setup()
        # Initialize MuJoCo world and data
        self._xml_path = xml_path or self.default_path
        self._spec, self._mj_model, self._mj_data, self._mjx_model, self._mjx_data = self.setup_mj()
        self.viewer: MujocoRenderer | None = None
        self._step = self.setup_pipeline()  # Overwrite _step with the compiled pipeline

    def setup_mj(self) -> tuple[Any, Any, Any, Model, Data]:
        assert self._xml_path.exists(), f"Model file {self._xml_path} does not exist"
        spec = mujoco.MjSpec.from_file(str(self._xml_path))
        # Add additional drones to the world
        for i in range(1, self.n_drones):
            clone_body(spec.worldbody, spec.find_body("drone0"), f"drone{i}")
        mj_model = spec.compile()
        mj_data = mujoco.MjData(mj_model)
        mjx_model = mjx.put_model(mj_model, device=self.device)
        mjx_data = mjx.put_data(mj_model, mj_data, device=self.device)
        mjx_data = jax.vmap(lambda _: mjx_data)(jnp.arange(self.n_worlds))
        if self.n_drones > 1:  # If multiple drones, arrange them in a grid
            grid = grid_2d(self.n_drones)
            states = self.data.states.replace(pos=self.data.states.pos.at[..., :2].set(grid))
            self.data = self.data.replace(states=states)
            mjx_data = self.sync_sim2mjx(self.data, mjx_data, mjx_model)
            # Update default to reflect changes after resetting
        self.default_data = self.data.replace()
        return spec, mj_model, mj_data, mjx_model, mjx_data

    def setup_pipeline(self) -> Callable[[SimData, int], SimData]:
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
        physics_fn = generate_physics_fn(self.physics)

        # None is required by jax.lax.scan to unpack the tuple returned by single_step.
        def single_step(data: SimData, _: None) -> tuple[SimData, None]:
            data = ctrl_fn(data)
            data = physics_fn(data)
            # data = integrator_fn(data)
            data = data.replace(sim=data.sim.replace(steps=data.sim.steps + 1))
            return data, None

        # ``scan`` can be lowered to a single WhileOp, reducing compilation times while still fusing
        # the loops and giving XLA maximum freedom to reorder operations and jointly optimize the
        # pipeline. This is especially relevant for the common use case of running multiple sim
        # steps in an outer loop, e.g. in gym environments.
        # Having n_steps as a static argument is fine, since patterns with n_steps > 1 will almost
        # always use the same n_steps value for successive calls.
        @partial(jax.jit, static_argnames="n_steps")
        def step(
            data: SimData, mjx_data: Data, mjx_model: Model, n_steps: int = 1
        ) -> tuple[SimData, Data]:
            data, _ = jax.lax.scan(single_step, data, length=n_steps)
            mjx_data = self.sync_sim2mjx(data, mjx_data, mjx_model)
            return data, mjx_data

        return step

    def reset(self, mask: Array | None = None):
        """Reset the simulation to the initial state.

        Args:
            mask: Boolean array of shape (n_worlds, ) that indicates which worlds to reset. If None,
                all worlds are reset.
        """
        assert mask is None or mask.shape == (self.n_worlds,), f"Mask shape mismatch {mask.shape}"
        self.data = self._reset(self.data, self.default_data, mask)
        self._mjx_data = self.sync_sim2mjx(self.data, self._mjx_data, self._mjx_model)

    def step(self, n_steps: int = 1):
        """Simulate all drones in all worlds for n time steps."""
        assert n_steps > 0, "Number of steps must be positive"
        self.data, self._mjx_data = self._step(self.data, self._mjx_data, self._mjx_model, n_steps)

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
        self.controls = self._thrust_control(cmd, self.controls, self.device)

    def render(self):
        if self.viewer is None:
            self.viewer = MujocoRenderer(self._mj_model, self._mj_data)
        self._mj_data.qpos[:] = mjx.get_data(m=self._mj_model, d=self._mjx_data)[0].qpos
        mujoco.mj_forward(self._mj_model, self._mj_data)
        self.viewer.render("human")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    @property
    def time(self) -> Array:
        return self.data.sim.steps / self.data.sim.freq

    @property
    def freq(self) -> int:
        return self.data.sim.freq

    @property
    def control_freq(self) -> int:
        if self.control == Control.state:
            return self.data.controls.state_freq
        if self.control == Control.attitude:
            return self.data.controls.attitude_freq
        if self.control == Control.thrust:
            raise NotImplementedError("Thrust control is not yet supported by the sim config")
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
            case _:
                raise NotImplementedError(f"Control mode {self.control} not implemented")
        return controllable(self.data.sim.steps, self.data.sim.freq, control_steps, control_freq)

    def contacts(self, body: str | None = None) -> Array:
        """Get contact information from the simulation.

        Args:
            body: Optional body name to filter contacts for. If None, returns flags for all bodies.

        Returns:
            An boolean array of shape (n_worlds,) that is True if any contact is present.
        """
        if body is None:
            return self._mjx_data.contact.dist < 0
        body_id = self._mj_model.body(body).id
        geom_start = self._mj_model.body_geomadr[body_id]
        geom_count = self._mj_model.body_geomnum[body_id]
        return contacts(geom_start, geom_count, self._mjx_data)

    @staticmethod
    @jax.jit
    def sync_sim2mjx(data: SimData, mjx_data: Data, mjx_model: Model) -> Data:
        states = data.states
        pos, quat, vel, ang_vel = states.pos, states.quat, states.vel, states.ang_vel
        quat = quat[..., [-1, 0, 1, 2]]  # MuJoCo quat is [w, x, y, z], ours is [x, y, z, w]
        qpos = rearrange(jnp.concat([pos, quat], axis=-1), "w d qpos -> w (d qpos)")
        qvel = rearrange(jnp.concat([vel, ang_vel], axis=-1), "w d qvel -> w (d qvel)")
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        mjx_data = mjx_kinematics(mjx_model, mjx_data)
        mjx_data = mjx_collision(mjx_model, mjx_data)
        return mjx_data

    @staticmethod
    @jax.jit
    def _reset(data: SimData, default_data: SimData, mask: Array | None = None) -> SimData:
        data = data.replace(states=pytree_replace(data.states, default_data.states, mask))
        data = data.replace(controls=pytree_replace(data.controls, default_data.controls, mask))
        data = data.replace(params=pytree_replace(data.params, default_data.params, mask))
        return data

    def _step(self, data: SimData, n_steps: int) -> SimData:
        raise NotInitializedError("_step call before compiling the simulation pipeline.")


def generate_control_fn(control: Control) -> Callable[[SimData], SimData]:
    """Generate the control function for the given control mode."""
    match control:
        case Control.state:
            return lambda data: step_attitude_controller(step_state_controller(data))
        case Control.attitude:
            return step_attitude_controller
        case _:
            raise NotImplementedError(f"Control mode {control} not implemented")


def generate_physics_fn(physics: Physics) -> Callable[[SimData], SimData]:
    """Generate the physics function for the given physics mode."""
    match physics:
        case Physics.analytical:
            return fused_analytical_dynamics
        case Physics.sys_id:
            return fused_identified_dynamics
        case _:
            raise NotImplementedError(f"Physics mode {physics} not implemented")


def generate_integrator_fn(integrator: Integrator) -> Callable[[SimData], SimData]:
    """Generate the integrator function for the given integrator mode."""
    match integrator:
        case Integrator.euler:
            return ...
        case _:
            raise NotImplementedError(f"Integrator {integrator} not implemented")


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
def state_control(controls: Array, data: SimData, device: str) -> SimData:
    controls = jnp.array(controls, device=device)
    return data.replace(controls=data.controls.replace(state=controls))


@jax.jit
def controllable(step: Array, freq: int, control_steps: Array, control_freq: int) -> Array:
    return ((step - control_steps) >= (freq / control_freq)) | (control_steps == 0)


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
    controls = data.controls
    mask = controllable(data.sim.steps, data.sim.freq, controls.state_steps, controls.state_freq)
    state_steps = jnp.where(mask, data.sim.steps, controls.state_steps)
    data = data.replace(controls=controls.replace(state_steps=state_steps))
    return state2attitude(data)


def step_attitude_controller(data: SimData) -> SimData:
    """Compute the updated controls for the attitude controller."""
    controls = data.controls
    steps, freq = data.sim.steps, data.sim.freq
    mask = controllable(steps, freq, controls.attitude_steps, controls.attitude_freq)
    data = _commit_attitude_controls(data, mask)
    return _step_attitude_controller(data, mask)


def _commit_attitude_controls(data: SimData, mask: Array) -> SimData:
    mask = mask.reshape(-1, 1, 1)
    controls = data.controls
    staged_attitude = controls.staged_attitude
    controls = leaf_replace(controls, mask, attitude_steps=data.sim.steps, attitude=staged_attitude)
    return data.replace(controls=controls)


def _step_attitude_controller(data: SimData, mask: Array) -> SimData:
    mask = mask.reshape(-1, 1, 1)
    quat, attitude = data.states.quat, data.controls.attitude
    last_rpy, rpy_err_i = data.controls.last_rpy, data.controls.rpy_err_i
    rpms, rpy_err_i = attitude2rpm(attitude, quat, last_rpy, rpy_err_i, 1 / data.sim.freq)
    controls = leaf_replace(data.controls, mask, rpms=rpms, rpy_err_i=rpy_err_i, last_rpy=last_rpy)
    return data.replace(controls=controls)


T = TypeVar("T", SimState, SimControls, SimParams)


def pytree_replace(data: T, defaults: T, mask: Array | None = None) -> T:
    """Overwrite elements of a pytree with values from another pytree filtered by a mask.

    The mask indicates which elements of the leaf arrays to overwrite with new values, and which
    ones to leave unchanged.
    """

    def masked_replace(x: Array, y: Array) -> Array:
        """Resize the mask to match the shape of x and select from x and y accordingly."""
        _mask = jnp.ones((x.shape[0],)) if mask is None else mask
        _mask = _mask.reshape(-1, *[1] * (x.ndim - 1))
        return jnp.where(_mask, y, x)

    return jax.tree.map(masked_replace, data, defaults)


def leaf_replace(data: T, mask: Array | None = None, **kwargs: dict[str, Array]) -> T:
    """Replace elements of a pytree with the given keyword arguments.

    If a mask is provided, the replacement is applied only to the elements indicated by the mask.

    Args:
        data: The pytree to be modified.
        mask: Boolean array matching the first dimension of all kwargs entries in data.
        kwargs: Leaf names and their replacement values.
    """
    replace = {}
    for k, v in kwargs.items():
        replace[k] = jnp.where(mask.reshape(-1, *[1] * (v.ndim - 1)), v, getattr(data, k))
    return data.replace(**replace)


mjx_kinematics = jax.vmap(mjx.kinematics, in_axes=(None, 0))
mjx_collision = jax.vmap(mjx.collision, in_axes=(None, 0))
