from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
from einops import rearrange
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from jax import Array
from mujoco.mjx import Data, Model

from crazyflow.control.controller import J_INV, Control, Controller, J
from crazyflow.exception import ConfigError, NotInitializedError
from crazyflow.sim.fused import (
    fused_analytical_dynamics,
    fused_identified_dynamics,
    fused_masked_attitude2rpm,
    fused_masked_state2attitude,
    fused_rpms2collective_wrench,
)
from crazyflow.sim.physics import Physics
from crazyflow.sim.structs import SimControls, SimParams, SimState
from crazyflow.utils import clone_body, grid_2d


class Sim:
    default_path = Path(__file__).parents[1] / "models/cf2/scene.xml"

    def __init__(
        self,
        n_worlds: int = 1,
        n_drones: int = 1,
        physics: Physics = Physics.default,
        control: Control = Control.default,
        controller: Controller = Controller.default,
        freq: int = 500,
        control_freq: int = 500,
        device: str = "cpu",
        xml_path: Path | None = None,
    ):
        assert Physics(physics) in Physics, f"Physics mode {physics} not implemented"
        assert Control(control) in Control, f"Control mode {control} not implemented"
        assert Controller(controller) in Controller, f"Controller {controller} not implemented"
        self.physics = physics
        self.control = control
        self.controller = controller
        self.device = jax.devices(device)[0]
        self.freq = freq
        self.control_freq = control_freq
        self.dt = jnp.array(1 / freq, device=self.device)
        # pycffirmware uses global states which make it impossible to simulate multiple drones at
        # the same time. We raise if the user tries a combination of pycffirmware and drones > 1.
        if controller == Controller.pycffirmware and n_worlds * n_drones != 1:
            raise ConfigError("pycffirmware controller is only supported for single drone sims")
        # Allocate internal states and controls for analytical and sys_id physics.
        self.n_worlds = n_worlds
        self.n_drones = n_drones
        self.defaults = {}  # Populated at the end of self.setup()
        self._default_mask = jnp.ones((n_worlds,), dtype=bool, device=self.device)
        self.steps = jnp.zeros((n_worlds,), dtype=int, device=self.device)
        # We need to set the last control step to -freq to ensure that the first control
        # command is applied at t=0
        self.last_ctrl_steps = jnp.ones((n_worlds,), dtype=int, device=self.device) * -freq
        self.states = SimState(
            pos=jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            quat=jnp.zeros((n_worlds, n_drones, 4), device=self.device),
            vel=jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            ang_vel=jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            rpy_rates=jnp.zeros((n_worlds, n_drones, 3), device=self.device),
        )
        self.states = self.states.replace(quat=self.states.quat.at[..., -1].set(1.0))
        # Allocate internal control variables and physics parameters.
        self.controls = SimControls(
            state=jnp.zeros((n_worlds, n_drones, 13), device=self.device),
            staged_state=jnp.zeros((n_worlds, n_drones, 13), device=self.device),
            attitude=jnp.zeros((n_worlds, n_drones, 4), device=self.device),
            staged_attitude=jnp.zeros((n_worlds, n_drones, 4), device=self.device),
            thrust=jnp.zeros((n_worlds, n_drones, 4), device=self.device),
            rpms=jnp.zeros((n_worlds, n_drones, 4), device=self.device),
            rpy_err_i=jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            pos_err_i=jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            last_rpy=jnp.zeros((n_worlds, n_drones, 3), device=self.device),
        )
        # Allocate physics parameter buffers.
        j, j_inv = jnp.array(J, device=self.device), jnp.array(J_INV, device=self.device)
        self.params = SimParams(
            mass=jnp.ones((n_worlds, n_drones, 1), device=self.device) * 0.025,
            J=jnp.tile(j[None, None, :, :], (n_worlds, n_drones, 1, 1)),
            J_INV=jnp.tile(j_inv[None, None, :, :], (n_worlds, n_drones, 1, 1)),
        )
        # Initialize MuJoCo world and data
        self._xml_path = xml_path or self.default_path
        self._spec, self._mj_model, self._mj_data, self._mjx_model, self._mjx_data = self.setup()
        self.viewer: MujocoRenderer | None = None

    def setup(self) -> tuple[Any, Any, Any, Model, Data]:
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
        # Optimization: Compile the sync function with the finalized model (mjx_model is not
        # changing anymore) to avoid overhead. See Sim._sync_mjx for details.
        self._sync_mjx = jax.jit(partial(self._sync_mjx_full, mjx_model=mjx_model))
        if self.n_drones > 1:  # If multiple drones, arrange them in a grid
            grid = grid_2d(self.n_drones)
            self.states = self.states.replace(pos=self.states.pos.at[..., :2].set(grid))
            self._mjx_data = self._sync_mjx(self.states, mjx_data)
            # Update default to reflect changes after resetting
        self.defaults["states"] = self.states.replace()
        self.defaults["controls"] = self.controls.replace()
        self.defaults["params"] = self.params.replace()
        return spec, mj_model, mj_data, mjx_model, mjx_data

    def reset(self, mask: Array | None = None):
        """Reset the simulation to the initial state.

        Args:
            mask: Boolean array of shape (n_worlds, ) that indicates which worlds to reset. If None,
                all worlds are reset.
        """
        mask = self._default_mask if mask is None else mask
        assert mask.shape == (self.n_worlds,), f"Mask shape mismatch {mask.shape}"
        self.states = self._masked_states_reset(mask, self.states, self.defaults["states"])
        self.controls = self._masked_controls_reset(mask, self.controls, self.defaults["controls"])
        self.params = self._masked_params_reset(mask, self.params, self.defaults["params"])
        self.steps = jnp.where(mask, 0, self.steps)
        self.last_ctrl_steps = jnp.where(mask, -self.freq, self.last_ctrl_steps)
        self._sync_mjx(self.states, self._mjx_data)

    def step(self):
        """Simulate all drones in all worlds for one time step."""
        match self.physics:
            case Physics.mujoco:
                raise NotImplementedError
            case Physics.analytical:
                self._step_analytical()
            case Physics.sys_id:
                self._step_sys_id()
            case _:
                raise ValueError(f"Physics mode {self.physics} not implemented")
        self.steps = self.steps + 1

    def attitude_control(self, cmd: Array):
        """Set the desired attitude for all drones in all worlds."""
        assert cmd.shape == (self.n_worlds, self.n_drones, 4), "Command shape mismatch"
        assert self.control == Control.attitude, "Attitude control is not enabled by the sim config"
        self.controls = self._attitude_control(cmd, self.controls, self.device)

    def state_control(self, cmd: Array):
        """Set the desired state for all drones in all worlds."""
        assert cmd.shape == self.controls.state.shape, f"Command shape mismatch {cmd.shape}"
        assert self.control == Control.state, "State control is not enabled by the sim config"
        self.controls = self._state_control(cmd, self.controls, self.device)

    def thrust_control(self, cmd: Array):
        raise NotImplementedError

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
        return self.steps / self.freq

    @property
    def controllable(self) -> Array:
        """Boolean array of shape (n_worlds,) that indicates which worlds are controllable.

        A world is controllable if the last control step was more than 1/control_freq seconds ago.
        Desired control commands get stashed in the staged control buffers and are applied in `step`
        as soon as the controller frequency allows for an update. Successive control updates that
        happen before the staged buffers are applied overwrite the desired values.
        """
        return self._controllable(self.steps, self.last_ctrl_steps, self.control_freq, self.freq)

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

    def _step_sys_id(self):
        mask = self.controllable
        # Optional optimization: check if mask.any() before updating the controls. This breaks jax's
        # gradient tracing, so we omit it for now.
        if self.control == Control.state:
            self.controls = self._masked_state_controls_update(mask, self.controls)
            self.controls = fused_masked_state2attitude(mask, self.states, self.controls, self.dt)
        self.controls = self._masked_attitude_controls_update(mask, self.controls)
        self.last_ctrl_steps = self._masked_controls_step_update(
            mask, self.steps, self.last_ctrl_steps
        )
        self.states = fused_identified_dynamics(self.states, self.controls, self.dt)
        self._mjx_data = self._sync_mjx(self.states, self._mjx_data)

    def _step_analytical(self):
        mask = self.controllable
        # Optional optimization: check if mask.any() before updating the controls. This breaks jax's
        # gradient tracing, so we omit it for now.
        match self.controller:
            case Controller.emulatefirmware:
                self.controls = self._step_emulate_firmware()
            case Controller.pycffirmware:
                raise NotImplementedError
            case _:
                raise ValueError(f"Controller {self.controller} not implemented")
        self.last_ctrl_steps = self._masked_controls_step_update(
            mask, self.steps, self.last_ctrl_steps
        )
        forces, torques = fused_rpms2collective_wrench(self.states, self.controls, self.params)
        self.states = fused_analytical_dynamics(forces, torques, self.states, self.params, self.dt)
        self._mjx_data = self._sync_mjx(self.states, self._mjx_data)

    def _step_emulate_firmware(self) -> SimControls:
        mask = self.controllable
        if self.control == Control.state:
            self.controls = self._masked_state_controls_update(mask, self.controls)
            self.controls = fused_masked_state2attitude(mask, self.states, self.controls, self.dt)
        self.controls = self._masked_attitude_controls_update(mask, self.controls)
        return fused_masked_attitude2rpm(mask, self.states, self.controls, self.dt)

    @staticmethod
    def _sync_mjx(states: SimState, mjx_data: Data) -> Data:
        """Sync the states to the MuJoCo data.

        We initialize this function in Sim.setup() to compile it with the finalized MuJoCo model.
        This allows us to avoid the overhead associated with flattening and unflattening the model
        struct on every call in Sim._sync_mjx_full.

        Warning:
            Raises NotInitializedError if Sim.setup() was not called yet.

        Warning:
            If the model changes, the sync function needs to be recompiled with the new model.
        """
        raise NotInitializedError("MuJoCo sync function not initialized, call Sim.setup() first")

    @staticmethod
    def _sync_mjx_full(states: SimState, mjx_data: Data, mjx_model: Model) -> Data:
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
    def _masked_states_reset(mask: Array, states: SimState, defaults: SimState) -> SimState:
        mask_3d = mask[:, None, None]
        states = states.replace(pos=jnp.where(mask_3d, defaults.pos, states.pos))
        states = states.replace(quat=jnp.where(mask_3d, defaults.quat, states.quat))
        states = states.replace(vel=jnp.where(mask_3d, defaults.vel, states.vel))
        states = states.replace(ang_vel=jnp.where(mask_3d, defaults.ang_vel, states.ang_vel))
        states = states.replace(rpy_rates=jnp.where(mask_3d, defaults.rpy_rates, states.rpy_rates))
        return states

    @staticmethod
    @jax.jit
    def _masked_controls_reset(
        mask: Array, controls: SimControls, defaults: SimControls
    ) -> SimControls:
        mask = mask[:, None, None]
        controls = controls.replace(
            state=jnp.where(mask, defaults.state, controls.state),
            attitude=jnp.where(mask, defaults.attitude, controls.attitude),
            thrust=jnp.where(mask, defaults.thrust, controls.thrust),
            rpms=jnp.where(mask, defaults.rpms, controls.rpms),
            rpy_err_i=jnp.where(mask, defaults.rpy_err_i, controls.rpy_err_i),
            pos_err_i=jnp.where(mask, defaults.pos_err_i, controls.pos_err_i),
            last_rpy=jnp.where(mask, defaults.last_rpy, controls.last_rpy),
            staged_attitude=jnp.where(mask, defaults.staged_attitude, controls.staged_attitude),
            staged_state=jnp.where(mask, defaults.staged_state, controls.staged_state),
        )
        return controls

    @staticmethod
    @jax.jit
    def _masked_params_reset(mask: Array, params: SimParams, defaults: SimParams) -> SimParams:
        params = params.replace(mass=jnp.where(mask[:, None, None], defaults.mass, params.mass))
        mask_4d = mask[:, None, None, None]
        params = params.replace(J=jnp.where(mask_4d, defaults.J, params.J))
        params = params.replace(J_INV=jnp.where(mask_4d, defaults.J_INV, params.J_INV))
        return params

    @staticmethod
    @partial(jax.jit, static_argnames="device")
    def _attitude_control(cmd: Array, controls: SimControls, device: str) -> SimControls:
        return controls.replace(staged_attitude=jnp.array(cmd, device=device))

    @staticmethod
    @partial(jax.jit, static_argnames="device")
    def _state_control(cmd: Array, controls: SimControls, device: str) -> SimControls:
        return controls.replace(staged_state=jnp.array(cmd, device=device))

    @staticmethod
    @jax.jit
    def _masked_attitude_controls_update(mask: Array, controls: SimControls) -> SimControls:
        cmd, staged_cmd = controls.attitude, controls.staged_attitude
        return controls.replace(attitude=jnp.where(mask[:, None, None], staged_cmd, cmd))

    @staticmethod
    @jax.jit
    def _masked_state_controls_update(mask: Array, controls: SimControls) -> SimControls:
        cmd, staged_cmd = controls.state, controls.staged_state
        return controls.replace(state=jnp.where(mask[:, None, None], staged_cmd, cmd))

    @staticmethod
    @jax.jit
    def _masked_controls_step_update(mask: Array, steps: Array, last_ctrl_steps: Array) -> Array:
        return jnp.where(mask, steps, last_ctrl_steps)

    @staticmethod
    @jax.jit
    def _controllable(step: Array, ctrl_step: Array, ctrl_freq: int, freq: int) -> Array:
        return (step - ctrl_step) >= (freq / ctrl_freq)


@jax.jit
def contacts(geom_start: int, geom_count: int, data: Data) -> Array:
    """Filter contacts from MuJoCo data."""
    geom1_valid = data.contact.geom1 >= geom_start
    geom1_valid &= data.contact.geom1 < geom_start + geom_count
    geom2_valid = data.contact.geom2 >= geom_start
    geom2_valid &= data.contact.geom2 < geom_start + geom_count
    return data.contact.dist < 0 & (geom1_valid | geom2_valid)


mjx_kinematics = jax.vmap(mjx.kinematics, in_axes=(None, 0))
mjx_collision = jax.vmap(mjx.collision, in_axes=(None, 0))
