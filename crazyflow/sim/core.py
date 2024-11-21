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
from jax.scipy.spatial.transform import Rotation as R
from mujoco.mjx import Data, Model

from crazyflow.control.controller import J_INV, Control, Controller, J, attitude2rpm, state2attitude
from crazyflow.exception import ConfigError
from crazyflow.sim.physics import (
    Physics,
    analytical_dynamics,
    identified_dynamics,
    rpms2collective_wrench,
)
from crazyflow.sim.structs import SimControls, SimParams, SimState
from crazyflow.utils import clone_body


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
        if physics == Physics.sys_id and control == Control.state:
            raise ConfigError("sys_id physics does not support state control")  # TODO: Implement
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
        self.states = SimState(
            step=jnp.zeros((n_worlds,), device=self.device),
            pos=jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            quat=jnp.zeros((n_worlds, n_drones, 4), device=self.device),
            vel=jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            ang_vel=jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            rpy_rates=jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            device=self.device,
        )
        self.states = self.states.replace(quat=self.states.quat.at[..., -1].set(1.0))
        # Allocate internal control variables and physics parameters.
        self.controls = SimControls(
            state=jnp.zeros((n_worlds, n_drones, 13), device=self.device),
            attitude=jnp.zeros((n_worlds, n_drones, 4), device=self.device),
            thrust=jnp.zeros((n_worlds, n_drones, 4), device=self.device),
            rpms=jnp.zeros((n_worlds, n_drones, 4), device=self.device),
            rpy_err_i=jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            pos_err_i=jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            last_rpy=jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            device=self.device,
        )
        # Allocate physics parameter buffers.
        j, j_inv = jnp.array(J, device=self.device), jnp.array(J_INV, device=self.device)
        self.params = SimParams(
            mass=jnp.ones((n_worlds, n_drones, 1), device=self.device) * 0.025,
            J=jnp.tile(j[None, None, :, :], (n_worlds, n_drones, 1, 1)),
            J_INV=jnp.tile(j_inv[None, None, :, :], (n_worlds, n_drones, 1, 1)),
            device=self.device,
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
        if self.n_drones > 1:  # If multiple drones, arrange them in a grid
            N = int(jnp.ceil(jnp.sqrt(self.n_drones)))
            points = jnp.linspace(-0.5 * (N - 1), 0.5 * (N - 1), N)
            x, y = jnp.meshgrid(points, points)
            grid = jnp.stack((x.flatten(), y.flatten()), axis=-1)
            grid = grid[: self.n_drones]
            self.states = self.states.replace(pos=self.states.pos.at[..., :2].set(grid))
            self._mjx_data = self._sync_mjx(self.states, mjx_model, mjx_data)
            # Update default to reflect changes after resetting
        self.defaults["states"] = self.states.replace()
        self.defaults["controls"] = self.controls.replace()
        self.defaults["params"] = self.params.replace()
        return None, mj_model, mj_data, mjx_model, mjx_data

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
        self._sync_mjx(self.states, self._mjx_model, self._mjx_data)

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
        self.states = self.states.replace(step=self.states.step + 1)

    def attitude_control(self, cmd: Array):
        assert cmd.shape == (self.n_worlds, self.n_drones, 4), "Command shape mismatch"
        assert self.control == Control.attitude, "Attitude control is not enabled by the sim config"
        # sys_id physics uses the controller input to compute the physics step, so changes in the
        # control input are always visible. To simulate the controller at a lower frequency, we only
        # update the attitude buffer at the control frequency. This does not apply to other
        # physics / controller combinations.
        if self.physics == Physics.sys_id:
            mask = self.controllable
            self.controls = self._masked_attitude_controls_update(mask, self.controls, cmd)
            return
        self.controls = self.controls.replace(attitude=jnp.array(cmd, device=self.device))

    def state_control(self, cmd: Array):
        """Set the desired state for all drones in all worlds."""
        assert cmd.shape == self.controls.state.shape, f"Command shape mismatch {cmd.shape}"
        assert self.control == Control.state, "State control is not enabled by the sim config"
        self.controls = self.controls.replace(state=jnp.array(cmd, device=self.device))

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
        return self.states.step / self.freq

    @property
    def controllable(self) -> Array:
        """Boolean array of shape (n_worlds,) that indicates which worlds are controllable."""
        return self._controllable(self.states.step, self.control_freq, self.freq)

    @staticmethod
    @jax.jit
    def _controllable(step: Array, ctrl_freq: int, freq: int) -> Array:
        return step * ctrl_freq % freq < ctrl_freq

    def _step_sys_id(self):
        c = self.controllable
        if c.any() and self.control == Control.state:
            attitude_cmd, pos_err_i = state2attitude(
                self.states.pos[c],
                self.states.vel[c],
                self.states.quat[c],
                self.controls.state[c, ..., :3],
                self.controls.state[c, ..., 3:6],
                self.controls.state[c, ..., 9:10],
                self.controls.pos_err_i[c],
                self.dt,
            )
            self.controls = self.controls.replace(
                attitude=self.controls.attitude.at[c].set(attitude_cmd),
                pos_err_i=self.controls.pos_err_i.at[c].set(pos_err_i),
            )
        pos, quat, vel, ang_vel = identified_dynamics(
            self.controls.attitude,
            self.states.pos,
            self.states.quat,
            self.states.vel,
            self.states.ang_vel,
            self.dt,
        )
        self._sync_states(pos, quat, vel, ang_vel)
        self._mjx_data = self._sync_mjx(self.states, self._mjx_model, self._mjx_data)

    def _step_analytical(self):
        # Only update the RPMs at the control frequency
        c = self.controllable
        if c.any():
            match self.controller:
                case Controller.emulatefirmware:
                    rpms = self._step_emulate_firmware()
                case Controller.pycffirmware:
                    raise NotImplementedError
                case _:
                    raise ValueError(f"Controller {self.controller} not implemented")
            self.controls = self.controls.replace(rpms=self.controls.rpms.at[c].set(rpms))

        self.controls = self.controls.replace(last_rpy=quat2rpy(self.states.quat))
        forces, torques = rpms2collective_wrench(
            self.controls.rpms, self.states.quat, self.states.rpy_rates, self.params.J
        )
        pos, quat, vel, rpy_rates = analytical_dynamics(
            forces,
            torques,
            self.states.pos,
            self.states.quat,
            self.states.vel,
            self.states.rpy_rates,
            self.params.mass,
            self.params.J_INV,
            self.dt,
        )
        self._sync_states(pos, quat, vel, rpy_rates=rpy_rates)
        self._mjx_data = self._sync_mjx(self.states, self._mjx_model, self._mjx_data)

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

    def _step_emulate_firmware(self) -> Array:
        c = self.controllable
        if self.control == Control.state:
            attitude_cmd, pos_err_i = state2attitude(
                self.states.pos[c],
                self.states.vel[c],
                self.states.quat[c],
                self.controls.state[c, ..., :3],
                self.controls.state[c, ..., 3:6],
                self.controls.state[c, ..., 9:10],
                self.controls.pos_err_i[c],
                self.dt,
            )
            self.controls = self.controls.replace(
                attitude=self.controls.attitude.at[c].set(attitude_cmd),
                pos_err_i=self.controls.pos_err_i.at[c].set(pos_err_i),
            )
        rpms, rpy_err_i = attitude2rpm(
            self.controls.attitude[c],
            self.states.quat[c],
            self.controls.last_rpy[c],
            self.controls.rpy_err_i[c],
            self.dt,
        )
        self.controls = self.controls.replace(
            rpy_err_i=self.controls.rpy_err_i.at[c].set(rpy_err_i)
        )
        return rpms

    def _sync_states(
        self,
        pos: Array,
        quat: Array,
        vel: Array,
        ang_vel: Array | None = None,
        rpy_rates: Array | None = None,
    ):
        self.states = self.states.replace(pos=pos, quat=quat, vel=vel)
        if ang_vel is not None:
            self.states = self.states.replace(ang_vel=ang_vel)
        if rpy_rates is not None:
            self.states = self.states.replace(rpy_rates=rpy_rates)

    @staticmethod
    @jax.jit
    def _sync_mjx(states: SimState, mjx_model, mjx_data) -> Data:
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
        states = states.replace(step=jnp.where(mask, defaults.step, states.step))
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
        atts = ["state", "attitude", "thrust", "rpms", "rpy_err_i", "pos_err_i", "last_rpy"]
        mask = mask.reshape((-1, 1, 1))
        return controls.replace(
            **{k: jnp.where(mask, getattr(defaults, k), getattr(controls, k)) for k in atts}
        )

    @staticmethod
    @jax.jit
    def _masked_params_reset(mask: Array, params: SimParams, defaults: SimParams) -> SimParams:
        params = params.replace(mass=jnp.where(mask[:, None, None], defaults.mass, params.mass))
        mask_4d = mask[:, None, None, None]
        params = params.replace(J=jnp.where(mask_4d, defaults.J, params.J))
        params = params.replace(J_INV=jnp.where(mask_4d, defaults.J_INV, params.J_INV))
        return params

    @staticmethod
    @jax.jit
    def _masked_attitude_controls_update(
        mask: Array, controls: SimControls, cmd: Array
    ) -> SimControls:
        return controls.replace(attitude=jnp.where(mask[:, None, None], cmd, controls.attitude))


@jax.jit
def contacts(geom_start: int, geom_count: int, data: Data) -> Array:
    """Filter contacts from MuJoCo data."""
    geom1_valid = data.contact.geom1 >= geom_start
    geom1_valid &= data.contact.geom1 < geom_start + geom_count
    geom2_valid = data.contact.geom2 >= geom_start
    geom2_valid &= data.contact.geom2 < geom_start + geom_count
    return data.contact.dist < 0 & (geom1_valid | geom2_valid)


mjx_forward = jax.vmap(mjx.forward, in_axes=(None, 0))
mjx_kinematics = jax.vmap(mjx.kinematics, in_axes=(None, 0))
mjx_collision = jax.vmap(mjx.collision, in_axes=(None, 0))


quat2rpy = jax.jit(
    partial(jnp.vectorize, signature="(4)->(3)")(lambda x: R.from_quat(x).as_euler("xyz"))
)


@jax.jit
def jit_where(mask: Array, x: Array, y: Array) -> Array:
    mask = mask.reshape((-1,) + (1,) * (x.ndim - 1))  # Broadcast mask to match x and y
    return jnp.where(mask, x, y)
