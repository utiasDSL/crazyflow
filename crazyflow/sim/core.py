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
from crazyflow.sim.physics import Physics, analytical_dynamics, identified_dynamics
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
            raise ConfigError("sys_id physics does not support state control")
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
        self.states = {
            "pos": jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            "quat": jnp.zeros((n_worlds, n_drones, 4), device=self.device),
            "vel": jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            "ang_vel": jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            "rpy_rates": jnp.zeros((n_worlds, n_drones, 3), device=self.device),
        }
        self.states["quat"] = self.states["quat"].at[:, :, 3].set(1.0)
        # Allocate internal control variables and physics parameters.
        self._controls = {
            "state": jnp.zeros((n_worlds, n_drones, 13), device=self.device),
            "attitude": jnp.zeros((n_worlds, n_drones, 4), device=self.device),
            "thrust": jnp.zeros((n_worlds, n_drones, 4), device=self.device),
            "rpms": jnp.zeros((n_worlds, n_drones, 4), device=self.device),
            "rpy_err_i": jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            "pos_err_i": jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            "last_rpy": jnp.zeros((n_worlds, n_drones, 3), device=self.device),
        }
        self._params = {
            "mass": jnp.ones((n_worlds, n_drones, 1), device=self.device) * 0.025,
            "J": jnp.tile(J, n_worlds * n_drones).reshape(n_worlds, n_drones, 3, 3),
            "J_INV": jnp.tile(J_INV, n_worlds * n_drones).reshape(n_worlds, n_drones, 3, 3),
        }
        self._default_params = self._params.copy()
        self._step = 0
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
            self.states["pos"] = self.states["pos"].at[..., :2].set(grid)
            pos, quat = self.states["pos"], self.states["quat"]
            vel, ang_vel = self.states["vel"], self.states["ang_vel"]
            self._mjx_data = self._sync_mjx(pos, quat, vel, ang_vel, mjx_model, mjx_data)
        return None, mj_model, mj_data, mjx_model, mjx_data

    def reset(self):
        """Reset the simulation to the initial state.

        TODO: Reset all models, control variables, params, states etc.
        """
        self._step = 0
        for key in self._params:
            self._params[key] = self._default_params[key]

    def step(self):
        """Simulate all drones in all worlds for one time step."""
        self._step += 1
        match self.physics:
            case Physics.mujoco:
                raise NotImplementedError
            case Physics.analytical:
                self._step_analytical()
            case Physics.sys_id:
                self._step_sys_id()
            case _:
                raise ValueError(f"Physics mode {self.physics} not implemented")

    def attitude_control(self, cmd: Array):
        assert cmd.shape == (self.n_worlds, self.n_drones, 4), "Command shape mismatch"
        assert self.control == Control.attitude, "Attitude control is not enabled by the sim config"
        # sys_id physics uses the controller input to compute the physics step, so changes in the
        # control input are always visible. To simulate the controller at a lower frequency, we only
        # update the attitude buffer at the control frequency. This does not apply to other
        # physics / controller combinations.
        if self.physics == Physics.sys_id:
            if self._step * self.control_freq % self.freq < self.control_freq:
                self._controls["attitude"] = cmd
            return
        self._controls["attitude"] = cmd

    def state_control(self, cmd: Array):
        """Set the desired state for all drones in all worlds."""
        assert cmd.shape == self._controls["state"].shape, f"Command shape mismatch {cmd.shape}"
        assert self.control == Control.state, "State control is not enabled by the sim config"
        self._controls["state"] = cmd

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
    def time(self) -> float:
        return self._step / self.freq

    @property
    def controllable(self) -> bool:
        """True if the controller can update the control input this step, else False."""
        return self._step * self.control_freq % self.freq < self.control_freq

    def _step_sys_id(self):
        pos, quat, vel, ang_vel = identified_dynamics(
            self._controls["attitude"],
            self.states["pos"],
            self.states["quat"],
            self.states["vel"],
            self.states["ang_vel"],
            self.dt,
        )
        self._sync_states(pos, quat, vel, ang_vel)
        self._mjx_data = self._sync_mjx(pos, quat, vel, ang_vel, self._mjx_model, self._mjx_data)

    def _step_analytical(self):
        # Only update the RPMs at the control frequency
        if self._step * self.control_freq % self.freq < self.control_freq:
            match self.controller:
                case Controller.emulatefirmware:
                    rpms = self._step_emulate_firmware()
                case Controller.pycffirmware:
                    raise NotImplementedError
                case _:
                    raise ValueError(f"Controller {self.controller} not implemented")
            self._controls["rpms"] = rpms

        self._controls["last_rpy"] = quat2rpy(self.states["quat"])
        pos, quat, vel, rpy_rates = analytical_dynamics(
            self._controls["rpms"],
            self.states["pos"],
            self.states["quat"],
            self.states["vel"],
            self.states["rpy_rates"],
            self._params["mass"],
            self._params["J"],
            self._params["J_INV"],
            self.dt,
        )
        self._sync_states(pos, quat, vel, rpy_rates=rpy_rates)
        ang_vel = self.states["ang_vel"]
        self._mjx_data = self._sync_mjx(pos, quat, vel, ang_vel, self._mjx_model, self._mjx_data)

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
        if self.control == Control.state:
            attitude_cmd, pos_err_i = state2attitude(
                self.states["pos"],
                self.states["vel"],
                self.states["quat"],
                self._controls["state"][..., :3],
                self._controls["state"][..., 3:6],
                self._controls["state"][..., 9:10],
                self._controls["pos_err_i"],
                self.dt,
            )
            self._controls["attitude"] = attitude_cmd
            self._controls["pos_err_i"] = pos_err_i
        rpms, rpy_err_i = attitude2rpm(
            self._controls["attitude"],
            self.states["quat"],
            self._controls["last_rpy"],
            self._controls["rpy_err_i"],
            self.dt,
        )
        self._controls["rpy_err_i"] = rpy_err_i
        return rpms

    def _sync_states(
        self,
        pos: Array,
        quat: Array,
        vel: Array,
        ang_vel: Array | None = None,
        rpy_rates: Array | None = None,
    ):
        self.states["pos"], self.states["quat"] = pos, quat
        self.states["vel"] = vel
        if ang_vel is not None:
            self.states["ang_vel"] = ang_vel
        if rpy_rates is not None:
            self.states["rpy_rates"] = rpy_rates

    @staticmethod
    @jax.jit
    def _sync_mjx(pos: Array, quat: Array, vel: Array, ang_vel: Array, mjx_model, mjx_data) -> Data:
        quat = quat[..., [-1, 0, 1, 2]]  # MuJoCo quat is [w, x, y, z], ours is [x, y, z, w]
        qpos = rearrange(jnp.concat([pos, quat], axis=-1), "w d qpos -> w (d qpos)")
        qvel = rearrange(jnp.concat([vel, ang_vel], axis=-1), "w d qvel -> w (d qvel)")
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        return batched_mjx_forward(mjx_model, mjx_data)


@jax.jit
def contacts(geom_start: int, geom_count: int, data: Data) -> Array:
    """Filter contacts from MuJoCo data."""
    geom1_valid = data.contact.geom1 >= geom_start
    geom1_valid &= data.contact.geom1 < geom_start + geom_count
    geom2_valid = data.contact.geom2 >= geom_start
    geom2_valid &= data.contact.geom2 < geom_start + geom_count
    return data.contact.dist < 0 & (geom1_valid | geom2_valid)


batched_mjx_forward = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))


quat2rpy = jax.jit(
    partial(jnp.vectorize, signature="(4)->(3)")(lambda x: R.from_quat(x).as_euler("xyz"))
)
