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

from crazyflow.control.controller import Control, Controller
from crazyflow.exception import ConfigError
from crazyflow.sim.physics import Physics, identified_dynamics


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
        device: str = "cpu",
        xml_path: Path | None = None,
    ):
        self.physics = physics
        self.control = control
        self.controller = controller
        self.device = jax.devices(device)[0]
        self.freq = freq
        self._dt = jnp.array(1 / freq, device=self.device)
        # pycffirmware uses global states which make it impossible to simulate multiple drones at
        # the same time. We raise if the user tries a combination of pycffirmware and drones > 1.
        if controller == Controller.pycffirmware and n_worlds != n_drones != 1:
            raise ConfigError("pycffirmware controller is only supported for single drone sims")
        if physics == Physics.sys_id and control != Control.attitude:
            raise ConfigError("sys_id physics requires attitude control")
        # Allocate internal states and controls for analytical and sys_id physics.
        self.n_worlds = n_worlds
        self.n_drones = n_drones
        self.states = {
            "pos": jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            "quat": jnp.zeros((n_worlds, n_drones, 4), device=self.device),
            "vel": jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            "ang_vel": jnp.zeros((n_worlds, n_drones, 3), device=self.device),
        }
        self.states["quat"] = self.states["quat"].at[:, :, 3].set(1.0)
        # Allocate internal control variables and physics parameters.
        self._controls = {
            "state": jnp.zeros((n_worlds, n_drones, 13), device=self.device),
            "attitude": jnp.zeros((n_worlds, n_drones, 4), device=self.device),
            "thrust": jnp.zeros((n_worlds, n_drones, 4), device=self.device),
        }
        self._params = {"mass": jnp.ones((n_worlds, n_drones, 1), device=self.device) * 0.025}
        self._step = 0
        # Initialize MuJoCo world and data
        self._xml_path = xml_path or self.default_path
        self._spec, self._mj_model, self._mj_data, self._mjx_model, self._mjx_data = self.setup()
        self.viewer: MujocoRenderer | None = None

    def setup(self) -> tuple[Any, Any, Any, Model, Data]:
        assert self._xml_path.exists(), f"Model file {self._xml_path} does not exist"
        spec = mujoco.MjSpec.from_file(str(self._xml_path))
        mj_model = spec.compile()
        mj_data = mujoco.MjData(mj_model)
        mjx_model = mjx.put_model(mj_model, device=self.device)
        mjx_data = mjx.put_data(mj_model, mj_data, device=self.device)
        mjx_data = jax.vmap(lambda _: mjx_data)(jnp.arange(self.n_worlds))
        return None, mj_model, mj_data, mjx_model, mjx_data

    def reset(self):
        self._step = 0
        raise NotImplementedError

    def step(self):
        """Simulate all drones in all worlds for one time step."""
        self._step += 1
        match self.physics:
            case Physics.mujoco:
                raise NotImplementedError
            case Physics.analytical:
                raise NotImplementedError
            case Physics.sys_id:
                self._step_sys_id()
            case _:
                raise ValueError(f"Physics mode {self.physics} not implemented")

    def attitude_control(self, cmd: Array):
        assert cmd.shape == (self.n_worlds, self.n_drones, 4), "Command shape mismatch"
        assert self.control == Control.attitude, "Attitude control is not enabled by the sim config"
        if self.physics == Physics.sys_id:  # No controller is used for sys_id physics
            self._controls["attitude"] = cmd
            return
        raise NotImplementedError

    def state_control(self, cmd: Array):
        assert cmd.shape == (self.n_worlds, self.n_drones, 13), "Command shape mismatch"
        assert self.control == Control.state, "State control is not enabled by the sim config"
        assert self.physics != Physics.sys_id, "State control is not supported for sys_id physics"
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

    def _step_sys_id(self):
        pos, quat, vel, ang_vel = batched_identified_dynamics(
            self._controls["attitude"], *self.states.values(), self._dt
        )
        self.states["pos"] = pos
        self.states["quat"] = quat
        self.states["vel"] = vel
        self.states["ang_vel"] = ang_vel
        quat = quat[..., [-1, 0, 1, 2]]  # MuJoCo quat is [w, x, y, z], ours is [x, y, z, w]
        qpos = rearrange(jnp.concat([pos, quat], axis=-1), "w d qpos -> w (d qpos)")
        qvel = rearrange(jnp.concat([vel, ang_vel], axis=-1), "w d qvel -> w (d qvel)")
        assert self._mjx_data.qpos.shape == qpos.shape, f"Shape mismatch: {qpos.shape}"
        assert self._mjx_data.qvel.shape == qvel.shape
        self._mjx_data = self._mjx_data.replace(qpos=qpos, qvel=qvel)
        self._mjx_data = batched_mjx_forward(self._mjx_model, self._mjx_data)


in_axes1 = (0, 0, 0, 0, 0, None)
in_axes2 = (1, 1, 1, 1, 1, None)
batched_identified_dynamics = jax.jit(
    jax.vmap(jax.vmap(identified_dynamics, in_axes1, 0), in_axes2, 1)
)

batched_mjx_forward = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))
