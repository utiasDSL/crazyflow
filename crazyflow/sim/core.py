import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.control.controller import Control, Controller
from crazyflow.sim.physics import Physics, identified_dynamics


class Sim:
    def __init__(
        self,
        n_worlds: int,
        n_drones: int,
        physics: Physics = Physics.default,
        control: Control = Control.default,
        controller: Controller = Controller.default,
        freq: int = 500,
        device: str = "cpu",
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
            raise ValueError("pycffirmware controller is only supported for single drone sims")
        # Allocate internal states and controls for analytical and sys_id physics.
        self.n_worlds = n_worlds
        self.n_drones = n_drones
        self._states = {
            "pos": jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            "quat": jnp.zeros((n_worlds, n_drones, 4), device=self.device),
            "vel": jnp.zeros((n_worlds, n_drones, 3), device=self.device),
            "ang_vel": jnp.zeros((n_worlds, n_drones, 3), device=self.device),
        }
        self._states["quat"] = self._states["quat"].at[:, :, 3].set(1.0)
        # Allocate internal control variables and physics parameters.
        self._controls = {
            "state": jnp.zeros((n_worlds, n_drones, 13), device=self.device),
            "attitude": jnp.zeros((n_worlds, n_drones, 4), device=self.device),
            "thrust": jnp.zeros((n_worlds, n_drones, 4), device=self.device),
        }
        self._params = {"mass": jnp.ones((n_worlds, n_drones, 1), device=self.device) * 0.025}

    def setup(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self):
        """Simulate all drones in all worlds for one time step."""
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
        if self.physics == Physics.sys_id:  # No controller is used for sys_id physics
            self._controls["attitude"] = cmd
            return
        raise NotImplementedError

    def state_control(self, cmd: Array):
        raise NotImplementedError

    def thrust_control(self, cmd: Array):
        raise NotImplementedError

    def _step_sys_id(self):
        pos, quat, vel, ang_vel = batched_identified_dynamics(
            self._controls["attitude"], *self._states.values(), self._dt
        )
        self._states["pos"] = pos
        self._states["quat"] = quat
        self._states["vel"] = vel
        self._states["ang_vel"] = ang_vel


in_axes1 = (0, 0, 0, 0, 0, None)
in_axes2 = (1, 1, 1, 1, 1, None)
batched_identified_dynamics = jax.jit(
    jax.vmap(jax.vmap(identified_dynamics, in_axes1, 0), in_axes2, 1)
)
