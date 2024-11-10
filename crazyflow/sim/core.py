import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.sim.physics import Physics, identified_dynamics


class Sim:
    def __init__(
        self,
        n_worlds: int,
        n_drones: int,
        physics: Physics = Physics.DEFAULT,
        freq: int = 500,
        device: str = "cpu",
    ):
        self.physics = physics
        self.device = jax.devices(device)[0]
        self.freq = freq
        self._dt = jnp.ones((n_worlds, n_drones, 1), dtype=jnp.float32, device=self.device) / freq
        # self._dt = 1 / freq

        self._n_worlds = n_worlds
        self._n_drones = n_drones
        self._states = {
            "pos": jnp.zeros((n_worlds, n_drones, 3), dtype=jnp.float32, device=self.device),
            "quat": jnp.zeros((n_worlds, n_drones, 4), dtype=jnp.float32, device=self.device),
            "vel": jnp.zeros((n_worlds, n_drones, 3), dtype=jnp.float32, device=self.device),
            "ang_vel": jnp.zeros((n_worlds, n_drones, 3), dtype=jnp.float32, device=self.device),
        }
        self._controls = {
            "state": jnp.zeros((n_worlds, n_drones, 13), dtype=jnp.float32, device=self.device),
            "attitude": jnp.zeros((n_worlds, n_drones, 4), dtype=jnp.float32, device=self.device),
            "thrust": jnp.zeros((n_worlds, n_drones, 4), dtype=jnp.float32, device=self.device),
        }
        self._params = {
            "mass": jnp.ones((n_worlds, n_drones, 1), dtype=jnp.float32, device=self.device) * 0.025
        }

    def setup(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self):
        """Simulate all drones in all worlds for one time step."""
        self._step(self.physics)

    def attitude_control(self, cmd: Array):
        raise NotImplementedError

    def state_control(self, cmd: Array):
        raise NotImplementedError

    def thrust_control(self, cmd: Array):
        raise NotImplementedError

    def _step(self, physics: Physics):
        match physics:
            case Physics.MUJOCO:
                raise NotImplementedError
            case Physics.ANALYTICAL:
                raise NotImplementedError
            case Physics.SYS_ID:
                pos, quat, vel, ang_vel = batched_identified_dynamics(
                    self._controls["attitude"], *self._states.values(), self._dt
                )
                self._states["pos"] = pos
                self._states["quat"] = quat
                self._states["vel"] = vel
                self._states["ang_vel"] = ang_vel
            case _:
                raise ValueError(f"Physics mode {physics} not implemented")


in_axes1 = (0, 0, 0, 0, 0, None)
in_axes2 = (1, 1, 1, 1, 1, None)
batched_identified_dynamics = jax.vmap(jax.vmap(identified_dynamics, in_axes1, 0), in_axes2, 1)
