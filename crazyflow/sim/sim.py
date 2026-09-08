from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from collections import OrderedDict
from functools import partial, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from jax import Array, Device

import crazyflow.sim.functional as F
from crazyflow.control import Control
from crazyflow.control.mellinger import (
    control_attitude2force_torque,
    control_commit_attitude,
    control_force_torque2rotor_vel,
    control_state2attitude,
)
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.drones import load_params as load_drone_params
from crazyflow.dynamics import Dynamics
from crazyflow.dynamics.first_principles import sim_dynamics as first_principles_dynamics
from crazyflow.dynamics.so_rpy import sim_dynamics as so_rpy_dynamics
from crazyflow.dynamics.so_rpy_rotor import sim_dynamics as so_rpy_rotor_dynamics
from crazyflow.dynamics.so_rpy_rotor_drag import sim_dynamics as so_rpy_rotor_drag_dynamics
from crazyflow.exception import ConfigError, NotInitializedError
from crazyflow.sim.data import SimControls, SimCore, SimData, SimParams, SimState, SimStateDeriv
from crazyflow.sim.integration import Integrator, euler, rk4, symplectic_euler
from crazyflow.sim.pipeline import append_fn
from crazyflow.sim.sharding import placement
from crazyflow.utils import grid_2d, pytree_replace, world_mask

if TYPE_CHECKING:
    from jax.sharding import Mesh
    from mujoco.mjx import Data, Model
    from numpy.typing import NDArray

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
    """Crazyflow simulation.

    Used both through its object-oriented methods ([step][crazyflow.sim.Sim.step],
    [reset][crazyflow.sim.Sim.reset], the ``*_control`` setters) and as the builder for the
    functional API in [crazyflow.sim.functional][], which operates on the ``sim.data`` and pipelines
    constructed here.

    The simulation is always batched. Every quantity in ``sim.data`` has a leading
    ``(n_worlds, n_drones, ...)`` shape, even for a single world and drone. ``n_worlds`` indexes
    parallel copies of the scene and ``n_drones`` the drones within each world.
    """

    def __init__(
        self,
        n_worlds: int = 1,
        n_drones: int = 1,
        drone: str = "cf21B_500",
        dynamics: Dynamics = Dynamics.default,
        control: Control = Control.default,
        integrator: Integrator = Integrator.default,
        freq: int = 500,
        state_freq: int = 100,
        attitude_freq: int = 500,
        force_torque_freq: int = 500,
        device: str = "cpu",
        xml_path: Path | None = None,
        rng_key: int = 0,
        fused_mjx_model: bool = False,
    ):
        """Build the scene and the step and reset pipelines, and allocate the batched sim data.

        Args:
            n_worlds: Number of parallel worlds to simulate.
            n_drones: Number of drones per world.
            drone: Name of the drone.
            dynamics: Dynamics used to advance the drone state.
            control: Control interface exposed to the user.
            integrator: Integration scheme for the dynamics.
            freq: Dynamics step frequency in Hz.
            state_freq: Frequency in Hz at which the state controller runs.
            attitude_freq: Frequency in Hz at which the attitude controller runs.
            force_torque_freq: Frequency in Hz at which the force/torque controller runs.
            device: Device to place the simulation data on (e.g. ``"cpu"`` or ``"gpu"``).
            xml_path: Path to a custom scene XML. Defaults to ``crazyflow/scene.xml``.
            rng_key: Seed for the JAX rng key.
            fused_mjx_model: If True, use the ``drone_fused`` body whose visual geometry is fused
                into a single mesh. This shrinks the MJX model and reduces its memory footprint at
                the cost of visual detail.
        """
        assert Dynamics(dynamics) in Dynamics, f"Dynamics mode {dynamics} not implemented"
        assert Control(control) in Control, f"Control mode {control} not implemented"
        if dynamics != Dynamics.first_principles:
            if control in (Control.force_torque, Control.rotor_vel):
                raise ConfigError(f"Control mode {control} requires first principles dynamics")
        if freq > 10_000 and not jax.config.jax_enable_x64:
            raise ConfigError("High frequency simulations require double precision mode")
        self.dynamics = dynamics
        self.control = control
        self.drone = drone
        self.integrator = integrator
        self.device = jax.devices(device)[0]
        self.n_worlds = n_worlds
        self.n_drones = n_drones
        self.freq = freq
        self.max_visual_geom = 1000

        # Initialize MuJoCo world and data
        self.fused_mjx_model = fused_mjx_model
        self._xml_path = xml_path or Path(__file__).parents[1] / "scene.xml"
        self.drone_path = Path(__file__).parents[1] / f"drones/{drone}.xml"
        self.spec = self.build_mjx_spec()
        self.mj_model, self.mj_data, self.mjx_model, self.mjx_data = self.build_mjx_model(self.spec)
        self.viewer: MujocoRenderer | None = None

        self.data = self.init_data(state_freq, attitude_freq, force_torque_freq, rng_key)
        self.default_data: SimData = self.build_default_data()

        # Build the simulation pipeline and overwrite the default _step implementation with it
        self.reset_pipeline: OrderedDict[
            str, Callable[[SimData, SimData, Array[bool] | None], SimData]
        ] = OrderedDict()
        append_fn(self.reset_pipeline, reset)

        self.step_pipeline: OrderedDict[str, Callable[[SimData], SimData]] = OrderedDict()
        # The ``select_xxx_fn`` methods return functions, not the results of calling those
        # functions. They act as factories that produce building blocks for the construction of our
        # simulation pipeline.
        for name, fn in build_control_fns(self.control, self.dynamics):
            append_fn(self.step_pipeline, fn, name=name)
        integrate_fn = select_integrate_fn(self.integrator, select_dynamics_fn(self.dynamics))
        append_fn(self.step_pipeline, integrate_fn, name="integration")
        # Keep the rotor state (RPM or thrust, see ``rotor_vel_limits``) within its physical limits
        lower, upper = rotor_vel_limits(self.dynamics, self.drone)
        clip_fn = partial(clip_rotor_vel, lower=lower, upper=upper)
        append_fn(self.step_pipeline, clip_fn, name="clip_rotor_vel")
        append_fn(self.step_pipeline, increment_steps)
        # We never drop below -0.001 (drones can't pass through the floor). We use -0.001 to
        # enable checks for negative z sign
        append_fn(self.step_pipeline, clip_floor_pos)

        self._reset = self.build_reset_fn()
        self._step = self.build_step_fn()

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
        self.data = F.state_control(self.data, controls)

    def attitude_control(self, controls: Array):
        """Set the desired attitude for all drones in all worlds."""
        self.data = F.attitude_control(self.data, controls)

    def force_torque_control(self, controls: Array):
        """Set the desired force and torque for all drones in all worlds."""
        self.data = F.force_torque_control(self.data, controls)

    def rotor_vel_control(self, controls: Array):
        """Set the desired rotor velocities for all drones in all worlds."""
        self.data = F.rotor_vel_control(self.data, controls)

    @requires_mujoco_sync
    def render(
        self,
        mode: str | None = "human",
        world: int = 0,
        camera: int | str = -1,
        cam_config: dict | None = None,
        width: int = 1920,
        height: int = 1080,
    ) -> NDArray | None:
        """Render one world of the simulation.

        Args:
            mode: Render mode. ``"human"`` opens an interactive MuJoCo viewer window,
                ``"rgb_array"``, ``"depth_array"`` and ``"rgbd_tuple"`` return offscreen renderings.
            world: Index of the world to render.
            camera: Camera name or id. -1 is the global free camera.
            cam_config: Camera configuration for the MuJoCo viewer.
            width: Image width for offscreen rendering.
            height: Image height for offscreen rendering.

        Returns:
            The rendered image(s) for the offscreen modes, None for ``"human"``.
        """
        if self.viewer is None:
            if isinstance(camera, str):
                cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
                assert cam_id > -1, f"Camera name '{camera}' not found in the model."
            elif isinstance(camera, int):
                cam_id = camera
                assert cam_id >= -1, f"camera id must be >=-1, was {cam_id}"
            else:
                raise TypeError("camera argument must be integer or string")
            self.mj_model.vis.global_.offwidth = width
            self.mj_model.vis.global_.offheight = height
            self.viewer = MujocoRenderer(
                self.mj_model,
                self.mj_data,
                max_geom=self.max_visual_geom,
                default_cam_config=cam_config,
                height=height,
                width=width,
                camera_id=cam_id,
            )
            # In human mode, cam_id is set to -1, so we force it to the desired value
            if mode == "human" and cam_id > -1:
                # Render one frame to force mj to create the viewer
                self.viewer.render(mode)
                self.viewer.viewer.cam.fixedcamid = cam_id
                self.viewer.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        self.mj_data.qpos[:] = self.mjx_data.qpos[world, :]
        self.mj_data.mocap_pos[:] = self.mjx_data.mocap_pos[world, :]
        self.mj_data.mocap_quat[:] = self.mjx_data.mocap_quat[world, :]
        # mj_forward raises on contacts between two static bodies (e.g. drones welded to the world).
        # We only need poses, cameras and lights for rendering
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        mujoco.mj_comPos(self.mj_model, self.mj_data)
        mujoco.mj_camlight(self.mj_model, self.mj_data)
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
        """Build the MuJoCo mjx_model specification for the simulation."""
        assert self._xml_path.exists(), f"Model file {self._xml_path} does not exist"
        spec = mujoco.MjSpec.from_file(str(self._xml_path))
        spec.option.timestep = 1 / self.freq
        spec.copy_during_attach = True
        drone_spec = mujoco.MjSpec.from_file(str(self.drone_path))
        frame = spec.worldbody.add_frame(name="world")
        name = "drone_fused" if self.fused_mjx_model else "drone"
        if (drone_body := drone_spec.body(name)) is None:
            raise ValueError("Drone body not found in drone spec")
        drone_body.mocap = True
        frame.attach_body(drone_body, "", ":0")  # Attach a single drone, then stamp out the swarm.
        spec = self._replicate_drone(spec, f"{name}:0", drone_spec.meshdir)
        # Mocap bodies avoid the nv^2 cost of qM/qLD/efc_J. A single dummy slide joint keeps nv=1 so
        # mjx.kinematics doesn't error on a zero-DOF mjx_model. Added after _replicate_drone because
        # to_xml() drops this geom-less body's mass/inertia (the result would fail to recompile).
        dummy = spec.worldbody.add_body()
        dummy.name = "_dummy"
        dummy.mass = 1e-6
        dummy.inertia = jnp.full(3, 1e-9)
        dummy_joint = dummy.add_joint()
        dummy_joint.name = "_dummy_joint"
        dummy_joint.type = mujoco.mjtJoint.mjJNT_SLIDE
        return spec

    def _replicate_drone(
        self, spec: mujoco.MjSpec, drone0_body: str, drone_meshdir: str
    ) -> mujoco.MjSpec:
        """Stamp the single attached drone body into ``n_drones`` instances by cloning its XML.

        Avoids the O(n_drones^2) cost and per-instance mesh copies of repeated ``attach_body``.
        Only element ``name``s are suffixed ``:0`` -> ``:i``, so meshes and static materials stay
        shared. Dynamic materials are duplicated per drone.
        """
        # Name prefixes of the drone's "dynamic" materials which get per-copy materials
        _DYN_MATERIAL = ("led",)
        # Absolute meshdir so the serialized mesh file paths resolve when re-parsed from a string.
        spec.meshdir = str((self.drone_path.parent / drone_meshdir).resolve())
        root = ET.fromstring(spec.to_xml())
        asset = root.find("asset")
        body0 = root.find(f".//body[@name='{drone0_body}']")
        frame = next(p for p in root.iter() if body0 in p)  # the body lives under the attach frame
        dynamic = [m for m in asset.findall("material") if m.get("name").startswith(_DYN_MATERIAL)]
        for i in range(1, self.n_drones):
            body = copy.deepcopy(body0)
            for el in body.iter():  # suffix unique names + dynamic-material refs. keep shared refs
                if (name := el.get("name", "")).endswith(":0"):
                    el.set("name", f"{name[:-2]}:{i}")
                if (mat := el.get("material", "")).startswith(_DYN_MATERIAL):
                    el.set("material", f"{mat[:-2]}:{i}")
            frame.append(body)
            for material in dynamic:
                clone = copy.deepcopy(material)
                clone.set("name", f"{material.get('name')[:-2]}:{i}")
                asset.append(clone)
        new_spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
        new_spec.copy_during_attach = True  # write-only, re-apply after from_string() drops it
        return new_spec

    def build_mjx_model(self, spec: mujoco.MjSpec) -> tuple[Any, Any, Model, Data]:
        """Build the MuJoCo model and data structures for the simulation."""
        mj_model = spec.compile()
        self._unweld_drones(mj_model)
        mj_data = mujoco.MjData(mj_model)
        mjx_model = mjx.put_model(mj_model, device=self.device)
        mjx_data = mjx.put_data(mj_model, mj_data, device=self.device)
        mjx_data = jax.vmap(lambda _: mjx_data)(jnp.arange(self.n_worlds))
        return mj_model, mj_data, mjx_model, mjx_data

    def _unweld_drones(self, mj_model: mujoco.MjModel):
        """Relabel each drone as its own weld root so collisions are detected natively.

        Drones are mocap bodies, so MuJoCo welds them all to the world (``weldid == 0``). MJX skips
        collisions within a weld tree, leading to no drone contact being generated. A distinct weld
        id per drone re-enables drone-drone/-floor/-object contacts without explicit ``<pair>``s.
        """
        base = "drone_fused" if self.fused_mjx_model else "drone"
        for i in range(self.n_drones):
            body_id = mj_model.body(f"{base}:{i}").id
            mj_model.body_weldid[body_id] = body_id  # Each drone becomes its own weld root

    def build_step_fn(self) -> Callable[[SimData, int], SimData]:
        """Setup the chain of functions that are called in Sim.step().

        We know all the functions that are called in succession since the simulation is configured
        at initialization time. Instead of branching through options at runtime, we construct a step
        function at initialization that selects the correct functions based on the settings.

        Note:
            This function both changes the underlying implementation of Sim.step() in-place to the
            current pipeline and returns the function for pure functional style programming.

        Warning:
            If any settings change, the pipeline of functions needs to be reconstructed.

        Returns:
            The pure JAX function that steps through the simulation. It takes the current SimData
            and the number of steps to simulate, and returns the updated SimData.
        """
        # Snapshot the pipeline functions into a tuple. jax.jit traces lazily on the first call,
        # so without a snapshot, modifying the pipeline between building and the first step would
        # silently get compiled in.
        pipeline = tuple(self.step_pipeline.values())

        # None is required by jax.lax.scan to unpack the tuple returned by single_step.
        def single_step(data: SimData, _: None) -> tuple[SimData, None]:
            for fn in pipeline:
                data = fn(data)
            return data, None

        # ``scan`` allows us control over loop unrolling for single steps from a single WhileOp to
        # complete unrolling, reducing either compilation times or fusing the loops to give XLA
        # maximum freedom to reorder operations and jointly optimize the pipeline. This is
        # especially relevant for the common use case of running multiple sim steps in an outer
        # loop, e.g. in gym environments.
        # Having n_steps as a static argument is fine, since patterns with n_steps > 1 will almost
        # always use the same n_steps value for successive calls.
        @partial(jax.jit, static_argnames="n_steps")
        def step(data: SimData, n_steps: int = 1) -> SimData:
            data, _ = jax.lax.scan(single_step, data, length=n_steps, unroll=1)
            data = data.replace(core=data.core.replace(mjx_synced=False))  # Flag mjx data as stale
            return data

        self._step = step
        return step

    def build_reset_fn(self) -> Callable[[SimData, SimData, Array | None], SimData]:
        """Build the reset function for the current simulation configuration.

        Note:
            This function both changes the underlying implementation of Sim.reset() in-place to the
            current pipeline and returns the function for pure functional style programming.

        Returns:
            The pure JAX function that resets simulation data. It takes the current SimData, default
            SimData, and an optional mask for worlds to reset, returning the updated SimData.
        """
        # Snapshot the pipeline functions into a tuple. jax.jit traces lazily on the first call,
        # so without a snapshot, modifying the pipeline between building and the first reset would
        # silently get compiled in.
        pipeline = tuple(self.reset_pipeline.values())

        @jax.jit
        def reset(data: SimData, default_data: SimData, mask: Array | None = None) -> SimData:
            for fn in pipeline:
                data = fn(data, default_data, mask)
            data = data.replace(core=data.core.replace(mjx_synced=False))  # Flag mjx data as stale
            return data

        self._reset = reset
        return reset

    def build_data(self) -> SimData:
        """Build the simulation data for the current configuration.

        Note:
            This function re-initializes the simulation data according to the current configuration.
            It also returns the constructed data for use with pure functions.

        Returns:
            The simulation data as a single PyTree that can be passed to the pure simulation
            functions for stepping and resetting.
        """
        state_freq = 0 if (s := self.data.controls.state) is None else s.freq
        attitude_freq = 0 if (a := self.data.controls.attitude) is None else a.freq
        force_torque_freq = 0 if (ft := self.data.controls.force_torque) is None else ft.freq
        self.data = self.init_data(
            state_freq, attitude_freq, force_torque_freq, self.data.core.rng_key
        )
        return self.data

    def shard(self, mesh: Mesh) -> SimData:
        """Distribute the data and default data over a mesh along the world axis.

        Args:
            mesh: Mesh to distribute the worlds over, as built by
                [world_mesh][crazyflow.sim.sharding.world_mesh].

        Returns:
            The placed simulation data.
        """
        self.data = jax.device_put(self.data, placement(self.data, mesh))
        self.default_data = jax.device_put(self.default_data, placement(self.default_data, mesh))
        return self.data

    def build_default_data(self) -> SimData:
        """Initialize the default data for the simulation.

        Note:
            This function initializes the default data used as a reference in the reset function to
            reset the simulation to. It also returns the constructed data for use with pure
            functions.

        Returns:
            The default simulation data used as a reference in the reset function to reset the
            simulation to.
        """
        self.default_data = self.data.replace()
        return self.default_data

    def build_mjx(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.mj_model, self.mj_data, self.mjx_model, self.mjx_data = self.build_mjx_model(self.spec)

    def init_data(
        self, state_freq: int, attitude_freq: int, force_torque_freq: int, rng_key: Array
    ) -> SimData:
        """Initialize the simulation data."""
        drone_name = "drone_fused" if self.fused_mjx_model else "drone"
        drone_mocap_ids = [
            self.mj_model.body(f"{drone_name}:{i}").mocapid.item() for i in range(self.n_drones)
        ]
        N, D = self.n_worlds, self.n_drones
        data = SimData(
            states=SimState.create(N, D, self.device),
            states_deriv=SimStateDeriv.create(N, D, self.device),
            controls=SimControls.create(
                N,
                D,
                self.control,
                self.drone,
                state_freq,
                attitude_freq,
                force_torque_freq,
                self.device,
            ),
            params=SimParams.create(self.dynamics, self.drone, self.device),
            core=SimCore.create(self.freq, N, D, drone_mocap_ids, rng_key, self.device),
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
        return F.controllable(self.data)

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
    control: Control, dynamics: Dynamics
) -> tuple[tuple[str, Callable[[SimData], SimData]], ...]:
    """Select the named control stages for the given control mode.

    Note:
        Returns ``(name, fn)`` pairs, called in succession in the simulation pipeline. The names are
        the stable pipeline stage identifiers used to insert, replace, or remove stages.
    """
    state = ("state_controller", control_state2attitude)
    attitude = ("attitude_controller", control_attitude2force_torque)
    force_torque = ("force_torque_controller", control_force_torque2rotor_vel)
    commit_attitude = ("commit_attitude", control_commit_attitude)
    match control:
        case Control.state:
            stages = (state, attitude)
            if dynamics == Dynamics.first_principles:
                stages = stages + (force_torque,)
        case Control.attitude:
            if dynamics == Dynamics.first_principles:
                stages = (attitude, force_torque)
            elif dynamics in (Dynamics.so_rpy, Dynamics.so_rpy_rotor, Dynamics.so_rpy_rotor_drag):
                stages = (commit_attitude,)
            else:
                raise NotImplementedError(f"Control mode {control} not implemented for {dynamics}")
        case Control.force_torque:
            stages = (force_torque,)
        case Control.rotor_vel:
            stages = ()
        case _:
            raise NotImplementedError(f"Control mode {control} not implemented")

    return stages


def select_dynamics_fn(dynamics: Dynamics) -> Callable[[SimData], SimData]:
    """Select the dynamics function for the given dynamics mode."""
    match dynamics:
        case Dynamics.first_principles:
            return first_principles_dynamics
        case Dynamics.so_rpy:
            return so_rpy_dynamics
        case Dynamics.so_rpy_rotor:
            return so_rpy_rotor_dynamics
        case Dynamics.so_rpy_rotor_drag:
            return so_rpy_rotor_drag_dynamics
        case _:
            raise NotImplementedError(f"Dynamics mode {dynamics} not implemented")


def select_integrate_fn(
    integrator: Integrator, dynamics_fn: Callable[[SimData], SimData]
) -> Callable[[SimData], SimData]:
    """Select the integration function for the given dynamics and integrator mode."""
    match integrator:
        case Integrator.euler:
            integrate_fn = euler
        case Integrator.rk4:
            integrate_fn = rk4
        case Integrator.symplectic_euler:
            integrate_fn = symplectic_euler
        case _:
            raise NotImplementedError(f"Integrator {integrator} not implemented")

    return partial(integrate_fn, deriv_fn=dynamics_fn)


def reset(data: SimData, default_data: SimData, mask: Array | None = None) -> SimData:
    """Reset the simulation data to the default data for the worlds specified by the mask.

    Without a mask, the full data is restored. The mask selects along the world axis, so it only
    restores per-world arrays. The rng key is never restored.
    """
    if mask is None:
        return default_data.replace(core=default_data.core.replace(rng_key=data.core.rng_key))
    return pytree_replace(data, default_data, world_mask(data), mask)


def increment_steps(data: SimData) -> SimData:
    """Increment the simulation steps."""
    return data.replace(core=data.core.replace(steps=data.core.steps + 1))


@jax.jit
def contacts(geom_start: int, geom_count: int, data: Data) -> Array:
    """Filter contacts from MuJoCo data."""
    contact = data._impl.contact
    geom1_valid = contact.geom1 >= geom_start
    geom1_valid &= contact.geom1 < geom_start + geom_count
    geom2_valid = contact.geom2 >= geom_start
    geom2_valid &= contact.geom2 < geom_start + geom_count
    return (contact.dist < 0) & (geom1_valid | geom2_valid)


@jax.jit
def sync_sim2mjx(data: SimData, mjx_data: Data, mjx_model: Model) -> tuple[SimData, Data]:
    """Synchronize the simulation data with the MuJoCo mjx_model."""
    pos, quat = data.states.pos, data.states.quat
    quat_mjx = jnp.roll(quat, 1, axis=-1)  # MuJoCo quat is [w, x, y, z], ours is [x, y, z, w]
    ids = data.core.drone_mocap_ids
    mocap_pos = mjx_data.mocap_pos.at[:, ids, :].set(pos)
    mocap_quat = mjx_data.mocap_quat.at[:, ids, :].set(quat_mjx)
    mjx_data = mjx_data.replace(mocap_pos=mocap_pos, mocap_quat=mocap_quat)
    mjx_data = jax.vmap(mjx.kinematics, in_axes=(None, 0))(mjx_model, mjx_data)
    # Required for rendering w. ray casting
    mjx_data = jax.vmap(mjx.camlight, in_axes=(None, 0))(mjx_model, mjx_data)
    mjx_data = jax.vmap(mjx.collision, in_axes=(None, 0))(mjx_model, mjx_data)
    data = data.replace(core=data.core.replace(mjx_synced=True))
    return data, mjx_data


def clip_floor_pos(data: SimData) -> SimData:
    """Clip the position of the drone to the floor."""
    clip = data.states.pos[..., 2] < -0.001
    clip_pos = data.states.pos.at[..., 2].set(jnp.where(clip, -0.001, data.states.pos[..., 2]))
    clip_vel = data.states.vel.at[..., :3].set(
        jnp.where(clip[..., None], 0, data.states.vel[..., :3])
    )
    return data.replace(states=data.states.replace(pos=clip_pos, vel=clip_vel))


def rotor_vel_limits(dynamics: Dynamics, drone: str) -> tuple[Array | float, Array | float]:
    """Limits of ``rotor_vel`` in RPM (first principles) or collective thrust in N (others)."""
    params = load_drone_params(drone)
    thrust_min, thrust_max = params["thrust_min"], params["thrust_max"]
    if dynamics == Dynamics.first_principles:
        rpm = motor_force2rotor_vel(jnp.asarray([thrust_min, thrust_max]), params["rpm2thrust"])
        return rpm[0], rpm[1]
    return 4 * thrust_min, 4 * thrust_max


def clip_rotor_vel(data: SimData, lower: Array | float, upper: Array | float) -> SimData:
    """Clip ``rotor_vel`` to ``[lower, upper]``."""
    rotor_vel = jnp.clip(data.states.rotor_vel, lower, upper)
    return data.replace(states=data.states.replace(rotor_vel=rotor_vel))


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
