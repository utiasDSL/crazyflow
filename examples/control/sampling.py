"""Track a Lissajous curve with a sampling-based MPC while avoiding a grid of obstacles.

The controller rolls out thousands of candidate control sequences in parallel using identified
dynamics, then applies the first action from a cost-weighted update of the best samples. It uses a
GPU when one is available and lowers the sample count on CPU.

WARNING: This is an advanced example meant for advanced Crazyflow users.
"""

import os
from collections import deque
from functools import partial
from typing import Callable

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.lax import scan

from crazyflow.control import Control
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.dynamics import load_params
from crazyflow.sim import Dynamics, Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.visualize import draw_capsule, draw_line

try:
    # If available, use the GPU for the controller to be able to run more samples
    jax.devices("gpu")
    DEVICE_CONTROLLER = "gpu"
except RuntimeError:
    DEVICE_CONTROLLER = "cpu"

# Simulation configuration
DRONE = "cf21B_500"
DURATION = 10.0
FPS = 60
RENDER = True

# Controller configuration
CTRL_FREQ = 50
T = 1.0  # Prediction horizon in seconds
N = 25  # Prediction steps
N_SAMPLES = 500_000 if DEVICE_CONTROLLER == "gpu" else 2_000
NOISE_SIGMA = jnp.array([0.10, 0.10, 0.0, 0.08], dtype=jnp.float32)
ELITE_FRACTION = 0.01  # Fraction of samples in (0,1] to use for the mean update
MAX_CMD_ANGLE = np.deg2rad(60.0)

# Lissajous trajectory configuration
TRAJ_CENTER = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
TRAJ_SIZE = jnp.array([1.0, 0.75, 0.0], dtype=jnp.float32)
TRAJ_PERIOD = 6.0

# Visualization configuration
HISTORY_DURATION = 1.0
N_RENDER_SAMPLES = 32  # How many rollouts are visualized

# Pole-grid configuration
OBSTACLE_GRID = (6, 5)
OBSTACLE_SPACING = (0.5, 0.5)
OBSTACLE_CENTER = (0.0, 0.0)
OBSTACLE_RADIUS = 0.055
OBSTACLE_HEIGHT = 1.6
DRONE_RADIUS = 0.12
OBSTACLE_MARGIN = 0.02


def trajectory(t: Array | float) -> dict[str, Array]:
    """Return the position, velocity, and yaw reference at time ``t``."""
    omega = 2.0 * jnp.pi / TRAJ_PERIOD
    phi = omega * jnp.asarray(t)
    zero = jnp.zeros_like(phi)
    pos = TRAJ_CENTER + TRAJ_SIZE * jnp.stack([jnp.sin(phi), jnp.sin(2.0 * phi), zero], axis=-1)
    vel = TRAJ_SIZE * omega * jnp.stack([jnp.cos(phi), 2.0 * jnp.cos(2.0 * phi), zero], axis=-1)
    return {"pos": pos, "vel": vel, "yaw": zero}


def obstacle_grid() -> np.ndarray:
    """Return pole base positions for a centered rectangular grid."""
    nx, ny = OBSTACLE_GRID
    x = (np.arange(nx) - (nx - 1) / 2) * OBSTACLE_SPACING[0] + OBSTACLE_CENTER[0]
    y = (np.arange(ny) - (ny - 1) / 2) * OBSTACLE_SPACING[1] + OBSTACLE_CENTER[1]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.column_stack((xx.ravel(), yy.ravel(), np.zeros(nx * ny))).astype(np.float32)


def step_sim(
    data: SimData,
    inputs: tuple[Array, dict[str, Array]],
    step_fn: Callable[[SimData, int], SimData],
    obstacles: Array,
    hover_thrust: Array,
) -> tuple[SimData, tuple[Array, Array]]:
    """Apply one candidate input per world, advance all rollouts one step and compute the cost."""
    command, reference = inputs
    data = data.replace(
        controls=data.controls.replace(
            attitude=data.controls.attitude.replace(staged_cmd=command[:, None, :])
        )
    )
    next_data = step_fn(data, 1)
    pos = next_data.states.pos[:, 0]
    vel = next_data.states.vel[:, 0]
    cmd = next_data.controls.attitude.staged_cmd[:, 0]

    # Tracking cost
    pos_error = jnp.linalg.norm(pos - reference["pos"], axis=-1)
    vel_error = jnp.linalg.norm(vel - reference["vel"], axis=-1)
    track_cost = 50.0 * pos_error**2 + vel_error**2

    # Control cost
    tilt_cost = 5.0 * jnp.linalg.norm(cmd[:, :2], axis=-1) ** 2
    thrust_cost = 5.0 * (cmd[:, 3] - hover_thrust) ** 2
    yaw_cost = 100.0 * (cmd[:, 2] - reference["yaw"]) ** 2
    input_cost = tilt_cost + thrust_cost + yaw_cost

    # Obstacle cost
    obstacle_distance = jnp.linalg.norm(pos[:, None, :2] - obstacles[None, :, :2], axis=-1)
    obstacle_hits = obstacle_distance < OBSTACLE_RADIUS + DRONE_RADIUS + OBSTACLE_MARGIN
    obstacle_cost = 1_000.0 * jnp.sum(obstacle_hits, axis=-1)
    cost = track_cost + input_cost + obstacle_cost

    return next_data, (cost, next_data.states.pos[:, 0])


def rollout_sim(
    obs: dict[str, Array],
    command: Array,
    reference: dict[str, Array],
    rollout_data: SimData,
    step_fn: Callable[[SimData, int], SimData],
    obstacles: Array,
    hover_thrust: Array,
) -> tuple[Array, Array]:
    """Roll out control sequences from the current state (obs) and compute their costs."""
    states = rollout_data.states.replace(
        pos=rollout_data.states.pos.at[...].set(obs["pos"]),
        quat=rollout_data.states.quat.at[...].set(obs["quat"]),
        vel=rollout_data.states.vel.at[...].set(obs["vel"]),
        ang_vel=rollout_data.states.ang_vel.at[...].set(obs["ang_vel"]),
        # The reduced dynamics store collective thrust in its rotor_vel state.
        rotor_vel=rollout_data.states.rotor_vel.at[...].set(obs["collective_thrust"]),
    )
    data = rollout_data.replace(states=states)
    _, (costs, positions) = scan(
        partial(step_sim, step_fn=step_fn, obstacles=obstacles, hover_thrust=hover_thrust),
        data,
        (command, reference),
    )
    return jnp.sum(costs, axis=0), positions


def update_controller(
    t: Array,
    obs: dict[str, Array],
    key: Array,
    mean_controls: Array,
    rollout_fn: Callable[[dict[str, Array], Array, dict[str, Array]], tuple[Array, Array]],
    hover_cmd: Array,
    action_low: Array,
    action_high: Array,
    noise_sigma: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Update the control trajectory from a cost-weighted mean of the elite samples."""
    key, sample_key = jax.random.split(key)
    noise = jax.random.normal(sample_key, (N_SAMPLES, N, 4)) * noise_sigma
    candidates = jnp.clip(mean_controls[None] + noise, action_low, action_high)
    candidates = candidates.at[0].set(mean_controls)

    prediction_dt = T / N
    times = t + (jnp.arange(N) + 1) * prediction_dt
    references = trajectory(times)
    costs, positions = rollout_fn(obs, candidates.transpose(1, 0, 2), references)

    n_elites = max(1, int(N_SAMPLES * ELITE_FRACTION))
    elite_indices = jnp.argsort(costs)[:n_elites]
    updated_controls = jnp.mean(candidates[elite_indices], axis=0)

    action = updated_controls[0]
    prediction_times = jnp.arange(N) * prediction_dt
    shifted_times = prediction_times + 1.0 / CTRL_FREQ

    def shift_control(control: Array, hover: Array) -> Array:
        return jnp.interp(shifted_times, prediction_times, control, right=hover)

    mean_controls = jax.vmap(shift_control, in_axes=(1, 0), out_axes=1)(updated_controls, hover_cmd)
    best_index = elite_indices[0]
    best_positions = positions[:, best_index]
    render_indices = elite_indices[jnp.linspace(0, n_elites - 1, N_RENDER_SAMPLES, dtype=jnp.int32)]
    sampled_positions = positions[:, render_indices].transpose(1, 0, 2)
    return action, key, mean_controls, best_positions, sampled_positions


def control(
    t: float,
    obs: dict[str, Array],
    key: Array,
    mean_controls: Array,
    controller_fn: Callable,
    controller_device: jax.Device,
) -> tuple[np.ndarray, Array, Array, np.ndarray, np.ndarray]:
    """Move the observation to the rollout device and compute one SMPC update."""
    obs = jax.tree.map(lambda value: jax.device_put(value, controller_device), obs)
    t_device = jax.device_put(jnp.asarray(t, dtype=jnp.float32), controller_device)
    action, key, mean_controls, best_positions, sampled_positions = controller_fn(
        t_device, obs, key, mean_controls
    )
    return (
        np.asarray(action),
        key,
        mean_controls,
        np.asarray(best_positions),
        np.asarray(sampled_positions),
    )


def main() -> None:
    obstacles = obstacle_grid()

    # Set up the main sim
    sim = Sim(n_worlds=1, drone=DRONE, dynamics=Dynamics.first_principles, control=Control.attitude)
    sim.max_visual_geom = 100_000  # To be able to show all rollouts
    sim.reset()
    start_pos = trajectory(0.0)["pos"]
    drone_params = load_params(Dynamics.first_principles, DRONE)
    hover_thrust_value = np.asarray(drone_params["mass"] * 9.81, dtype=np.float32)
    hover_rotor_vel = motor_force2rotor_vel(
        np.full(4, hover_thrust_value / 4.0, dtype=np.float32), drone_params["rpm2thrust"]
    )
    sim.data = sim.data.replace(
        states=sim.data.states.replace(
            pos=sim.data.states.pos.at[0, 0].set(start_pos),
            rotor_vel=sim.data.states.rotor_vel.at[0, 0].set(hover_rotor_vel),
        )
    )

    # Set up the controller
    controller_device = jax.devices(DEVICE_CONTROLLER)[0]
    rollout_freq = int(N / T)
    rollout_simulator = Sim(
        n_worlds=N_SAMPLES,
        device=controller_device.platform,
        drone=DRONE,
        dynamics=Dynamics.so_rpy_rotor_drag,
        control=Control.attitude,
        freq=rollout_freq,
        attitude_freq=rollout_freq,
    )
    rollout_simulator.reset()

    thrust_estimate = hover_thrust_value  # Initial thrust estimate
    thrust_dyn_coef = float(rollout_simulator.data.params.thrust_dyn_coef[0])
    hover_cmd = jax.device_put(
        jnp.array([0.0, 0.0, 0.0, hover_thrust_value], dtype=jnp.float32), controller_device
    )
    action_low = jax.device_put(
        jnp.array([-MAX_CMD_ANGLE, -MAX_CMD_ANGLE, 0.0, 0.0], dtype=jnp.float32), controller_device
    )
    max_thrust_value = np.asarray(4.0 * drone_params["thrust_max"], dtype=np.float32)
    action_high = jax.device_put(
        jnp.array([MAX_CMD_ANGLE, MAX_CMD_ANGLE, 0.0, max_thrust_value], dtype=jnp.float32),
        controller_device,
    )
    noise_sigma = jax.device_put(NOISE_SIGMA, controller_device)

    rollout_fn = partial(
        rollout_sim,
        rollout_data=rollout_simulator.data,
        step_fn=rollout_simulator.build_step_fn(),
        obstacles=jax.device_put(jnp.asarray(obstacles), controller_device),
        hover_thrust=jax.device_put(jnp.asarray(hover_thrust_value), controller_device),
    )
    controller_fn = jax.jit(
        partial(
            update_controller,
            rollout_fn=rollout_fn,
            hover_cmd=hover_cmd,
            action_low=action_low,
            action_high=action_high,
            noise_sigma=noise_sigma,
        )
    )
    mean_controls = jax.device_put(jnp.broadcast_to(hover_cmd, (N, 4)), controller_device)
    key = jax.device_put(jax.random.key(0), controller_device)

    # Plotting helpers
    position_history = deque(maxlen=max(2, round(HISTORY_DURATION * CTRL_FREQ)))
    render_times = np.linspace(0.0, TRAJ_PERIOD, round(TRAJ_PERIOD * CTRL_FREQ) + 1)
    reference = np.asarray(trajectory(render_times)["pos"])

    for step in range(int(DURATION * CTRL_FREQ)):
        t = step / CTRL_FREQ
        obs = {
            "pos": sim.data.states.pos[0, 0],
            "quat": sim.data.states.quat[0, 0],
            "vel": sim.data.states.vel[0, 0],
            "ang_vel": sim.data.states.ang_vel[0, 0],
            # Thrust is not observable and difficult to estimate, so use the thrust dynamics.
            "collective_thrust": thrust_estimate,
        }
        action, key, mean_controls, best_positions, sampled_positions = control(
            t, obs, key, mean_controls, controller_fn, controller_device
        )
        thrust_estimate += thrust_dyn_coef * (action[3] - thrust_estimate) / CTRL_FREQ
        sim.attitude_control(action[None, None])
        sim.step(sim.freq // CTRL_FREQ)
        position_history.append(np.asarray(sim.data.states.pos[0, 0]))

        if RENDER and ((step * FPS) % CTRL_FREQ) < FPS:
            col_ref = np.array([0.0, 0.1, 1.0, 1.0])
            col_pred = np.array([0.2, 1.0, 0.2, 0.8])
            col_rollouts = np.array([0.25, 0.85, 0.85, 0.18])
            col_hist = np.array([1.0, 0.25, 0.0, 1.0])
            col_pole = np.array([0.9, 0.9, 0.9, 1.0])
            draw_line(sim, reference, rgba=col_ref, start_size=2.0, end_size=2.0)
            for sampled_path in sampled_positions:
                draw_line(sim, sampled_path, rgba=col_rollouts, start_size=0.15, end_size=0.15)
            draw_line(sim, best_positions, rgba=col_pred)
            if len(position_history) > 1:
                draw_line(sim, np.array(position_history), rgba=col_hist, start_size=0.01)
            for pole in obstacles:
                pole_top = pole + np.array([0.0, 0.0, OBSTACLE_HEIGHT])
                draw_capsule(sim, pole, pole_top, radius=OBSTACLE_RADIUS, rgba=col_pole)
            sim.render(
                cam_config={
                    "distance": 3.0,
                    "elevation": -45.0,
                    "azimuth": 90.0,
                    "lookat": [0.0, 0.0, 1.0],
                }
            )

    sim.close()
    rollout_simulator.close()


if __name__ == "__main__":
    main()
