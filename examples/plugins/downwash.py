"""Model the downwash between identical Crazyflies with a far-field external-wrench plugin.

The plugin uses the far-field jet from [1] and the thrust-decay model of [2]. It samples the flow
at each target rotor and center of mass, computes rotor thrust loss and drag, and writes the
resulting world-frame external force and torque.

The demo holds one `cf21B_500` drone in hover while a second makes three passes underneath at
different speeds and heights.

Run with `--plot` to plot the results, which requires matplotlib from the `tests` environment.

[1] Bauersfeld et al. https://arxiv.org/abs/2403.13321
[2] Su et al. https://arxiv.org/abs/2207.09645
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.sim import Sim
from crazyflow.sim.pipeline import insert_fn_before

if TYPE_CHECKING:
    from jax import Array

    from crazyflow.sim.data import SimData

# Physical parameters for the cf21B_500
AIR_DENSITY = 1.225  # kg/m^3
PROPELLER_RADIUS = 27.5e-3  # m
MOTOR_DISTANCE = 0.1  # m, distance between opposite motors

# This must be fitted for the propeller/downwash setup.
THRUST_DECAY_COEFFICIENT = 0.07  # s/m

# Far-field fit in Eq. (9) of [1]
BD = 10.11
S = 0.07668
S0 = -5.817


def downwash_speed(s: Array, r: Array, u_hover: Array) -> Array:
    """Return downwash speed at axial and radial distances in metres.

    Positive s points downstream along the source drone's negative body z-axis.
    Inputs broadcast to the sampling grid; the returned speeds are in m/s.
    """
    # Normalization according to [1] Eq. (8).
    s_normalized = s / MOTOR_DISTANCE
    r_normalized = r / MOTOR_DISTANCE

    # Keep the fit finite upstream, where its contribution is masked below.
    axial_distance = jnp.maximum(s_normalized - S0, 1e-6)
    half_width = S * axial_distance  # [1] Eq. (6)
    centerline_speed = u_hover * BD / axial_distance  # [1] Eq. (5)
    radial_ratio = r_normalized / half_width  # [1] Eq. (4)

    # [1] Eq. (3).
    speed = centerline_speed / (1.0 + (jnp.sqrt(2.0) - 1.0) * radial_ratio**2) ** 2
    return jnp.where(s_normalized > 0.1, speed, 0.0)


def downwash_fn(data: SimData) -> SimData:
    """Apply downwash-induced thrust loss and drag as a world-frame external wrench.

    The source flow originates at each drone centre, while the field is sampled
    at every target rotor and CoM in the source's body frame.
    """
    body_to_world = R.from_quat(data.states.quat)

    mixing_matrix = data.params.mixing_matrix
    drag_matrix = data.params.drag_matrix

    offsets = data.params.L * jnp.stack(
        [-mixing_matrix[1], mixing_matrix[0], jnp.zeros_like(mixing_matrix[0])], axis=-1
    )
    rotor_offsets_world = R.from_quat(data.states.quat[..., None, :]).apply(offsets)
    rotor_positions = data.states.pos[..., None, :] + rotor_offsets_world

    sample_positions = jnp.concatenate([rotor_positions, data.states.pos[..., None, :]], axis=2)

    # Axis 1 indexes the source, axis 2 the target, and axis 3 its rotors then CoM.
    source_to_target = data.states.pos[:, :, None, None, :] - sample_positions[:, None, :, :, :]

    # Broadcast each source rotation across all target drones and sampling points.
    source_to_target_body = R.from_quat(data.states.quat[..., None, None, :]).apply(
        source_to_target, inverse=True
    )
    s = source_to_target_body[..., 2]
    r = jnp.linalg.vector_norm(source_to_target_body[..., :2], axis=-1)

    mass = data.params.mass[0]
    gravity = -data.params.gravity_vec[2]
    n_propellers = mixing_matrix.shape[-1]

    u_hover = jnp.sqrt(
        mass * gravity / (2.0 * AIR_DENSITY * jnp.pi * PROPELLER_RADIUS**2 * n_propellers)
    )  # [1] Eq. (1)

    # Shape: (world, source, target, sample), with rotors followed by the CoM.
    sample_speed = downwash_speed(s, r, u_hover)

    z_axes = body_to_world.as_matrix()[..., 2]

    # The final sample is the CoM; each source's wind follows its negative z-axis.
    wind_com_world = jnp.sum(-sample_speed[..., -1, None] * z_axes[:, :, None, :], axis=1)
    wind_com_body = body_to_world.apply(wind_com_world, inverse=True)

    # Project only the rotor samples onto the target axis for thrust loss.
    cos_theta = jnp.sum(z_axes[:, :, None, :] * z_axes[:, None, :, :], axis=-1)
    # Sum all sources at each target rotor: (world, target, rotor).
    rotor_inflow = jnp.sum(sample_speed[..., :-1] * cos_theta[..., None], axis=1)

    # [2] Eq. (5): each motor loses a fraction b_v * U_D of its current thrust
    loss_fraction = THRUST_DECAY_COEFFICIENT * rotor_inflow
    rotor_vel = data.states.rotor_vel
    k0, k1, k2 = (
        data.params.rpm2thrust[..., 0],
        data.params.rpm2thrust[..., 1],
        data.params.rpm2thrust[..., 2],
    )
    motor_thrust = k0 + k1 * rotor_vel + k2 * rotor_vel**2
    thrust_delta = -loss_fraction * motor_thrust

    # Map the per-motor force changes to a body-frame wrench, as in [2] Eq. (7).
    total_thrust_delta = jnp.sum(thrust_delta, axis=-1)

    zeros = jnp.zeros_like(total_thrust_delta)
    force_body = jnp.stack((zeros, zeros, total_thrust_delta), axis=-1)

    # Compute drag induced through downwash
    drag_body = (-drag_matrix @ wind_com_body[..., None])[..., 0]
    force_body += drag_body

    # Compute the torque generated by downwash
    lever = jnp.array([1.0, 1.0, 0.0])
    torque_body = (mixing_matrix @ (thrust_delta * data.params.L)[..., None])[..., 0] * lever

    states = data.states.replace(
        force=body_to_world.apply(force_body), torque=body_to_world.apply(torque_body)
    )
    return data.replace(states=states)


def plot_hover_velocity_field(source_positions: np.ndarray, data: SimData) -> None:
    """Plot the far-field downwash-speed magnitude in the y=0 plane."""
    import matplotlib.pyplot as plt

    x = np.linspace(-0.6, 0.6, 300)
    z = np.linspace(0.0, 1.15, 300)
    X, Z = np.meshgrid(x, z)

    # Every grid point lies in the y=0 plane.
    points = np.stack((X, np.zeros_like(X), Z), axis=-1)
    u_downwash = np.zeros_like(X)

    gravity = -data.params.gravity_vec[2]
    n_propellers = data.params.mixing_matrix.shape[-1]
    mass = data.params.mass[0]

    u_hover = np.sqrt(
        mass * gravity / (2.0 * AIR_DENSITY * np.pi * PROPELLER_RADIUS**2 * n_propellers)
    )

    for source_pos in source_positions:
        source_to_point = source_pos - points
        s = source_to_point[..., 2]
        r = np.linalg.vector_norm(source_to_point[..., :2], axis=-1)
        u_downwash += np.asarray(downwash_speed(jnp.asarray(s), jnp.asarray(r), u_hover))

    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    image = ax.pcolormesh(X, Z, u_downwash, shading="auto", cmap="viridis")
    ax.scatter(source_positions[:, 0], source_positions[:, 2], color="red", label="source drone")
    ax.set_xlabel("x (m)", fontsize=14)
    ax.set_ylabel("z (m)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title("Hovering-drone downwash speed", fontsize=14)
    ax.legend()
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("downward airspeed $U_D$ (m/s)", fontsize=14)
    colorbar.ax.tick_params(labelsize=12)
    plt.show()


def main(plot: bool = False) -> None:
    """Hover drone 0 while drone 1 makes three downwash passes at different height and velocity."""
    sim = Sim(n_drones=2, drone="cf21B_500", control="state")

    insert_fn_before(sim.step_pipeline, "integration", downwash_fn)
    sim.build_step_fn()

    upper_pos = np.array([0.0, 0.0, 1.2])
    first_run_height = 0.5
    second_run_height = 0.95
    lower_start = np.array([-0.5, 0.0, first_run_height])

    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=jnp.array([[upper_pos, lower_start]]))
    )
    sim.build_default_data()

    command = np.zeros((1, 2, 16))
    command[..., 9:13] = R.from_euler("z", 0).as_quat()
    command[0, 0, :3] = upper_pos

    waypoints = np.concatenate(
        (
            # First pass slow
            np.linspace(
                lower_start, [0.3, 0.0, first_run_height], 3 * sim.control_freq, endpoint=False
            ),
            # Pause to stabilize
            np.tile([0.3, 0.0, first_run_height], (2 * sim.control_freq, 1)),
            # Second pass fast
            np.linspace([0.3, 0.0, first_run_height], lower_start, int(0.5 * sim.control_freq)),
            # Pause to stabilize
            np.tile(lower_start, (2 * sim.control_freq, 1)),
            # Increase altitude
            np.linspace(
                lower_start, [-0.5, 0.0, second_run_height], sim.control_freq, endpoint=False
            ),
            # Third pass closer to hovering drone
            np.linspace(
                [-0.5, 0.0, second_run_height], [0.5, 0.0, second_run_height], 3 * sim.control_freq
            ),
        )
    )

    z_positions = []
    downwash_force_z = []
    downwash_pitch_torque = []

    for position in waypoints:
        command[0, 1, :3] = position

        sim.state_control(command)
        sim.step(sim.freq // sim.control_freq)

        z_positions.append(np.asarray(sim.data.states.pos[0, :, 2]))
        downwash_force_z.append(np.asarray(sim.data.states.force[0, 1, 2]))
        downwash_pitch_torque.append(np.asarray(sim.data.states.torque[0, 1, 1]))
        sim.render()

    sim.close()

    if plot:
        plot_results(
            z_positions,
            downwash_force_z,
            downwash_pitch_torque,
            upper_pos,
            sim.data,
            sim.control_freq,
        )


def plot_results(
    z_positions: list[np.ndarray],
    downwash_force_z: list[np.ndarray],
    downwash_pitch_torque: list[np.ndarray],
    upper_pos: np.ndarray,
    data: SimData,
    control_freq: int,
) -> None:
    """Plot the passes of the lower drone and the downwash field of the hovering drone."""
    import matplotlib.pyplot as plt

    t = np.arange(len(z_positions)) / control_freq
    z_positions = np.asarray(z_positions)

    fig, axes = plt.subplots(3, 1, sharex=True)

    axes[0].plot(t, z_positions[:, 0], label="upper drone")
    axes[0].plot(t, z_positions[:, 1], label="lower drone")
    axes[0].set_ylabel("z position (m)")
    axes[0].legend()

    axes[1].plot(t, downwash_force_z, label="lower drone")
    axes[1].set_ylabel("downwash force z (N)")
    axes[1].legend()

    axes[2].plot(t, downwash_pitch_torque, label="lower drone")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("downwash pitch torque y (Nm)")
    axes[2].legend()

    plot_hover_velocity_field(np.asarray([upper_pos]), data)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot the results")
    main(plot=parser.parse_args().plot)
