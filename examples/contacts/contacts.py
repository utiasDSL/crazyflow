import mujoco
import numpy as np

from crazyflow.control import Control
from crazyflow.sim import CollisionType, Dynamics, Sim
from crazyflow.sim.sim import select_collision_type

RED, GREEN = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)


def add_contact_indicators(sim: Sim) -> list[int]:
    """Add a disc above each drone that turns red on contact and return the geom ids."""
    for i in range(sim.n_drones):
        geom = sim.spec.body(f"drone:{i}").add_geom()
        geom.name = f"indicator:{i}"
        geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        geom.size[:] = [0.05, 0.003, 0.0]
        geom.pos[:] = [0.0, 0.0, 0.1]
        geom.contype = 0
        geom.conaffinity = 0
        geom.rgba[:] = [*GREEN, 1.0]
    sim.build_mjx()
    return [sim.mj_model.geom(f"indicator:{i}").id for i in range(sim.n_drones)]


def crossing_trajectory(t: float, amplitude: float, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Drones 0 and 2 oscillate along x in opposite directions through the hovering drone 1.

    Returns:
        Position and velocity setpoints at time ``t`` with shape (3, 3) each.
    """
    direction = np.array([1.0, 0.0, -1.0])
    phase = 2 * np.pi * t / period
    pos, vel = np.zeros((3, 3)), np.zeros((3, 3))
    pos[:, 0] = amplitude * np.sin(phase) * direction
    pos[:, 2] = 1.0
    vel[:, 0] = amplitude * 2 * np.pi / period * np.cos(phase) * direction
    return pos, vel


def main():
    """Fly the outer drones back and forth through a hovering drone and check for contacts.

    The crossing is repeated with every collision type. Each type reports contacts for a different
    fraction of the crossing, from the loose sphere to the tight convex hull. A disc above each
    drone turns red while the drone is in contact. Press 3 in the viewer to show the collision
    geometry.
    """
    n_worlds, n_drones = 2, 3
    sim = Sim(
        n_worlds=n_worlds,
        n_drones=n_drones,
        control=Control.state,
        dynamics=Dynamics.so_rpy,
        device="cpu",
    )
    indicators = add_contact_indicators(sim)
    fps = 60
    amplitude, period = 0.3, 2.0

    pos, _ = crossing_trajectory(0.0, amplitude, period)  # Start on the trajectory
    states = sim.data.states
    sim.data = sim.data.replace(states=states.replace(pos=states.pos.at[:].set(pos)))
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 16))
    cmd[..., 12] = 1.0  # Identity quaternion
    for collision_type in CollisionType:
        print(f"Collision type: {collision_type}")
        select_collision_type(sim, collision_type)  # Switches mid-flight, the sim keeps its state
        for i in range(int(2 * period * sim.control_freq)):
            t = i / sim.control_freq
            cmd[..., :3], cmd[..., 3:6] = crossing_trajectory(t, amplitude, period)
            sim.state_control(cmd)
            sim.step(sim.freq // sim.control_freq)
            if ((i * fps) % sim.control_freq) < fps:
                for drone, gid in enumerate(indicators):
                    hit = sim.contacts(f"drone:{drone}")[0].any()
                    sim.mj_model.geom_rgba[gid, :3] = RED if hit else GREEN
                sim.render()
                print(f"Contacts: {sim.contacts().any()}")
    sim.close()


if __name__ == "__main__":
    main()
