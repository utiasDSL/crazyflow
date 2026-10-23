"""Unit tests for contact detection."""

from __future__ import annotations

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from crazyflow.control import Control
from crazyflow.sim import Dynamics, Sim
from crazyflow.sim.sim import use_box_collision


@pytest.mark.unit
@pytest.mark.parametrize("dynamics", Dynamics)
def test_contacts(dynamics: Dynamics):
    sim = Sim(n_worlds=2, n_drones=2, dynamics=dynamics, freq=500, device="cpu")
    sim.reset()
    sim.step(10)  # Make sure the drone is on the ground
    contacts = sim.contacts()
    assert contacts.shape == (2, 2), "Contacts should be flagged per world and drone"
    assert jnp.all(contacts), "Drone should be in contact with the floor"
    sim.close()


@pytest.mark.unit
@pytest.mark.parametrize("box_collision", [False, True])
def test_contacts_between_drones(box_collision: bool):
    """Two overlapping drones must register a contact."""
    # Regression test: Drones are mocap bodies that are all welded to the world. MuJoCo/MJX filters
    # out collisions between geoms welded into the same kinematic tree, so without explicit contact
    # pairs no drone-drone contact is ever generated.
    sim = Sim(n_drones=2, control=Control.attitude, freq=500, device="cpu")
    if box_collision:
        use_box_collision(sim, True)
    sim.reset()
    states = sim.data.states
    # Place both drones at the same position above the floor
    pos = states.pos.at[:, 1, :].set(states.pos[:, 0, :]).at[..., 2].set(1.0)
    sim.data = sim.data.replace(states=states.replace(pos=pos))
    sim.data = sim.data.replace(core=sim.data.core.replace(mjx_synced=False))
    assert jnp.all(sim.contacts()), "Overlapping drones should be in contact"
    sim.close()


@pytest.mark.unit
def test_contacts_ignore():
    sim = Sim(n_worlds=2, n_drones=3)
    sim.reset()
    pos = sim.data.states.pos.at[..., 2].set(1.0)
    pos = pos.at[0, 1].set(pos[0, 0])  # World 0: drones 0 and 1 overlap
    pos = pos.at[1, 2, 2].set(-0.05)  # World 1: drone 2 is in the floor
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos))
    sim.data = sim.data.replace(core=sim.data.core.replace(mjx_synced=False))
    expected = jnp.array([[True, True, False], [False, False, True]])
    assert jnp.array_equal(sim.contacts(), expected)

    # Drone pairs are correctly ignored
    sim.build_contact_fn(ignore=[("drone:1", "drone:0")])
    expected = jnp.array([[False, False, False], [False, False, True]])
    assert jnp.array_equal(sim.contacts(), expected), "Pairs apply in both directions"

    # Single pairs are correctly ignored. Using the body ID also works
    sim.build_contact_fn(ignore=[(sim.mj_model.body("drone:2").id, "world")])
    expected = jnp.array([[True, True, False], [False, False, False]])
    assert jnp.array_equal(sim.contacts(), expected), "Floor contact not ignored"

    # Single bodies are ignored by all drones
    sim.build_contact_fn(ignore=["world"])
    expected = jnp.array([[True, True, False], [False, False, False]])
    assert jnp.array_equal(sim.contacts(), expected), "Floor contact not ignored"

    # Single drones are also ignored by all other drones
    sim.build_contact_fn(ignore=["drone:0"])
    expected = jnp.array([[False, False, False], [False, False, True]])
    assert jnp.array_equal(sim.contacts(), expected), "Drone 0 not ignored"
    sim.close()


@pytest.mark.unit
def test_contact_bodies():
    sim = Sim(n_worlds=2, n_drones=3)
    sim.reset()
    pos = sim.data.states.pos.at[..., 2].set(1.0)
    pos = pos.at[0, 1].set(pos[0, 0])  # World 0: drones 0 and 1 overlap
    pos = pos.at[1, 2, 2].set(-0.05)  # World 1: drone 2 is in the floor
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos))
    sim.data = sim.data.replace(core=sim.data.core.replace(mjx_synced=False))
    body1, body2, touching = sim.contact_bodies()
    assert body1.shape == body2.shape == touching.shape
    assert touching.sum(axis=-1).tolist() == [1, 1], "Each collision should appear once"
    drone_0, drone_1, drone_2 = (sim.mj_model.body(f"drone:{i}").id for i in range(3))
    floor = sim.mj_model.body("world").id
    collisions = [{int(body1[w][touching[w]][0]), int(body2[w][touching[w]][0])} for w in range(2)]
    assert collisions == [{drone_0, drone_1}, {floor, drone_2}]
    sim.close()


def _attach_obstacle(sim: Sim, pos: list[float], mocap: bool):
    """Attach the minimal obstacle to the sim at ``pos``, mirroring downstream track loading."""
    OBSTACLE_XML = """
    <mujoco>
    <worldbody>
        <body name="obstacle">
        <geom name="obstacle" type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
    </mujoco>
    """
    obstacle_spec = mujoco.MjSpec.from_string(OBSTACLE_XML)
    frame = sim.spec.worldbody.add_frame()
    obstacle = frame.attach_body(obstacle_spec.body("obstacle"), "", ":0")
    obstacle.pos = pos
    obstacle.mocap = mocap  # Mocap bodies (like the gates) are still welded to the world
    sim.build_mjx()


@pytest.mark.unit
@pytest.mark.parametrize("mocap", [True, False])
def test_contacts_with_obstacle(mocap: bool):
    """A drone placed inside an obstacle must register a contact.

    Covers both a mocap obstacle (attached like the gates) and a static one. Both are welded to the
    world just like the drones, so the collision is only detected if explicit contact pairs are
    generated.
    """
    obstacle_pos = [0.0, 0.0, 1.0]
    sim = Sim(control=Control.attitude, freq=500, device="cpu")
    _attach_obstacle(sim, obstacle_pos, mocap=mocap)
    sim.reset()
    states = sim.data.states
    pos = states.pos.at[..., :].set(jnp.array(obstacle_pos))
    sim.data = sim.data.replace(states=states.replace(pos=pos))
    sim.data = sim.data.replace(core=sim.data.core.replace(mjx_synced=False))
    assert jnp.all(sim.contacts()), "Drone should collide with the obstacle"
    sim.build_contact_fn(ignore=["obstacle:0"])
    assert not jnp.any(sim.contacts()), "Obstacle contact should be ignored"
    sim.close()


@pytest.mark.unit
def test_max_geom_pairs_caps_contact_buffer():
    """Small swarms check all geom pairs, large swarms cap the buffer to stay linear."""
    sim = Sim(n_drones=32)
    assert sim.mjx_data._impl.contact.dist.shape[-1] == 32 * 33 // 2
    sim.close()
    n_drones = 64
    sim = Sim(n_drones=n_drones)
    assert sim.mjx_data._impl.contact.dist.shape[-1] == n_drones + 2 * n_drones  # Drones + floor
    sim.max_geom_pairs = -1  # applies on the next build
    sim.build_mjx()
    assert sim.mjx_data._impl.contact.dist.shape[-1] == n_drones * (n_drones + 1) // 2
    sim.close()


@pytest.mark.unit
def test_capped_contacts_identify_colliding_drones():
    """Test that capping the contact buffer still correctly identifies drone collisions by name."""
    n_drones = 16
    sim = Sim(n_drones=n_drones)
    sim.max_geom_pairs = 4
    sim.build_mjx()
    sim.reset()
    pos = np.stack([[i * 2.0, 0.0, 1.0] for i in range(n_drones)])[None]
    pos[0, 4] = pos[0, 3]  # overlap drones 3 and 4, leave the rest far apart
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=jnp.array(pos)))
    sim.step()
    expected = (jnp.arange(n_drones) == 3) | (jnp.arange(n_drones) == 4)
    assert jnp.array_equal(sim.contacts()[0], expected), "Only drones 3 and 4 should collide"
    sim.close()
