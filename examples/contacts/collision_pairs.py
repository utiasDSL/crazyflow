"""Find out which drone collides with which body."""

import jax.numpy as jnp
import mujoco
import numpy as np

from crazyflow.sim import Sim

OBSTACLE_XML = """
<mujoco>
  <worldbody>
    <body name="obstacle">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    """Place drones into obstacles, the floor and each other, and print each collision."""
    sim = Sim(n_drones=5)
    obstacle_spec = mujoco.MjSpec.from_string(OBSTACLE_XML)
    frame = sim.spec.worldbody.add_frame()
    for i, pos in enumerate([[1.0, 0.0, 0.5], [2.0, 0.0, 0.5]]):
        frame.attach_body(obstacle_spec.body("obstacle"), "", f":{i}").pos = pos
    sim.build_mjx()
    sim.reset()

    pos = jnp.array(
        [[0.85, 0.0, 0.5], [2.15, 0.0, 0.5], [1.5, 0.5, 0.5], [1.5, 0.5, 0.5], [1.5, -0.5, 0.0]]
    )
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=pos[None]), core=sim.data.core.replace(mjx_synced=False)
    )

    body1, body2, touching = sim.contact_bodies()
    for i in np.flatnonzero(touching[0]):
        name1 = sim.mj_model.body(int(body1[0, i])).name
        name2 = sim.mj_model.body(int(body2[0, i])).name
        print(f"{name1} collides with {name2}")

    for _ in range(120):
        sim.render()
    sim.close()


if __name__ == "__main__":
    main()
