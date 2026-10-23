# Contact examples

Querying contacts between drones and the scene, and choosing the collision geometry of the drone. See [MuJoCo Integration](../user-guide/mujoco.md) for how the scene is built.

---

## Contact queries

<div class="example-media-grid example-media-grid--contacts">
  <figure>
    <img src="../../img/examples/contact_sphere.png" alt="Contact query visualization using the default sphere collision geometry">
  </figure>
  <figure>
    <img src="../../img/examples/contact_box.png" alt="Contact query visualization using the oriented box collision geometry">
  </figure>
</div>

--8<-- example "examples/contacts/contacts.py"

---

## Crash detection

--8<-- example "examples/contacts/crash.py"

---

## Collision pairs

--8<-- example "examples/contacts/collision_pairs.py"
