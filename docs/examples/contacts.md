# Contact examples

Querying contacts between drones and the scene, and choosing the collision geometry of the drone. See [MuJoCo Integration](../user-guide/mujoco.md) for how the scene is built.

---

## Contact queries

The default collision geometry is a sphere around the drone frame. `use_box_collision` replaces it with a tighter oriented box, useful for narrow-gap flight and accurate contact debugging.

<div class="example-media-grid example-media-grid--contacts">
  <figure>
    <img src="../../img/examples/contact_sphere.png" alt="Contact query visualization using the default sphere collision geometry">
  </figure>
  <figure>
    <img src="../../img/examples/contact_box.png" alt="Contact query visualization using the oriented box collision geometry">
  </figure>
</div>

<!-- notest: imported script, covered by tests/integration/test_examples.py -->
```{ .python notest }
--8<-- "examples/contacts/contacts.py"
```
