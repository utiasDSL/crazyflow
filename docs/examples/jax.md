# JAX examples

Applying JAX transformations to the simulator: differentiating through the dynamics with `jax.grad` and distributing the simulation across devices.

---

## Gradient descent through dynamics

Because the simulator is built entirely from JAX operations, `jax.grad` can differentiate through it. Starting the drone above the target height keeps it away from the floor, so the floor-clipping stage never fires and gradients flow freely through the entire trajectory.

<!-- notest: imported script, covered by tests/integration/test_examples.py -->
```{ .python notest }
--8<-- "examples/jax/gradient.py"
```

---

## Gradients and command clipping

Hard clips such as the `clip_rotor_vel_cmd` stage zero the gradients while the command is saturated. This example ramps a motor command beyond the rotor limits and compares the rotor state and the gradient of the vertical acceleration w.r.t. the command for three options: the default clip, a straight-through clip (clipped forward pass, unclipped gradients), and no clip. Replacing the default clip with the straight-through variant can help gradient-based methods such as trajectory optimization or policy learning, which would otherwise receive zero gradients whenever the motors saturate.

<!-- notest: imported script, covered by tests/integration/test_examples.py -->
```{ .python notest }
--8<-- "examples/jax/gradient_clipping.py"
```

---

## Sharding across devices

We can also distribute the simulation across devices. The host backend is asked for four logical devices. We then compare this against running on a single device. See [Sharding](../user-guide/sharding.md) for the details.

<!-- notest: imported script, covered by tests/integration/test_examples.py -->
```{ .python notest }
--8<-- "examples/jax/sharding.py"
```
