# JAX examples

Applying JAX transformations to the simulator: differentiating through the dynamics with `jax.grad`, distributing the simulation across devices, and caching compiled functions. See [Sharding](../user-guide/sharding.md) for the details on multiple devices.

---

## Gradient descent through dynamics

--8<-- example "examples/jax/gradient.py"

---

## Gradients and command clipping

--8<-- example "examples/jax/gradient_clipping.py"

---

## Sharding across devices

--8<-- example "examples/jax/sharding.py"

---

## Persistent compilation cache

--8<-- example "examples/jax/cache.py"
