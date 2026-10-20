# Examples

The runnable examples are grouped by topic, following the folders in `examples/`. Start with [hover](control.md#hover) if you're new, or jump to the group that matches your use case. Run an example from the repository root with `python examples/<group>/<script>.py`.

- [Control](control.md) — how to command the drones at different levels, such as positions, attitudes, or body rates, and how to build your own controllers on top
- [JAX](jax.md) — how to use JAX transformations on the simulation, such as gradients through the dynamics or sharding across devices
- [Plugins](plugins.md) — how to easily add features to the simulation, such as domain randomization, disturbances, or aerodynamic effects
- [Rendering](rendering.md) — how to render the simulation with MuJoCo or gaussian splats and extract sensor data, such as camera images or depth maps
- [Contacts](contacts.md) — how to detect collisions between drones and their environment
- [Environments](environments.md) — how to train and evaluate reinforcement learning policies with Gymnasium environments
