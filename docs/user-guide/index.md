# User Guide

In-depth documentation for every part of the simulator.

- [Simulator Overview](sim-overview.md) — `SimData` layout, worlds, drones, and the data convention
- [Object-Oriented API](oo-api.md) — `Sim` class, control methods, rendering, and reset
- [Functional API](functional-api.md) — purely functional interface for JAX transformations
- [Dynamics](dynamics/index.md) — first-principles vs. fitted dynamics, when to use each
- [Control Modes](control/index.md) — state, attitude, body rate, force/torque, and rotor velocity control
- [Adding a drone](adding-drones.md) — the MJCF model and parameter sections that define a platform
- [Pipelines](pipelines.md) — composable step and reset pipelines, randomization, and disturbances
- [The world axis](world-axis.md) — which arrays are batched over worlds, and what resets and sharding do with them
- [MuJoCo Rendering](mujoco-rendering.md) — rendering modes, cameras, raycasting, and materials
- [Gaussian Splat Rendering](splat-rendering.md) — photorealistic splat viewer and camera sensors
- [MuJoCo Integration](mujoco.md) — MJCF scene construction, adding objects, and sync internals
- [Gymnasium Environments](gymnasium-envs.md) — vectorized environments for RL training
