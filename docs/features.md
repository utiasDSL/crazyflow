# Features & Architecture

Crazyflow is a research‑first simulator for small quadrotors. Its design favours clarity, composability and performance: a compact Sim API composes a physics model, a numerical integrator and a controller layer. Around that core sits an extensible "step pipeline" where randomization, disturbances, logging or custom hooks can be inserted without changing the simulator internals.

### Architecture
The goals are reproducibility, throughput and research flexibility. First‑principles (analytic, identified) and simplified, data‑driven models are supported alongside a symbolic model API for controller development and analysis. The codebase is implemented to take advantage of JAX and MuJoCo: batched GPU execution, JIT compilation and automatic differentiation are available, and analytical gradients of the simulation are exposed where useful. The step pipeline makes it simple to compose randomization, disturbances and custom hooks without touching the core simulator.

### Models

Crazyflow supports two complementary model classes:

- First‑principles models — physics‑based analytical models with identified parameters intended for high‑fidelity simulation and sim‑to‑real work.
- Simplified, data‑driven models — lightweight models fitted from flight data to capture off‑nominal effects and to speed up learning/control experiments. These data‑driven models can be obtained from just a few minutes of flight data; the repository provides fitting scripts and a minimal identification pipeline (works quickly if a stable controller is available).

All models (first‑principles and data‑driven) are also exposed as symbolic model objects so they can be reused directly in model‑based controllers, MPC formulations, or analysis tools.

### Controllers
The repository includes reference controller implementations and integration points for common research workflows:

- Geometric controllers (Mellinger style) for standard tracking.
- Interfaces for MPC / MPCC workflows (note: MPC example controllers and advanced control code are available in the repo for our [drone racing course](https://github.com/utiasDSL/lsy_drone_racing/tree/main/lsy_drone_racing/control)).
- Reinforcement learning: we provide environments suitable for training and deploying RL agents and include example setups for PPO and SHAC agents.

### Performance & evaluation
Assuming free‑space flight (avoiding generic contact solving when possible) lets Crazyflow prioritise speed and simplicity. The combination of JAX, MuJoCo and a modular pipeline enables high‑throughput experiments and large‑scale benchmarking.

### Extensibility
Researchers can add new dynamics, controllers or pipeline stages without modifying the simulator core. Swap integrators, change physics models, or inject disturbances via small, well‑scoped functions.

### Publication
See our upcoming paper for full motivation, benchmarks and evaluation:  
Schuck, M. & Rath, M. (2025). Crazyflow: Fast, parallelizable simulations of Crazyflies with JAX and MuJoCo. (preprint — link TBD)

For API details and configuration options, see the API reference: [API Reference](api/index.md)