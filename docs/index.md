# Crazyflow

<div align="center">
  <img src="img/logo.svg" alt="Crazyflow Logo" width="360"/>
</div>

## Overview

Crazyflow is a high-performance research simulator for Crazyflie‑style small quadrotors.  
Built on JAX and MuJoCo, it supports batched GPU execution, differentiable dynamics, and accurate, identified models — designed for reproducible experiments at scale.

Audience: researchers working on control, learning, system identification, sim2real, multi-agent RL and swarm control for quadrotors.

## Highlights

- Modular simulation stack (physics, integrator, controller)
- GPU-ready, batched execution for massive parallelism
- Differentiable / autodiff-enabled dynamics
- Support for analytical (identified) and data-driven models
- Onboard-controller support and symbolic model matching
- Extensible step pipeline (randomization, disturbances, custom hooks)
- MuJoCo-based visualization and offscreen rendering

![Performance benchmarks and comparisons](img/performance.png)

## Quick start

1. Installation — follow the install instructions: [Installation Guide](installation.md)  
2. Run an example — see runnable demos and thumbnails: [Examples](examples.md)  
3. API & reference — full Python API generated with mkdocstrings: [API Reference](api/index.md)

## Want to learn more?

- Read about features and architecture: [Features & Architecture](features.md)  
- Try the simple quickstart: [Getting Started / Usage](usage.md)