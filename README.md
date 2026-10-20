![Crazyflow Logo](https://github.com/learnsyslab/crazyflow/raw/main/docs/img/logo.png)
--------------------------------------------------------------------------------
<div align="center">

  **Fast, parallelizable simulations of Quadrotor drones with JAX.**

  [![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
  [![arXiv](https://img.shields.io/badge/arXiv-2606.01478-b31b1b.svg)](https://arxiv.org/abs/2606.01478)
  [![Tests](https://github.com/learnsyslab/crazyflow/actions/workflows/testing.yml/badge.svg)](https://github.com/learnsyslab/crazyflow/actions/workflows/testing.yml)
  [![Ruff](https://github.com/learnsyslab/crazyflow/actions/workflows/ruff.yml/badge.svg)](https://github.com/learnsyslab/crazyflow/actions/workflows/ruff.yml)
  [![Docs](https://github.com/learnsyslab/crazyflow/actions/workflows/docs.yml/badge.svg)](https://learnsyslab.github.io/crazyflow)

</div>

Crazyflow is a research simulator for quadrotors. It runs batched, differentiable simulations on CPU and GPU via JAX, with analytical and abstracted dynamics for the Crazyflie 2.x family.

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control
from scipy.spatial.transform import Rotation as R

sim = Sim(n_worlds=4096, n_drones=1, control=Control.state)
cmd = np.zeros((4096, 1, 16))
cmd[..., 9:13] = R.from_euler("z", 0.0).as_quat()
cmd[..., 2] = 0.5  # hover at 0.5 m across all worlds

for _ in range(100):
    sim.state_control(cmd)
    sim.step(sim.freq // sim.control_freq)
    sim.render()
```

## Documentation

[learnsyslab.github.io/crazyflow](https://learnsyslab.github.io/crazyflow) — installation, user guide, examples, and API reference.

## Features

- **n\_worlds x n\_drones** — batched over independent environments and multi-drone swarms simultaneously
- **GPU-accelerated** — up to 914 M steps/s on an RTX 4090 (first-principles dynamics, 262 K worlds)
- **Differentiable** — `jax.grad` works through the full dynamics and control pipeline
- **First-principles dynamics** — dynamics using first-principles equations and parameters identified from real-world measurements
- **Abstracted dynamics** — simplified dynamics in three flavors fitted from real Crazyflie flight data
- **Modular pipelines** — step and reset are tuples of plain JAX functions; insert anything, anywhere
- **MuJoCo integration** — onscreen and offscreen rendering, raycasting, and contact detection via MJX

## Installation

```bash
pip install crazyflow           # CPU
pip install "crazyflow[gpu]"    # GPU (Linux x86-64, CUDA 12)
```

Developer install with editable install ([pixi](https://pixi.sh/) recommended):

```bash
git clone https://github.com/learnsyslab/crazyflow.git
cd crazyflow
pixi shell
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/learnsyslab/crazyflow.git
cd crazyflow
uv sync          # core + dev tooling (tests, docs, ruff)
```

## Performance

First-principles dynamics, one drone. CPU: AMD Ryzen 9 7950X. GPU: NVIDIA RTX 4090.

| n\_worlds | CPU steps/s | GPU steps/s |
|---|---|---|
| 64 | 3.3 M | 1.2 M |
| 1 024 | 9.2 M | 18.7 M |
| 16 384 | 11.9 M | 257 M |
| 65 536 | 15.6 M | 678 M |
| 262 144 | 12.6 M | 914 M |

Full benchmarks including multi-drone scaling are in the [documentation](https://learnsyslab.github.io/crazyflow).

## Citation

```bibtex
@misc{schuck2026crazyflow,
      title={Crazyflow: An Accurate, GPU-Accelerated, Differentiable Drone Simulator in JAX}, 
      author={Martin Schuck and Marcel P. Rath and Yufei Hua and Abhishek Goudar and SiQi Zhou and Angela P. Schoellig},
      year={2026},
      eprint={2606.01478},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.01478}, 
}
```
