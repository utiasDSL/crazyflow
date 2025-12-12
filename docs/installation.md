# Installation

## Requirements
The simulator is extensively tested on Ubuntu 24.04. However, other platforms should also work out of the box with the PyPI install. Further, you need an environment with:

- Python >= 3.11, < 3.14
- MuJoCo (follow the mujoco installation instructions for your platform)
- (Optional) pixi >= 0.6.1 (required for the provided developer environment)

## Quick install (PyPI)
Recommended for users who only want to use Crazyflow:

```bash
pip install crazyflow
```

## Developer install (recommended for contributors)
Use [pixi](https://pixi.sh/) to create a reproducible development environment that also installs submodules in editable mode. This requires some 64 bit linux distribution to work.

1. Clone the repo (with submodules)
```bash
git clone --recurse-submodules git@github.com:utiasDSL/crazyflow.git
cd crazyflow
```

2. Enter the [pixi](https://pixi.sh/) shell (pixi >= 0.6.1 required)
```bash
pixi shell
```

Inside the pixi shell you will have crazyflow, drone-models and drone-controllers installed in editable mode.

## Install from source (manual / without pixi)
If you prefer not to use pixi, install the packages manually in editable mode:

```bash
pip install -e .  # Install crazyflow
pip install -e ./submodules/drone-models
pip install -e ./submodules/drone-controllers
```

## Optional extras (GPU / benchmarking)
- GPU (JAX with CUDA): three ways to enable the gpu extras
  - Start the [pixi](https://pixi.sh/) gpu environment:
    ```bash
    pixi shell -e gpu
    ```
  - Install local editable package with gpu extras:
    ```bash
    pip install -e ".[gpu]"
    ```
  - Install from PyPI with gpu extras:
    ```bash
    pip install "crazyflow[gpu]"
    ```

- Benchmarking / plotting:
  - Use the dedicated pixi benchmarking environment:
    ```bash
    pixi shell -e benchmark
    ```
  - Or install the benchmark extras locally:
    ```bash
    pip install -e ".[benchmark]"
    ```
  - Or install from PyPI with extras:
    ```bash
    pip install "crazyflow[benchmark]"
    ```

## Verify installation
A quick smoke test — run one of the examples (offscreen or with rendering enabled depending on your platform):
```bash
python examples/cameras.py
```

Run the test suite (recommended via [pixi](https://pixi.sh/)):
```bash
pixi run tests
```
Or activate the test environment and run pytest directly:
```bash
pixi shell -e tests
pytest -v tests
```

See the [Examples](examples.md) page for more runnable scripts.

## Building and serving the docs locally
Recommended: use [pixi](https://pixi.sh/) to run the configured docs tasks:

```bash
pixi run docs-build  # build the static site
pixi run docs-serve  # serve with live reload
```