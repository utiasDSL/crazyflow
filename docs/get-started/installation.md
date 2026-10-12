# Installation

Select your installation method from the tabs below, then read the notes under each section for what it includes.

=== "pip"

    ```bash
    pip install crazyflow
    ```

=== "pip + GPU"

    ```bash
    pip install "crazyflow[gpu]"
    ```

=== "pixi"

    ```bash
    git clone https://github.com/learnsyslab/crazyflow.git
    cd crazyflow
    pixi shell
    ```

=== "pixi + tests"

    ```bash
    git clone https://github.com/learnsyslab/crazyflow.git
    cd crazyflow
    pixi shell -e tests
    ```

=== "uv"

    ```bash
    git clone https://github.com/learnsyslab/crazyflow.git
    cd crazyflow
    uv sync                 # core + dev tooling (tests, docs, ruff)
    uv run python -c "from crazyflow.sim import Sim; Sim().reset()"
    ```

---

## GPU support

JAX defaults to CPU-only execution. The `gpu` extra swaps in `jax[cuda12]`, enabling GPU execution. Setting `device="gpu"` in the `Sim` constructor then routes all computation through CUDA.

!!! note
    GPU support is only available on Linux x86-64.

## Developer install

[Pixi](https://pixi.sh/) creates a fully reproducible environment (locked via `pixi.lock`). This variant installs `crazyflow` in editable mode. Any source change takes effect immediately without reinstalling. Recommended for contributors and researchers who modify the simulator.

[uv](https://docs.astral.sh/uv/) is supported as an alternative. `uv sync` creates a `.venv` with `crazyflow` installed editable plus the `dev` dependency group (tests, docs, ruff). The dependency groups (`tests`, `docs`, `dist`, `dev`) mirror the pixi features and are defined under `[dependency-groups]` in `pyproject.toml`. The uv lockfile is not committed — `pixi.lock` remains the canonical reproducible environment.

```bash
uv sync                 # core + dev group (default)
uv sync --group docs    # only the docs group + core
uv sync --no-default-groups --extra gpu   # core + GPU extra, no dev tooling
```

## Testing

Adds `pytest` and `pytest-markdown-docs` for running the test suite and doc snippet tests.

=== "pixi"

    ```bash
    pixi run tests          # unit and integration tests
    pixi run test-docs      # doc code snippet tests
    ```

=== "uv"

    ```bash
    uv run pytest -v tests  # unit and integration tests
    uv run pytest -v --markdown-docs --markdown-docs-syntax=superfences crazyflow/ docs/
    ```

## Verify the installation

```bash
python -c "from crazyflow.sim import Sim; sim = Sim(); sim.reset(); print('OK')"
```
