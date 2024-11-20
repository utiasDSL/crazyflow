# crazyflow
Fast, parallelizable simulations of Crazyflies with JAX and MuJoCo.

## Architecture

<img src="/docs/img/architecture.png" width="75%" alt="Architecture">


## Known Issues
- `"RuntimeError: MUJOCO_PATH environment variable is not set"` upon installing the package: Try to not use `conda` with this project, as the `mujoco` install commonly fails with the mentioned error. Use another environment virtualization, e.g. `venv`, instead.
- If using `zsh` don't forget to escape brackets when installing additional dependencies: `pip install .\[gpu\]`.
