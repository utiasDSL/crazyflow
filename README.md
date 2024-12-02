# crazyflow
Fast, parallelizable simulations of Crazyflies with JAX and MuJoCo.

[![Python Version]][Python Version URL] [![Ruff Check]][Ruff Check URL] [![Documentation Status]][Documentation Status URL] [![Tests]][Tests URL]

[Python Version]: https://img.shields.io/badge/python-3.10+-blue.svg
[Python Version URL]: https://www.python.org

[Ruff Check]: https://github.com/utiasDSL/crazyflow/actions/workflows/ruff.yml/badge.svg?style=flat-square
[Ruff Check URL]: https://github.com/utiasDSL/crazyflow/actions/workflows/ruff.yml

[Documentation Status]: https://readthedocs.org/projects/crazyflow/badge/?version=latest
[Documentation Status URL]: https://crazyflow.readthedocs.io/en/latest/?badge=latest

[Tests]: https://github.com/utiasDSL/crazyflow/actions/workflows/testing.yml/badge.svg
[Tests URL]: https://github.com/utiasDSL/crazyflow/actions/workflows/testing.yml


## Architecture

<img src="/docs/img/architecture.png" width="75%" alt="Architecture">


## Known Issues
- `"RuntimeError: MUJOCO_PATH environment variable is not set"` upon installing this package. This error can be resolved by using `venv` instead of `conda`. Somtimes the `mujoco` install can [fail with `conda`](https://github.com/google-deepmind/mujoco/issues/1004).
- If using `zsh` don't forget to escape brackets when installing additional dependencies: `pip install .\[gpu\]`.
