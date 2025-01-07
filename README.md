### Using the project with vscode devcontainer
**Linux**

0. Install docker: https://docs.docker.com/engine/install/ (, and make sure docker daemon is running)
2. Install [vscode](https://code.visualstudio.com/), with [devcontainer extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), and [remote dev pack](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker).
3. Open this project in vscode. Vscode should automatically detect the devcontainer and prompt you to install the [required extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), and subsequently prompt you to `Reopen in container`. Note: Opening the container for the first time might take a while (up to 10mins), as the container is pulled from the web and build.

**Windows** (requires Windows 10 or later)

For windows, we use WSL2 to run the devcontainer. Full instructions can be found [in the official docs](https://code.visualstudio.com/blogs/2020/07/01/containers-wsl#_getting-started). Here are the important steps:
1. Install docker: https://docs.docker.com/desktop/setup/install/windows-install/, and WSL 2 and Ubuntu 22.04 LTS (, and make sure docker daemon is running)
2. Docker will recognize that you have WSL installed and prompt you to enable WSL integration -> confirm with `Enable WSL integration`. If not, open `Docker Desktop`, navigate to the settings, and manually Enable WSL integration. (There are TWO setting options for this. Make sure to enable both!)
3. Install [vscode](https://code.visualstudio.com/), with the [WSL extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl), [devcontainer extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), and [remote dev pack](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker).
4. Clone the source code for the exercises in the WSL 2 filesystem to `/home` (`~`), or wherever you like. (Performance is much better compared to copying it in the windows filesystem). You can access the WSL filesystem either by starting a WSL console, or via the Windows File explorer at `\\wsl.localhost\Ubuntu\home` (replace `Ubuntu` with your distro, if necessary).
5. Open this project in vscode. The easiest way is to start an Ubuntu shell, navigate the the source code, type `code .` to open VSCode. Vscode should automatically detect the devcontainer and prompt you to install the [required extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), and subsequently prompt you to `Reopen in container`. Note: Opening the container for the first time might take a while (up to 10mins), as the container is pulled from the web and build.


**Mac**

Should work like Ubuntu. Untested.

____________

Known Issues:
   - if building docker container fails at `RUN apt-get update`, make sure your host systems time is set correct: https://askubuntu.com/questions/1511514/docker-build-fails-at-run-apt-update-error-failed-to-solve-process-bin-sh

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
