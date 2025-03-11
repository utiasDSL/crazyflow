### Using the project with VSCode devcontainers

**Running on CPU**: by default the containers run on CPU. You don't need to take any action.

**Running on GPU**: The devcontsainers can easily run using your computer's NVIDIA GPU on Linux and Windows. This makes sense if you want to accelerate simulation by running thousands of simulation in parallel. In order to work you need to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local), [NVIDIA Container runtime](https://developer.nvidia.com/container-runtime) for your computer. Finally, enable GPU access to the devcontainers by setting the commented out `"--gpus=all"` and `"--runtime=nvidia"` flags in `devcontainer.json`. 


**Linux**
0. Make sure to be in a X11 session ([link](https://askubuntu.com/questions/1410256/how-do-i-use-the-x-window-manager-instead-of-wayland-on-ubuntu-22-04)), otherwise rendering of the drone will fail.
1. Install [Docker](https://docs.docker.com/engine/install/) (, and make sure Docker Daemon is running)
2. Install [VSCode](https://code.visualstudio.com/), with [devcontainer extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), and [remote dev pack](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker).
3. Clone this project's code. Rename `/.devcontainer/devcontainer.linux.json` to `/.devcontainer/devcontainer.json`.
4. Open this project in VSCode. VSCode should automatically detect the devcontainer and prompt you to `Reopen in container`. If not, see [here](https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container) to open manually. Note: Opening the container for the first time might take a while (up to 15 min), as the container is pulled from the web and build.

**Windows** (requires Windows 10 or later)

For windows, we require WSL2 to run the devcontainer. (So its actually Linux with extra steps.) Full instructions can be found [in the official docs](https://code.visualstudio.com/blogs/2020/07/01/containers-wsl#_getting-started). Here are the important steps:
1. Install [Docker](https://docs.docker.com/desktop/setup/install/windows-install/), and WSL2, and Ubuntu 22.04 LTS (, and make sure Docker Daemon is running)
2. Docker will recognize that you have WSL installed and prompt you via Windows Notifications to enable WSL integration -> confirm this with `Enable WSL integration`. If not, open `Docker Desktop`, navigate to the settings, and manually enable WSL integration. (There are TWO setting options for this. Make sure to enable BOTH!)
3. Install [VSCode](https://code.visualstudio.com/), with the [WSL extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl), [devcontainer extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), and [remote dev pack](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker).
4. Clone the source code for the exercises in the WSL2 file system to `/home` (`~`), or wherever you like. (Performance when working on the WSL file system is much better compared to Windows file system). You can access the WSL filesystem either by starting a WSL/Ubuntu console, or via the Windows File Explorer at `\\wsl.localhost\Ubuntu\home` (replace `Ubuntu` with your distro, if necessary).
7. Rename `/.devcontainer/devcontainer.windows.json` to `/.devcontainer/devcontainer.json`.
8. Open this project in VSCode. The easiest way to do so is by starting a WSL/Ubuntu shell, navigating via `cd` to the source code, then type `code .` to open VSCode. VSCode should automatically detect the devcontainer and prompt you to `Reopen in container`. If not, see [here](https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container) to open manually. Note: Opening the container for the first time might take a while (up to 15 min), as the container is pulled from the web and build.


**MacOS**

Unfortunately, we did not get the devcontainer to work with MacOS yet, even after following [those](https://gist.github.com/sorny/969fe55d85c9b0035b0109a31cbcb088) steps. We expect that the issue is related to Mujoco rendering from inside the Docker container and display forwarding with X11. There is also an [unresolved Issue](https://github.com/google-deepmind/mujoco/issues/1047) on GitHub. If you manage to make it work, please let us know.

Until then, MacOS users are required to install this project using an python environment manager such as [conda](https://docs.anaconda.com/anaconda/install/) or [mamba](https://mamba.readthedocs.io/en/latest/). If you use conda, these are the required commands: ```conda create --name crazyflow -c conda-forge python=3.11```, ```conda activate crazyflow```, ```conda install pip```, ```pip install -e .```.

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
