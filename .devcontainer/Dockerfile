FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    git \
    build-essential \
    curl \
    ssh \
    pkg-config \
    patchelf \
    software-properties-common \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libzmq3-dev \
    '^libxcb.*-dev' \
    libx11-dev \
    libx11-xcb-dev \
    libxi-dev \
    libxrender-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    libglu1-mesa-dev \
    python3-tk \
    && rm -rf /var/lib/apt/lists/* \
    libglfw3-dev\
    kbd

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Create non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set environment variables for CUDA
ENV PATH="/usr/local/cuda-12.0/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH}"

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Create virtual environment
ENV VIRTUAL_ENV=/home/vscode/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install development tools
RUN pip install --no-cache-dir ruff

# Create a temporary directory for installation
WORKDIR /tmp/install

# Copy only pyproject.toml for dependency installation
COPY pyproject.toml .

# Install all dependencies including test and gpu extras
RUN pip install --no-cache-dir -e ".[test,gpu]"

# Switch to non-root user
USER $USERNAME

# Set final working directory
WORKDIR /workspace