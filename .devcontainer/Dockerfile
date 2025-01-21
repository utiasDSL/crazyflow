FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    build-essential \
    git \
    curl \
    ssh \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Create non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Set environment variables for CUDA
ENV PATH="/usr/local/cuda-12.0/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH}"

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Create and set proper permissions for workspace directory
RUN mkdir -p /workspace && chown -R $USERNAME:$USERNAME /workspace && chmod -R 755 /workspace

# Create virtual environment with proper permissions
ENV VIRTUAL_ENV=/home/vscode/venv
RUN python -m venv $VIRTUAL_ENV && \
    chown -R $USERNAME:$USERNAME $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Switch to non-root user
USER $USERNAME

# Install development tools
RUN pip install --no-cache-dir ruff

# Create a temporary directory for installation with proper permissions
WORKDIR /tmp/install

# Copy pyproject.toml with correct ownership
COPY --chown=$USERNAME:$USERNAME pyproject.toml .

# Install dependencies as non-root user
RUN pip install --no-cache-dir -e ".[test,gpu]"

# Set final working directory
WORKDIR /workspace