#!/bin/bash

echo "Running postStartCommand: Checking for WSL2 environment..."

# Check if running on WSL2
if grep -q "microsoft" /proc/version; then
    echo "Detected WSL2. Setting up WSL-specific mounts..."

    # Ensure directories exist inside the container
    mkdir -p /tmp/.X11-unix
    mkdir -p /mnt/wslg
    mkdir -p /usr/lib/wsl

    # Bind the WSL2-specific mounts
    mount --bind /tmp/.X11-unix /tmp/.X11-unix
    mount --bind /mnt/wslg /mnt/wslg
    mount --bind /usr/lib/wsl /usr/lib/wsl

    echo "WSL-specific mounts set up successfully."
else
    echo "Not running on WSL2. Skipping WSL-specific mounts."
fi
