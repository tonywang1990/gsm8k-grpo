#!/bin/bash

# Set up git
git config --global user.email "tonybestrong@gmail.com"
git config --global user.name "Tony Wang"

# setup venv
source .venv/bin/activate

# Install CUDA compat only if the driver's CUDA version is older than 12.9
# See: https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html
DRIVER_CUDA=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "0.0")
REQUIRED_CUDA="12.9"

if [ "$(printf '%s\n' "$REQUIRED_CUDA" "$DRIVER_CUDA" | sort -V | head -n1)" != "$REQUIRED_CUDA" ]; then
    echo "Driver CUDA $DRIVER_CUDA < $REQUIRED_CUDA, installing compat libs..."
    if ! dpkg -s cuda-compat-12-9 &>/dev/null; then
        apt-get update -qq && apt-get install -y -qq cuda-compat-12-9
    fi
    export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH
else
    echo "Driver CUDA $DRIVER_CUDA >= $REQUIRED_CUDA, compat libs not needed."
fi

# Install and setup Claude Code -- run at end of script

echo "Installing Claude Code..."
curl -fsSL https://claude.ai/install.sh | bash

# Add to PATH in .bashrc if not already present
if ! grep -q '\.local/bin' ~/.bashrc; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# Export PATH for current script session
export PATH="$HOME/.local/bin:$PATH"