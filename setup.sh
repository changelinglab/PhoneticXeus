#!/bin/bash
# setup.sh — install script using pyproject.toml + uv sync
#
# Usage:
#   bash setup.sh          # auto-detects x86_64 vs aarch64
#   bash setup.sh x86      # force x86 extra (Delta / Babel)
#   bash setup.sh dai      # force dai extra (Delta-AI)
set -e

# Install uv if missing
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Installing..."
    curl -fsSL https://astral.sh/uv/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install ffmpeg if missing (via pixi)
if ! command -v ffmpeg >/dev/null 2>&1; then
    command -v pixi >/dev/null || curl -fsSL https://pixi.sh/install.sh | bash
    pixi global install ffmpeg
fi

# Determine extra: arg takes precedence, otherwise detect from arch
if [[ -n "$1" ]]; then
    EXTRA="$1"
elif [[ "$(uname -m)" == "aarch64" ]]; then
    EXTRA="dai"
else
    EXTRA="x86"
fi

echo "Platform: $(uname -m) → installing extra: $EXTRA"

# Create venv if needed and sync deps
uv sync --extra "$EXTRA"

if [[ "$EXTRA" == "x86" ]]; then
  uv pip install "espnet @ git+https://github.com/espnet/espnet.git"
elif [[ "$EXTRA" == "dai" ]]; then
  uv pip install "espnet @ git+https://github.com/espnet/espnet.git"
fi

source .venv/bin/activate

# x86 only: clone and install icefall (not pip-installable)
if [[ "$EXTRA" == "x86" ]]; then
    if [ ! -d icefall ]; then
        echo "Cloning icefall..."
        git clone https://github.com/k2-fsa/icefall
        rm -rf icefall/.git
    fi
    uv pip install -r icefall/requirements.txt
    export PYTHONPATH="$(pwd)/icefall:$PYTHONPATH"
    echo "PYTHONPATH updated. Add this to your shell profile to persist:"
    echo "  export PYTHONPATH=\"$(pwd)/icefall:\$PYTHONPATH\""
fi

# Delta-AI only: install flash_attn_3 from pre-built egg (aarch64, Hopper GPUs)
# Package name is flash_attn_3 (not flash_attn); uv does not support .egg — use pip.
# Set FLASH_ATTN_EGG to the path of a pre-built flash_attn_3 .egg file.
if [[ "$EXTRA" == "dai" ]]; then
    FLASH_EGG="${FLASH_ATTN_EGG:-}"
    if [[ -n "$FLASH_EGG" && -f "$FLASH_EGG" ]]; then
        echo "Installing flash_attn_3 from $FLASH_EGG"
        .venv/bin/pip install "$FLASH_EGG"
    else
        echo "NOTE: flash_attn_3 egg not found. Set FLASH_ATTN_EGG to install it."
        echo "Build with: cd /path/to/flash-attention/hopper && python setup.py bdist_egg"
    fi
fi

echo "Setup complete. Environment: .venv (extra=$EXTRA)"
echo "Activate with: source .venv/bin/activate"
