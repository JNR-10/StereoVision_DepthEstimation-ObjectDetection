#!/usr/bin/env bash
# Simple installer script to set up Miniforge (arm64) and a conda env with Open3D
# Usage: bash scripts/setup_miniforge_env.sh [env_name] [python_version]
# Example: bash scripts/setup_miniforge_env.sh depth 3.11

set -euo pipefail

ENV_NAME=${1:-depth}
PY_VER=${2:-3.11}

echo "==> Checking architecture"
ARCH=$(uname -m)
echo "machine: $ARCH"
if [[ "$ARCH" != "arm64" ]]; then
  echo "Warning: this script is for Apple Silicon (arm64). You are running: $ARCH"
  echo "You may still proceed, but ensure you download the correct installer for your CPU."
fi

MINIFORGE_SH="$HOME/miniforge_install.sh"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"

echo "==> Downloading Miniforge installer to $MINIFORGE_SH"
curl -L -o "$MINIFORGE_SH" "$MINIFORGE_URL"
chmod +x "$MINIFORGE_SH"

echo "==> Running Miniforge installer (batch mode). This will install to ~/miniforge3 by default."
bash "$MINIFORGE_SH" -b -p "$HOME/miniforge3"

CONDA_BIN="$HOME/miniforge3/bin/conda"
if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Conda binary not found at $CONDA_BIN — installer may have failed." >&2
  exit 1
fi

echo "==> Initializing conda for zsh"
"$CONDA_BIN" init zsh >/dev/null

echo "==> To use conda in this shell, either close and re-open your terminal or run:" 
echo "  source \"$HOME/.zshrc\""
echo "Then re-run this script or run the following commands manually to finish setup:"
echo
echo "  # create environment and install open3d"
echo "  $CONDA_BIN create -n $ENV_NAME python=$PY_VER -y"
echo "  source \"$HOME/.zshrc\""
echo "  conda activate $ENV_NAME"
echo "  conda install -c conda-forge open3d -y"
echo
echo "If you want the script to continue automatically, run it after sourcing your shell so 'conda' is on PATH."

echo "==> Finished installer step. Follow the printed commands to create the env and install Open3D."
