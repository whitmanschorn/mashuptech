#!/bin/bash
# One-time environment setup for mashuptech on macOS (Apple Silicon).
# Installs a *native arm64* Python toolchain even if your shell runs under Rosetta,
# so PyTorch can use the GPU (Metal / MPS).
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script targets macOS. On other platforms: install uv, run 'uv sync', install ffmpeg." >&2
  exit 1
fi

NATIVE_ARCH="$(arch -arm64 uname -m 2>/dev/null || uname -m)"
if [[ "$NATIVE_ARCH" != "arm64" ]]; then
  echo "Warning: this Mac is not Apple Silicon; separation will run on CPU." >&2
fi
run_native() { if [[ "$NATIVE_ARCH" == "arm64" ]]; then arch -arm64 "$@"; else "$@"; fi; }

# 1. uv (native binary)
UV="$HOME/.local/bin/uv"
if [[ ! -x "$UV" ]]; then
  say "Installing uv (native)"
  curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 run_native sh
fi
say "uv: $("$UV" --version) [$(file -b "$UV" | grep -o 'arm64\|x86_64' | head -1)]"

# 2. Python 3.11 + venv + deps
say "Installing Python 3.11 and dependencies (first run downloads ~1 GB)"
run_native "$UV" python install 3.11
run_native "$UV" sync --extra dev
PY=".venv/bin/python"
say "python: $($PY -c 'import platform,sys; print(sys.version.split()[0], platform.machine())')"

# 3. ffmpeg comes bundled via the imageio-ffmpeg package (native arm64 build with libmp3lame),
#    so no Homebrew is needed.
say "ffmpeg: $($PY -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"

# 4. Pre-download the separation model so the first overnight run doesn't stall
say "Warming up model download"
$PY -m mashuptech warmup

say "Done. Start the pipeline with: scripts/run.sh"
