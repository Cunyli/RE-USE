#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-reuse-triton}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile >/dev/null 2>&1 || true
fi

if command -v module >/dev/null 2>&1; then
  module load mamba >/dev/null 2>&1 || true
fi

if ! command -v mamba >/dev/null 2>&1 && ! command -v conda >/dev/null 2>&1; then
  echo "Neither mamba nor conda is available."
  echo "On Triton, try: module load mamba"
  exit 1
fi

PKG_TOOL="mamba"
if ! command -v mamba >/dev/null 2>&1; then
  PKG_TOOL="conda"
fi

ACTIVATE_TOOL="conda"
if ! command -v conda >/dev/null 2>&1; then
  ACTIVATE_TOOL="${PKG_TOOL}"
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required."
  exit 1
fi

if [ ! -d "${ROOT_DIR}/SEMamba" ]; then
  echo "Missing ${ROOT_DIR}/SEMamba. Run scripts/fetch_sources.sh first."
  exit 1
fi

${PKG_TOOL} create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" pip setuptools wheel

eval "$(${ACTIVATE_TOOL} shell.bash hook)"
${ACTIVATE_TOOL} activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install torch==2.2.2 torchaudio==2.2.2 --index-url "${TORCH_INDEX_URL}"
python -m pip install packaging librosa soundfile pyyaml argparse tensorboard pesq einops huggingface_hub resampy

pushd "${ROOT_DIR}/SEMamba/mamba_install" >/dev/null
if ! python -m pip install .; then
  popd >/dev/null
  echo "Primary mamba_install failed, retrying upstream fallback."
  pushd "${ROOT_DIR}/SEMamba/mamba-1_2_0_post1" >/dev/null
  python -m pip install .
fi
popd >/dev/null

if [ ! -d "${ROOT_DIR}/RE-USE" ]; then
  huggingface-cli download nvidia/RE-USE \
    --local-dir "${ROOT_DIR}/RE-USE" \
    --local-dir-use-symlinks False
fi

mkdir -p "${ROOT_DIR}/RE-USE/noisy_audio" "${ROOT_DIR}/RE-USE/enhanced_audio"

python - <<'PY'
import torch
import torchaudio
print("torch", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("cuda_available", torch.cuda.is_available())
PY

echo "Environment '${ENV_NAME}' is ready."
