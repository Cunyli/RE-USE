#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEMAMBA_DIR="${SEMAMBA_DIR:-${ROOT_DIR}/third_party/SEMamba}"
UPSTREAM_DIR="${UPSTREAM_DIR:-${ROOT_DIR}/upstream/RE-USE}"
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

if [ ! -d "${SEMAMBA_DIR}" ]; then
  echo "Missing ${SEMAMBA_DIR}. Run scripts/fetch_sources.sh first."
  exit 1
fi

${PKG_TOOL} create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" pip setuptools wheel

eval "$(${ACTIVATE_TOOL} shell.bash hook)"
${ACTIVATE_TOOL} activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install torch==2.2.2 torchaudio==2.2.2 --index-url "${TORCH_INDEX_URL}"
python -m pip install packaging librosa soundfile pyyaml argparse tensorboard pesq einops huggingface_hub resampy

pushd "${SEMAMBA_DIR}/mamba_install" >/dev/null
if ! python -m pip install .; then
  popd >/dev/null
  echo "Primary mamba_install failed, retrying upstream fallback."
  pushd "${SEMAMBA_DIR}/mamba-1_2_0_post1" >/dev/null
  python -m pip install .
fi
popd >/dev/null

mkdir -p "$(dirname "${UPSTREAM_DIR}")" "${ROOT_DIR}/data/noisy_audio" "${ROOT_DIR}/data/enhanced_audio"

if [ ! -d "${UPSTREAM_DIR}" ]; then
  huggingface-cli download nvidia/RE-USE \
    --local-dir "${UPSTREAM_DIR}" \
    --local-dir-use-symlinks False
fi

python - <<'PY'
import torch
import torchaudio
print("torch", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("cuda_available", torch.cuda.is_available())
PY

echo "Environment '${ENV_NAME}' is ready."
echo "Upstream RE-USE snapshot: ${UPSTREAM_DIR}"
