#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-reuse-triton}"
INPUT_DIR="${INPUT_DIR:-${ROOT_DIR}/RE-USE/noisy_audio}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/RE-USE/enhanced_audio}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/RE-USE/recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml}"
BWE="${BWE:-}"

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile >/dev/null 2>&1 || true
fi

if command -v module >/dev/null 2>&1; then
  module load mamba >/dev/null 2>&1 || true
fi

ACTIVATE_TOOL="conda"
if ! command -v conda >/dev/null 2>&1; then
  ACTIVATE_TOOL="mamba"
fi

eval "$(${ACTIVATE_TOOL} shell.bash hook)"
${ACTIVATE_TOOL} activate "${ENV_NAME}"

cd "${ROOT_DIR}/RE-USE"

CMD=(
  python inference.py
  --input_folder "${INPUT_DIR}"
  --output_folder "${OUTPUT_DIR}"
  --config "${CONFIG_PATH}"
  --checkpoint_file dummy
)

if [ -n "${BWE}" ]; then
  CMD+=(--BWE "${BWE}")
fi

"${CMD[@]}"
