#!/bin/bash
#SBATCH --job-name=reuse-infer
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Projects/RE-USE}"
UPSTREAM_DIR="${UPSTREAM_DIR:-${ROOT_DIR}/upstream/RE-USE}"
ENV_NAME="${ENV_NAME:-reuse-triton}"
INPUT_DIR="${INPUT_DIR:-${ROOT_DIR}/data/noisy_audio}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/data/enhanced_audio}"
CONFIG_PATH="${CONFIG_PATH:-${UPSTREAM_DIR}/recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml}"

module load mamba

ACTIVATE_TOOL="conda"
if ! command -v conda >/dev/null 2>&1; then
  ACTIVATE_TOOL="mamba"
fi

eval "$(${ACTIVATE_TOOL} shell.bash hook)"
${ACTIVATE_TOOL} activate "${ENV_NAME}"

if [ ! -d "${UPSTREAM_DIR}" ]; then
  echo "Missing ${UPSTREAM_DIR}. Run scripts/bootstrap_env.sh first."
  exit 1
fi

mkdir -p "${INPUT_DIR}" "${OUTPUT_DIR}"

cd "${UPSTREAM_DIR}"

nvidia-smi
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY

python inference.py \
  --input_folder "${INPUT_DIR}" \
  --output_folder "${OUTPUT_DIR}" \
  --config "${CONFIG_PATH}" \
  --checkpoint_file dummy
