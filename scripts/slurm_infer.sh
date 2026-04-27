#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/RE-USE}"
UPSTREAM_DIR="${UPSTREAM_DIR:-${ROOT_DIR}/RE-USE}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-reuse}"
INPUT_DIR="${INPUT_DIR:-${ROOT_DIR}/data/noisy_audio}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/data/enhanced_audio}"
CONFIG_PATH="${CONFIG_PATH:-${UPSTREAM_DIR}/recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml}"

cd "$ROOT_DIR"

module load mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

test -d "$UPSTREAM_DIR"
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

if ! find "$INPUT_DIR" -type f | grep -q .; then
  echo "No input files found in $INPUT_DIR"
  exit 1
fi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_name:", torch.cuda.get_device_name(0))
PY

test -f "$UPSTREAM_DIR/inference.py"
test -f "$CONFIG_PATH"

python "$UPSTREAM_DIR/inference.py" \
  --input_folder "$INPUT_DIR" \
  --output_folder "$OUTPUT_DIR" \
  --config "$CONFIG_PATH" \
  --checkpoint_file dummy

echo "Enhanced files:"
find "$OUTPUT_DIR" -maxdepth 1 -type f | sort
