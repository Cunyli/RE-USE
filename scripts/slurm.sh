#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/RE-USE}"
TASK="${TASK:-${1:-infer}}"
SEMAMBA_DIR="${SEMAMBA_DIR:-$ROOT_DIR/SEMamba}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-reuse}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
JOB_NAME="${JOB_NAME:-reuse-$TASK}"
PARTITION="${PARTITION:-gpu-a100-80g}"
GPU_TYPE="${GPU_TYPE:-a100}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

export ROOT_DIR TASK SEMAMBA_DIR CONDA_ENV_NAME LOG_DIR

mkdir -p "$LOG_DIR"

submit_self() {
  local script_path="$ROOT_DIR/scripts/slurm.sh"
  local sbatch_args=(
    "--job-name=$JOB_NAME"
    "--partition=$PARTITION"
    "--cpus-per-task=$CPUS_PER_TASK"
    "--mem=$MEMORY"
    "--time=$TIME_LIMIT"
    "--output=$LOG_DIR/slurm_%j.out"
    "--error=$LOG_DIR/slurm_%j.err"
  )

  if [[ -n "$GPU_TYPE" ]]; then
    sbatch_args+=("--gres=gpu:${GPU_TYPE}:${GPUS}")
  else
    sbatch_args+=("--gres=gpu:${GPUS}")
  fi

  echo "Submitting $JOB_NAME"
  sbatch "${sbatch_args[@]}" --export=ALL,TASK="$TASK" "$script_path" "$TASK"
}

load_runtime() {
  module load "$SOFTWARE_STACK_MODULE"
  module load "$COMPILER_MODULE"

  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH" | tee -a "$LIVE_LOG"
    exit 1
  fi

  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV_NAME"

  export CC="$(command -v gcc)"
  export CXX="$(command -v g++)"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

best_reuse_checkpoint() {
  local ckpt_dir="${1:-$SEMAMBA_DIR/exp/reuse_tau_fixed}"

  python - "$ckpt_dir" <<'PY'
import sys
from pathlib import Path

import torch

ckpt_dir = Path(sys.argv[1])
best_step = None

for path in sorted(ckpt_dir.glob("best_g_*.pth"), reverse=True):
    try:
        torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"Skipping unreadable best checkpoint {path}: {exc}", file=sys.stderr)
        continue
    print(path)
    raise SystemExit(0)

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    values = []
    for event in sorted((ckpt_dir / "logs").glob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(event))
        accumulator.Reload()
        if "Validation/PESQ Score" in accumulator.Tags().get("scalars", []):
            values.extend(accumulator.Scalars("Validation/PESQ Score"))
    if values:
        best_step = int(max(values, key=lambda item: item.value).step)
except Exception as exc:
    print(f"Could not read validation PESQ from TensorBoard logs: {exc}", file=sys.stderr)

if best_step is not None:
    candidate = ckpt_dir / f"g_{best_step:08d}.pth"
    if candidate.is_file():
        try:
            torch.load(candidate, map_location="cpu")
            print(candidate)
            raise SystemExit(0)
        except Exception as exc:
            print(f"Skipping best-PESQ checkpoint {candidate}: {exc}", file=sys.stderr)

for path in sorted(ckpt_dir.glob("g_*.pth"), reverse=True):
    try:
        torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"Skipping unreadable checkpoint {path}: {exc}", file=sys.stderr)
        continue
    print(path)
    break
PY
}

run_train() {
  CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/train/semamba_tau_fixed.yaml}"
  EXP_FOLDER="${EXP_FOLDER:-$SEMAMBA_DIR/exp}"
  EXP_NAME="${EXP_NAME:-reuse_tau_fixed}"
  USE_SIMULATION_ROOT="${USE_SIMULATION_ROOT:-/scratch/work/lil14/USE_simulation}"
  REUSE_TAU_FIXED_TRAIN_CSV="${REUSE_TAU_FIXED_TRAIN_CSV:-/scratch/work/lil14/data/TAU/simulated/phone_room/train/paired.csv}"
  REUSE_TAU_FIXED_VALID_CSV="${REUSE_TAU_FIXED_VALID_CSV:-/scratch/work/lil14/data/TAU/simulated/phone_room/valid/paired.csv}"
  export USE_SIMULATION_ROOT REUSE_TAU_FIXED_TRAIN_CSV REUSE_TAU_FIXED_VALID_CSV

  test -d "$SEMAMBA_DIR"
  test -f "$CONFIG_PATH"

  cd "$SEMAMBA_DIR"
  echo "Config: $CONFIG_PATH" | tee -a "$LIVE_LOG"
  python train.py \
    --config "$CONFIG_PATH" \
    --exp_folder "$EXP_FOLDER" \
    --exp_name "$EXP_NAME" \
    2>&1 | tee -a "$LIVE_LOG"
}

run_infer() {
  INPUT_DIR="${INPUT_DIR:-/scratch/work/lil14/data/TAU/simulated/phone_room/test/noisy}"
  OUTPUT_DIR="${OUTPUT_DIR:-/scratch/work/lil14/data/TAU/enhanced/reuse/phone_room/test}"
  CONFIG_PATH="${CONFIG_PATH:-$SEMAMBA_DIR/recipes/SEMamba_advanced/SEMamba_tau_fixed.yaml}"
  PAIR_CSV="${PAIR_CSV:-/scratch/work/lil14/data/TAU/simulated/phone_room/test/paired.csv}"
  CKPT="${CKPT:-}"
  WAV_DIR="$OUTPUT_DIR/wav"

  test -d "$SEMAMBA_DIR"
  test -f "$CONFIG_PATH"
  mkdir -p "$WAV_DIR"

  if ! find "$INPUT_DIR" -type f | grep -q .; then
    echo "No input files found in $INPUT_DIR" | tee -a "$LIVE_LOG"
    exit 1
  fi

  if [[ -z "$CKPT" ]]; then
    CKPT="$(best_reuse_checkpoint "$SEMAMBA_DIR/exp/reuse_tau_fixed")"
  fi
  if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "No RE-USE/SEMamba checkpoint found. Set CKPT or finish training under $SEMAMBA_DIR/exp/reuse_tau_fixed" | tee -a "$LIVE_LOG"
    exit 1
  fi

  echo "SEMAMBA_DIR=$SEMAMBA_DIR" | tee -a "$LIVE_LOG"
  echo "CONFIG_PATH=$CONFIG_PATH" | tee -a "$LIVE_LOG"
  echo "CKPT=$CKPT" | tee -a "$LIVE_LOG"

  python "$SEMAMBA_DIR/inference.py" \
    --input_folder "$INPUT_DIR" \
    --output_folder "$WAV_DIR" \
    --config "$CONFIG_PATH" \
    --checkpoint_file "$CKPT" \
    2>&1 | tee -a "$LIVE_LOG"

  python - "$PAIR_CSV" "$OUTPUT_DIR" <<'PY'
import csv
import sys
from pathlib import Path

pair_csv = Path(sys.argv[1])
out_root = Path(sys.argv[2])
rows = list(csv.DictReader(pair_csv.open()))

with (out_root / "inf.scp").open("w") as inf, (out_root / "ref.scp").open("w") as ref:
    for row in rows:
        uid = row["uid"]
        enhanced = out_root / "wav" / Path(row["noisy_filepath"]).name
        if not enhanced.is_file():
            raise FileNotFoundError(f"Missing enhanced wav for {uid}: {enhanced}")
        inf.write(f"{uid} {enhanced}\n")
        ref.write(f"{uid} {row['clean_filepath']}\n")

print(f"Wrote inf.scp/ref.scp for {len(rows)} utterances")
PY

  find "$WAV_DIR" -maxdepth 1 -type f -name "*.wav" | sort | tee -a "$LIVE_LOG"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  submit_self
  exit 0
fi

cd "$ROOT_DIR"
LIVE_LOG="$LOG_DIR/${TASK}_${SLURM_JOB_ID}.log"
echo "Live log: $LIVE_LOG"
echo "Job ${SLURM_JOB_ID} started at $(date)" | tee -a "$LIVE_LOG"
echo "Task: $TASK" | tee -a "$LIVE_LOG"
echo "Host: $(hostname)" | tee -a "$LIVE_LOG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$LIVE_LOG"

load_runtime

case "$TASK" in
  train)
    run_train
    ;;
  infer)
    run_infer
    ;;
  *)
    echo "Unknown task: $TASK" | tee -a "$LIVE_LOG"
    exit 2
    ;;
esac

echo "Job ${SLURM_JOB_ID} completed at $(date)" | tee -a "$LIVE_LOG"
