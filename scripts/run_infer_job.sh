#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/RE-USE}"
PARTITION="${PARTITION:-gpu-a100-80g}"
GPU_TYPE="${GPU_TYPE:-a100}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
JOB_NAME="${JOB_NAME:-reuse-infer}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-reuse}"
INPUT_DIR="${INPUT_DIR:-${ROOT_DIR}/data/noisy_audio}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/data/enhanced_audio}"
UPSTREAM_DIR="${UPSTREAM_DIR:-${ROOT_DIR}/RE-USE}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"

mkdir -p "${LOG_DIR}"

SBATCH_ARGS=(
  "--job-name=${JOB_NAME}"
  "--partition=${PARTITION}"
  "--cpus-per-task=${CPUS_PER_TASK}"
  "--mem=${MEMORY}"
  "--time=${TIME_LIMIT}"
  "--output=${LOG_DIR}/slurm_%j.out"
)

if [[ -n "${GPU_TYPE}" ]]; then
  SBATCH_ARGS+=("--gres=gpu:${GPU_TYPE}:${GPUS}")
else
  SBATCH_ARGS+=("--gres=gpu:${GPUS}")
fi

echo "Submitting RE-USE inference job:"
printf '  %q\n' sbatch "${SBATCH_ARGS[@]}" \
  --export=ALL,ROOT_DIR="${ROOT_DIR}",CONDA_ENV_NAME="${CONDA_ENV_NAME}",INPUT_DIR="${INPUT_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",UPSTREAM_DIR="${UPSTREAM_DIR}" \
  "${ROOT_DIR}/scripts/slurm_infer.sh"

sbatch "${SBATCH_ARGS[@]}" \
  --export=ALL,ROOT_DIR="${ROOT_DIR}",CONDA_ENV_NAME="${CONDA_ENV_NAME}",INPUT_DIR="${INPUT_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",UPSTREAM_DIR="${UPSTREAM_DIR}" \
  "${ROOT_DIR}/scripts/slurm_infer.sh"
