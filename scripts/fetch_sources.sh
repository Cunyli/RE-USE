#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEMAMBA_DIR="${SEMAMBA_DIR:-${ROOT_DIR}/SEMamba}"

mkdir -p "$(dirname "${SEMAMBA_DIR}")" "${ROOT_DIR}/data/noisy_audio" "${ROOT_DIR}/data/enhanced_audio"

if [ ! -d "${SEMAMBA_DIR}/.git" ]; then
  git clone https://github.com/RoyChao19477/SEMamba.git "${SEMAMBA_DIR}"
fi

echo "SEMamba is ready at ${SEMAMBA_DIR}."
echo "Input audio: ${ROOT_DIR}/data/noisy_audio"
echo "Run scripts/bootstrap_env.sh next to install dependencies and download RE-USE."
