#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "${ROOT_DIR}/SEMamba/.git" ]; then
  git clone https://github.com/RoyChao19477/SEMamba.git "${ROOT_DIR}/SEMamba"
fi

echo "SEMamba is ready under ${ROOT_DIR}."
echo "Run scripts/bootstrap_env.sh next to install dependencies and download RE-USE."
