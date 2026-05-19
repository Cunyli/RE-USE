#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEMAMBA_DIR="${SEMAMBA_DIR:-${ROOT_DIR}/SEMamba}"

if [[ ! -d "$SEMAMBA_DIR" ]]; then
  echo "Missing SEMamba checkout at $SEMAMBA_DIR. Run scripts/fetch_sources.sh first."
  exit 1
fi

cp "$ROOT_DIR/overlays/SEMamba/dataloaders/dataloader_use_simulation.py" \
  "$SEMAMBA_DIR/dataloaders/dataloader_use_simulation.py"
cp "$ROOT_DIR/overlays/SEMamba/train.py" \
  "$SEMAMBA_DIR/train.py"
mkdir -p "$SEMAMBA_DIR/models"
cp "$ROOT_DIR/overlays/SEMamba/models/discriminator.py" \
  "$SEMAMBA_DIR/models/discriminator.py"
cp "$ROOT_DIR/overlays/SEMamba/utils/util.py" \
  "$SEMAMBA_DIR/utils/util.py"

mkdir -p "$SEMAMBA_DIR/recipes/SEMamba_advanced"
cp "$ROOT_DIR/configs/train/semamba_tau_fixed.yaml" \
  "$SEMAMBA_DIR/recipes/SEMamba_advanced/SEMamba_tau_fixed.yaml"

echo "Applied SEMamba USE_simulation overrides."
