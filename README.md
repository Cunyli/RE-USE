# RE-USE Launcher

This repository is a lightweight launcher for running RE-USE/SEMamba
experiments on Triton. Upstream source checkouts, checkpoints, logs, W&B runs,
caches, and audio data are local runtime artifacts and are ignored by git.

## Layout

- `scripts/fetch_sources.sh`: clone SEMamba into local `SEMamba/`.
- `scripts/apply_semamba_overrides.sh`: apply the local USE_simulation adapter,
  SEMamba file overlays, and config.
- `scripts/setup_env.sh`: create the conda environment and install runtime
  dependencies, then download the RE-USE Hugging Face snapshot into local
  `RE-USE/`.
- `scripts/slurm.sh`: generic Slurm entry point for `train` and `infer`.
- `configs/train/`: launcher-owned training configs.
- `overlays/`: files copied into the local SEMamba checkout.

Ignored local directories include `SEMamba/`, `RE-USE/`, `data/`, `logs/`,
`outputs/`, `runs/`, `wandb/`, checkpoint directories, model weights, and
source-tree caches.

Repository conventions are kept in this README, `.gitignore`, `configs/`, and
`scripts/`; local agent notes such as `AGENTS.md` or `docs/` are intentionally
ignored.

## Setup

```bash
bash scripts/fetch_sources.sh
bash scripts/apply_semamba_overrides.sh
bash scripts/setup_env.sh
```

Re-run `scripts/apply_semamba_overrides.sh` after refreshing the local SEMamba
checkout.

## Slurm

Use generic task names and choose data through config/environment variables:

```bash
bash scripts/slurm.sh train
bash scripts/slurm.sh infer
```

Common overrides:

```bash
CONFIG_PATH=configs/train/semamba_tau_fixed.yaml bash scripts/slurm.sh train
OUTPUT_DIR=/path/to/enhanced CKPT=/path/to/g_00006000.pth bash scripts/slurm.sh infer
```

TAU fixed defaults are kept in `scripts/slurm.sh` for the local Triton workflow.
Portable config defaults use placeholder paths or environment variables.

## USE Simulation

The local SEMamba checkout should support fixed paired manifests exported by
USE_simulation:

```yaml
data_cfg:
  dataset_type: use_simulation_fixed
  use_simulation_root: ${USE_SIMULATION_ROOT:-../USE_simulation}
  train_pair_manifest: ${REUSE_TAU_FIXED_TRAIN_CSV:-/path/to/train/paired.csv}
  valid_pair_manifest: ${REUSE_TAU_FIXED_VALID_CSV:-/path/to/valid/paired.csv}
```

Inference writes ABQI/ABQY-style outputs:

```text
<output_dir>/wav/
<output_dir>/inf.scp
<output_dir>/ref.scp
```
