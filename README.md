# RE-USE

RE-USE is a lightweight launcher for SEMamba-based speech-enhancement experiments in the local USE research workspace. The repository owns the experiment configs, USE Simulation adapter, SEMamba overlays, and launch scripts. Downloaded source snapshots and generated research artifacts stay outside Git.

## At a glance

| Purpose | Location |
| --- | --- |
| Training configs | `configs/train/` |
| Local SEMamba adaptations | `overlays/SEMamba/` |
| Reusable launch/setup scripts | `scripts/` |
| Downloaded upstream source | `upstream/SEMamba/` |
| Downloaded external weights/assets | `pretrained/` |
| Locally trained checkpoints | `checkpoints/<run_id>/` |
| W&B and other run records | `runs/` |
| Small fixed listening set | `outputs/examples/` |
| Logs and temporary files | `logs/`, `tmp/` |

Generated artifacts are ignored by Git. Only the small listening set and the README files that explain artifact directories are tracked.

## Setup

```bash
bash scripts/fetch_sources.sh
bash scripts/apply_semamba_overrides.sh
bash scripts/setup_env.sh
```

Re-run `scripts/apply_semamba_overrides.sh` after refreshing the SEMamba checkout.

## Training and inference

```bash
bash scripts/slurm.sh train
bash scripts/slurm.sh infer
```

Common overrides:

```bash
CONFIG_PATH=configs/train/semamba_tau_fixed.yaml \
EXP_NAME=my_run \
bash scripts/slurm.sh train

CKPT=checkpoints/my_run/g_00006000.pth \
OUTPUT_DIR=/path/to/enhanced \
bash scripts/slurm.sh infer
```

Data locations are selected through the config files or environment variables. Datasets are not stored in this repository.

## Checkpoint boundary

- `pretrained/reuse_hf/` contains the downloaded NVIDIA RE-USE Hugging Face snapshot, including `model.safetensors` and its inference code.
- `pretrained/semamba/` contains externally released SEMamba weights.
- `checkpoints/<run_id>/` contains locally trained or resumed checkpoints.

Do not mix these categories. See `pretrained/README.md` and `checkpoints/README.md` for the directory contract.

## Listening samples

`outputs/examples/` is the only review-facing listening set. It contains four fixed degraded/reference/enhanced triplets and a manifest. The historical enhanced files did not record their generating checkpoint, so they are explicitly marked `legacy_unverified`.

Large or repeated outputs remain with their experiment and are not part of the review-facing demo. No historical checkpoints, logs, W&B runs, or upstream demo samples were deleted during normalization.

## Current limitations

- This launcher depends on the external USE Simulation and AVQI evaluation workspaces on Triton.
- The four inherited listening examples have complete audio triplets but incomplete checkpoint provenance.
- The ignored `.local_git_backup/` directory is a preserved historical backup, not active source code.
