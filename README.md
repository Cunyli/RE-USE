# RE-USE on Triton

This repository is the lightweight launcher for reproducing
[`nvidia/RE-USE`](https://huggingface.co/nvidia/RE-USE) on Triton A100 nodes.

It keeps only the scripts you want to sync through Git. Upstream code,
downloaded model files, logs, and audio data stay in ignored local folders.

## Layout

- `scripts/fetch_sources.sh`: clone SEMamba into `third_party/SEMamba`
- `scripts/bootstrap_env.sh`: create the environment, install dependencies,
  and download the RE-USE Hugging Face snapshot into `upstream/RE-USE`
- `scripts/run_interactive.sh`: run inference in an interactive GPU job
- `scripts/reuse_infer.sbatch`: batch job for Triton

Generated local directories:

- `third_party/SEMamba/`
- `upstream/RE-USE/`
- `data/noisy_audio/`
- `data/enhanced_audio/`

## Workflow on Triton

```bash
cd ~/Projects/RE-USE
bash scripts/fetch_sources.sh
bash scripts/bootstrap_env.sh
sbatch scripts/reuse_infer.sbatch
```

For a quick manual test inside an interactive GPU allocation:

```bash
sinteractive --partition=gpu-a100-80g --gpus=a100:1 --time=02:00:00 --mem=32G
cd ~/Projects/RE-USE
bash scripts/run_interactive.sh
```

## Notes

- Triton A100 nodes are `x86_64`, so this follows the x86 environment route.
- Upstream `inference.py` requires a GPU and exits on CPU.
- Upstream still marks `--checkpoint_file` as required, but the script actually
  loads weights with `from_pretrained("nvidia/RE-USE")`. These wrappers pass a
  dummy value.
- If `third_party/SEMamba/mamba_install` fails to build, the bootstrap step
  retries the upstream `mamba-1_2_0_post1` fallback mentioned in SEMamba.
