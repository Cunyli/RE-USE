# RE-USE on Triton

This directory contains a no-Docker reproduction path for running
[`nvidia/RE-USE`](https://huggingface.co/nvidia/RE-USE) on Triton A100 nodes.

## Why this route

Triton A100 nodes are x86_64, but Docker adds extra friction on a cluster.
The simpler path is:

1. create a clean Python environment
2. install PyTorch 2.2.2
3. install the official local `mamba_install` from SEMamba
4. download the RE-USE Hugging Face repo
5. run inference in a Slurm job

## Files

- `scripts/bootstrap_env.sh`: create the environment and install dependencies
- `scripts/fetch_sources.sh`: clone SEMamba and download RE-USE
- `scripts/run_interactive.sh`: quick interactive inference
- `scripts/reuse_infer.sbatch`: batch job for inference on Triton

## Expected workflow on Triton

```bash
cd /path/to/reuse-triton
bash scripts/fetch_sources.sh
bash scripts/bootstrap_env.sh
sbatch scripts/reuse_infer.sbatch
```

## Notes

- The RE-USE `inference.py` requires a GPU and exits on CPU.
- The `--checkpoint_file` argument is still marked required in upstream code,
  but the script actually pulls weights with `from_pretrained("nvidia/RE-USE")`.
  This setup passes a dummy value.
- If building `mamba_install` fails, the bootstrap script retries with the
  upstream `mamba-1_2_0_post1` fallback suggested by SEMamba.
