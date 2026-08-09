# Local checkpoints

This directory contains checkpoints created or resumed by local RE-USE/SEMamba training runs.

Use one directory per run:

```text
checkpoints/<run_id>/
├── config.yaml
├── g_<step>.pth
├── do_<step>.pth
└── logs/
```

Externally downloaded weights belong in `pretrained/`. Existing historical run directories are preserved as-is; no checkpoint is deleted merely to simplify the review surface.
