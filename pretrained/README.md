# External pretrained assets

This directory is only for weights and model bundles obtained outside this repository.

```text
pretrained/
├── reuse_hf/        # downloaded nvidia/RE-USE Hugging Face snapshot
└── semamba/         # externally released SEMamba weights
```

Expected legacy SEMamba files include `SEMamba_advanced.pth`, `pretrained_discriminator.pth`, and `vd.pth`. These files are external assets, not results of local RE-USE training.

Large assets are ignored by Git. Do not place locally trained checkpoints here; use `checkpoints/<run_id>/`.
