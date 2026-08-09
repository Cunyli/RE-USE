# Listening examples

This directory is the small, stable listening surface for review.

Each row in `manifest.csv` links one fixed sample ID across:

- `degraded/`: input speech;
- `reference/`: clean reference;
- `enhanced/legacy_unverified/`: inherited enhanced output.

Keep the set to roughly 3–5 representative IDs. Replace files intentionally instead of appending outputs from every epoch or run. Large listening dumps belong with their run artifacts.

The current four enhanced files were inherited from the former `data/enhanced_audio/` directory. Their exact generating checkpoint was not recorded, so the manifest does not claim either the downloaded RE-USE model or a locally trained SEMamba checkpoint.
