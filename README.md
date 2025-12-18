# CSIRO Biomass (Kaggle) module repo

This repo packages the core training code from `kaggle_CSIRO.ipynb` into a small Python module.

## Install (local)

```bash
pip install -e .
```

## Key knobs

- AMP dtype: set `DTYPE` (and optional GradScaler is enabled automatically for fp16).
  - In code: `from csiro.amp import DTYPE, set_dtype`
  - Typical:
    - T4: `torch.float16`
    - A100/H100: `torch.bfloat16` is often fine
- Script defaults (epochs, LR, SWA, neck blocks): edit `csiro/scripts/experiment_config.py`.

## Run grouped CV (script)

```bash
python csiro/scripts/train_cv.py ^
  --csv /path/to/train.csv ^
  --root /path/to/dataset/root ^
  --dino-repo /path/to/dinov3 ^
  --dino-weights /path/to/dinov3_vitb16_pretrain.pth
```

Notes:
- DINOv3 weights/repo are not included; pass paths via CLI.
- The provided notebook (`kaggle_CSIRO.ipynb`) can be updated to import from this package.
- `comet-ml` logging is optional: install it only if you pass `--comet-project` (or `comet_exp_name=...` in code).
