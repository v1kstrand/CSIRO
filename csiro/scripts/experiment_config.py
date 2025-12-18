from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from csiro.config import DEFAULT_IMG_SIZE, DEFAULT_SEED
from csiro.transforms import get_tfms_0

# Central place to tweak training defaults for the CLI scripts.

DEFAULTS = dict(
    seed=DEFAULT_SEED,
    img_size=DEFAULT_IMG_SIZE,
    epochs=80,
    batch_size=64,
    wd=3e-3,
    lr_start=3e-4,
    lr_final=5e-5,
    early_stopping=15,
    head_hidden=2048,
    head_depth=5,
    head_drop=0.1,
    num_neck=0,
    swa_epochs=20,
    swa_lr=None,
    swa_anneal_epochs=15,
    swa_load_best=False,
    swa_eval_freq=2,
    dtype="bf16",  # fp16|bf16|fp32
)

SWEEPS = [
    dict(num_neck=1, head_depth=4, tfms_fn=get_tfms_0),
    dict(num_neck=1, head_depth=5, tfms_fn=get_tfms_0),
    dict(num_neck=2, head_depth=4, tfms_fn=get_tfms_0),
    dict(num_neck=2, head_depth=5, tfms_fn=get_tfms_0),
]
