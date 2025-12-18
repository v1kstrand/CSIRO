from __future__ import annotations

from typing import Any, Sequence


TARGETS: tuple[str, ...] = ("Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g")
IDX_COLS: tuple[str, ...] = (
    "image_path",
    "Sampling_Date",
    "State",
    "Species",
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
)

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

DEFAULT_SEED: int = 420
DEFAULT_IMG_SIZE: int = 512
DEFAULT_LOSS_WEIGHTS: tuple[float, float, float, float, float] = (0.1, 0.1, 0.1, 0.2, 0.5)

WB = "https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoidW84aXJvdGQyeThwcGpuNXFveGthZTE4IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjU5NzI4MTd9fX1dfQ__&Signature=H5H5kLVc6V83i-s2euNHx6t9KlVeG27QKX6qtkXNiLwEzuCshJD4RfwUbQv8oBJOZXPezAVJZPRkYRdsb4jh-LQ72DZtEuNkjNKHf7Pn57wzee0bjEYjWdJmOqK4waaSe9TQqELM%7EPgzdAT4LCSHYcFQ%7EleRnHGGGJiHBmTd6e1xZYhvUCfkvVD1TG-zM7R0-P%7EMLetHMvWl%7EUapCMYthsWqZctsYAQKUQxsLrly8Y4EaM8hm5nowpArPZC4myNO1iiXld5Hc3t9CVLEdYT7LIct0x6cf3-B-6WOgxGb7LdLPCcZPPfoGgX3KGtTAgNQYOpGFs-hgILFHRKVOJ7T3A__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1893388161261111"
WL = "https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoidW84aXJvdGQyeThwcGpuNXFveGthZTE4IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjU5NzI4MTd9fX1dfQ__&Signature=H5H5kLVc6V83i-s2euNHx6t9KlVeG27QKX6qtkXNiLwEzuCshJD4RfwUbQv8oBJOZXPezAVJZPRkYRdsb4jh-LQ72DZtEuNkjNKHf7Pn57wzee0bjEYjWdJmOqK4waaSe9TQqELM%7EPgzdAT4LCSHYcFQ%7EleRnHGGGJiHBmTd6e1xZYhvUCfkvVD1TG-zM7R0-P%7EMLetHMvWl%7EUapCMYthsWqZctsYAQKUQxsLrly8Y4EaM8hm5nowpArPZC4myNO1iiXld5Hc3t9CVLEdYT7LIct0x6cf3-B-6WOgxGb7LdLPCcZPPfoGgX3KGtTAgNQYOpGFs-hgILFHRKVOJ7T3A__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1893388161261111"
WL_plus = "https://dinov3.llamameta.net/dinov3_vith16plus/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoidW84aXJvdGQyeThwcGpuNXFveGthZTE4IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjU5NzI4MTd9fX1dfQ__&Signature=H5H5kLVc6V83i-s2euNHx6t9KlVeG27QKX6qtkXNiLwEzuCshJD4RfwUbQv8oBJOZXPezAVJZPRkYRdsb4jh-LQ72DZtEuNkjNKHf7Pn57wzee0bjEYjWdJmOqK4waaSe9TQqELM%7EPgzdAT4LCSHYcFQ%7EleRnHGGGJiHBmTd6e1xZYhvUCfkvVD1TG-zM7R0-P%7EMLetHMvWl%7EUapCMYthsWqZctsYAQKUQxsLrly8Y4EaM8hm5nowpArPZC4myNO1iiXld5Hc3t9CVLEdYT7LIct0x6cf3-B-6WOgxGb7LdLPCcZPPfoGgX3KGtTAgNQYOpGFs-hgILFHRKVOJ7T3A__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1893388161261111"

DEFAULT_DINO_REPO_DIR: str = "/notebooks/dinov3"
DEFAULT_DATA_ROOT: str = "/notebooks/kaggle/csiro"
DEFAULT_MODEL_SIZE: str = "b"
DEFAULT_PLUS: str = ""
DEFAULT_DTYPE_STR: str = "bf16"  # fp16|bf16|fp32

# Script/experiment defaults (matches current notebook settings)
DEFAULTS: dict[str, Any] = dict(
    seed=DEFAULT_SEED,
    img_size=DEFAULT_IMG_SIZE,
    epochs=80,
    batch_size=64,
    wd=3e-3,
    lr_start=3e-4,
    lr_final=1e-7,
    early_stopping=15,
    head_hidden=2048,
    head_drop=0.1,
    head_depth=5,
    num_neck=0,
    swa_epochs=15,
    swa_lr=None,
    swa_anneal_epochs=20,
    swa_load_best=True,
    swa_eval_freq=2,
    dtype=DEFAULT_DTYPE_STR,
    clip_val=3.0,
    n_splits=5,
    group_col="Sampling_Date",
    stratify_col="State",
    comet_project=None,
    device="cuda",
    verbose=True,
)

# Sweep definitions with transform choice by name (avoids circular import)
SWEEPS: list[dict[str, object]] = [
    dict(num_neck=1, head_depth=4, tfms="tfms0"),
    dict(num_neck=1, head_depth=5, tfms="tfms0"),
    dict(num_neck=2, head_depth=4, tfms="tfms0"),
    dict(num_neck=2, head_depth=5, tfms="tfms0"),
]

def default_num_workers(reserve: int = 2) -> int:
    import os

    n = (os.cpu_count() or 0) - int(reserve)
    return max(0, n)


def as_tuple_str(xs: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(x) for x in xs)


def dino_weights_path(*, repo_dir: str, model_size: str, plus: str) -> str:
    import os

    return os.path.join(repo_dir, "weights", f"dinov3_vit{model_size}16_pretrain{plus}.pth")


def dino_hub_name(*, model_size: str, plus: str) -> str:
    return f"dinov3_vit{model_size}16{plus.replace('_', '')}"


def parse_dtype(dtype: str):
    import torch

    s = str(dtype).strip().lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")
