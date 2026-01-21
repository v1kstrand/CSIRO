from __future__ import annotations

import os
from pathlib import Path
from typing import Any


TARGETS: list[str] = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
IDX_COLS: list[str] = [
    "image_path",
    "Sampling_Date",
    "State",
    "Species",
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
]

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

DEFAULT_SEED: int = 420
DEFAULT_LOSS_WEIGHTS: tuple[float, float, float, float, float] = (0.1, 0.1, 0.1, 0.2, 0.5)

_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT: str = os.getenv("DEFAULT_DATA_ROOT")
DEFAULT_DINO_ROOT: str = str(_REPO_ROOT / "_dinov3")
DINO_B_WEIGHTS_PATH: str | None = os.getenv("DINO_B_WEIGHTS_PATH")
DINO_L_WEIGHTS_PATH: str | None = os.getenv("DINO_L_WEIGHTS_PATH")



def dino_weights_path_from_size(backbone_size: str) -> str | None:
    s = str(backbone_size).strip().lower()
    if s == "b":
        return DINO_B_WEIGHTS_PATH
    if s == "l":
        return DINO_L_WEIGHTS_PATH
    raise ValueError(f"Unknown backbone_size: {backbone_size}")


def neck_num_heads_for(backbone_size: str) -> int:
    s = str(backbone_size).strip().lower()
    if s == "b":
        return 12
    if s == "l":
        return 16
    raise ValueError(f"Unknown backbone_size: {backbone_size}")
    
DEFAULTS: dict[str, Any] = dict(
    cv_cfg=1,
    cv_resume=True,
    max_folds=None,
    device="cuda",  # [cuda, cpu]
    verbose=False,
    epochs=80,
    warmup_steps=0,
    batch_size=124,
    wd=1e-4,
    lr_start=3e-4,
    lr_final=1e-7,
    early_stopping=10,
    backbone_size="l",  # [b, l]
    run_name="default",
    model_name="tiled_stitch",  # [tiled_base, tiled_sum3, tiled_stitch, rect_full]
    head_style="single",  # [single, multi]
    out_format="cat_cls_w_mean",  # [mean, cat_cls, cat_cls_w_mean]
    pred_space="log",  # [log, gram]
    tile_geom_mode="shared",  # [shared, independent]
    neck_rope=True,
    rope_rescale=None, #NOTE
    neck_drop=0.0,
    drop_path=None,
    loss_weights=(1.0, 1.0, 1.0, 0.0, 0.0),
    huber_beta=5.0,
    tau_neg=0.0,
    tau_patch=0.0, #NOTE
    head_hidden=2048,
    head_drop=0.1,
    head_depth=4,
    num_neck=1,
    neck_ffn=True,
    neck_pool=False,
    norm_bb_out=False,
    top_k_weights=0,
    last_k_weights=0,
    k_weights_val=False,
    val_min_score=0.0, #NOTE
    val_num_retry=1,
    clip_val=1.0,
    n_models=1,
    val_freq=1,
    cutout=0.0,
    to_gray=0.0,
    mixup=dict(p=0.0, p3=0.0, alpha=0.2),
    val_bs=None,
    comet_exp_name="csiro",
    img_size=512,
    img_preprocess=True,
    bcs_range=(0.2, 0.4),
    hue_range=(0.02, 0.08), 
    tiled_inp=True,
    backbone_dtype="fp16",  # [fp16, bf16, fp32]
    trainable_dtype="fp16",  # [fp16, bf16, fp32]
    save_output_dir="/notebooks/kaggle/csiro/output"
)

def default_num_workers(reserve: int = 3) -> int:
    import os

    n = (os.cpu_count() or 0) - int(reserve)
    return max(0, n)


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


