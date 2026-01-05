from __future__ import annotations

import os
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
DEFAULT_IMG_SIZE: int = 512
DEFAULT_LOSS_WEIGHTS: tuple[float, float, float, float, float] = (0.1, 0.1, 0.1, 0.2, 0.5)

WB = os.getenv("DINO_WB")
WL = os.getenv("DINO_WL")
WL_plus = os.getenv("DINO_WL_plus")

DEFAULT_DINO_REPO_DIR: str = os.getenv("DEFAULT_DINO_REPO_DIR")
DEFAULT_DATA_ROOT: str = os.getenv("DEFAULT_DATA_ROOT")
DINO_WEIGHTS_PATH: str | None = os.getenv("DINO_WEIGHTS_PATH")

DEFAULT_MODEL_SIZE: str = "b"
DEFAULT_PLUS: str = ""
    
DEFAULTS: dict[str, Any] = dict(
    cv_params=dict(mode="gkf", cv_seed=0, n_splits=5, max_folds=None),
    cv_resume=True,
    max_folds=None,
    device="cuda",
    verbose=False,
    epochs=80,
    batch_size=124,
    wd=1e-4,
    lr_start=3e-4,
    lr_final=1e-7,
    early_stopping=10,
    head_hidden=2048,
    head_drop=0.1,
    head_depth=4,
    num_neck=1,
    swa_epochs=10,
    swa_lr_start=None,
    swa_lr_final=None,
    swa_anneal_epochs=0,
    swa_load_best=True,
    swa_eval_freq=1,
    clip_val=1.0,
    n_models=1,
    val_freq=1,
    tau_physics=0.0,
    comet_exp_name="csiro",
    img_size=DEFAULT_IMG_SIZE,
    bcs_range=(0.2, 0.4),
    hue_range=(0.02, 0.08),
    tiled_inp=True,
    tile_swap=False,
    lnk_params=dict(k=0, warm_up_n=0, lr_max=1e-5, lr_min=1e-7, wd=0.0),
    backbone_dtype="fp16",
    trainable_dtype="fp16",
    save_output_dir="/notebooks/kaggle/csiro/output"
)

def default_num_workers(reserve: int = 2) -> int:
    import os

    n = (os.cpu_count() or 0) - int(reserve)
    return max(0, n)


def dino_weights_path(*, repo_dir: str | None, model_size: str, plus: str) -> str:
    import os

    if not repo_dir:
        raise ValueError("Set DINO_WEIGHTS_PATH in config/env or pass dino_repo.")
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


