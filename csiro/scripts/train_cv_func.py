from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from csiro.amp import set_dtype
from csiro.config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DINO_REPO_DIR,
    DEFAULT_MODEL_SIZE,
    DEFAULT_PLUS,
    DEFAULTS,
    SWEEPS,
    dino_hub_name,
    dino_weights_path,
    parse_dtype,
)
from csiro.data import BiomassBaseCached, load_train_wide
from csiro.train import run_groupkfold_cv

def train_cv(
    *,
    csv: str | None = None,
    root: str = DEFAULT_DATA_ROOT,
    dino_repo: str = DEFAULT_DINO_REPO_DIR,
    dino_weights: str | None = None,
    model_size: str = DEFAULT_MODEL_SIZE,  # "b" == ViT-Base
    plus: str = DEFAULT_PLUS,
    tfms = None,
    overrides: dict[str, Any] | None = None,
    plot_imgs: bool =  False,
    sweeps: dict = None
) -> Any:
    
    cfg: dict[str, Any] = dict(DEFAULTS)
    dtype_t = parse_dtype(cfg["dtype"])
    set_dtype(dtype_t)

    device = str(cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    if csv is None:
        csv = os.path.join(root, "train.csv")
    if dino_weights is None:
        dino_weights = dino_weights_path(repo_dir=dino_repo, model_size=model_size, plus=plus)

    sys.path.insert(0, str(dino_repo))
    wide_df = load_train_wide(str(csv), root=str(root))
    dataset = BiomassBaseCached(wide_df, img_size=int(cfg["img_size"]))

    backbone = torch.hub.load(
        str(dino_repo),
        dino_hub_name(model_size=str(model_size), plus=str(plus)),
        source="local",
        weights=str(dino_weights) ,
    )

    base_kwargs = dict(cfg)
    base_kwargs.pop("dtype", None)
    base_kwargs.update(
        dict(
            dataset=dataset,
            wide_df=wide_df,
            backbone=backbone,
            device=device,
            tfms=tfms,
            plot_imgs=plot_imgs,
        )
     )
    
    if overrides:
        base_kwargs.update(overrides)
        base_kwargs.pop("dtype", None)

    outputs: list[dict[str, Any]] = []
    for sweep in sweeps or SWEEPS:
        kwargs = dict(base_kwargs)
        kwargs.update({k: v for k, v in sweep.items()})
        kwargs["config_name"] = str(sweep)

        result = run_groupkfold_cv(return_details=True, **kwargs)
        outputs.append(
            dict(
                config_name=kwargs["config_name"],
                fold_model_scores=result["fold_model_scores"],
                fold_scores=result["fold_scores"],
                mean=result["mean"],
                std=result["std"],
                states=result["states"],
            )
        )

    return outputs
