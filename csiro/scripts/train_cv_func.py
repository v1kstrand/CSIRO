from __future__ import annotations

import os
import sys
import uuid
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
from csiro.transforms import get_tfms


def _tfms_from_name(name: str):
    if name == "default":
        return get_tfms
    raise ValueError(f"Unknown tfms: {name!r}")

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
        weights=str(dino_weights),
    )

    base_kwargs = dict(
        dataset=dataset,
        wide_df=wide_df,
        backbone=backbone,
        device=device,
        n_splits=int(cfg.get("n_splits", 5)),
        seed=int(cfg.get("seed")),
        group_col=str(cfg.get("group_col", "Sampling_Date")),
        stratify_col=str(cfg.get("stratify_col", "State")),
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        wd=float(cfg["wd"]),
        lr_start=float(cfg["lr_start"]),
        lr_final=float(cfg["lr_final"]),
        early_stopping=int(cfg["early_stopping"]),
        head_hidden=int(cfg["head_hidden"]),
        head_depth=int(cfg["head_depth"]),
        head_drop=float(cfg["head_drop"]),
        num_neck=int(cfg["num_neck"]),
        swa_epochs=int(cfg["swa_epochs"]),
        swa_lr=cfg.get("swa_lr", None),
        swa_anneal_epochs=int(cfg["swa_anneal_epochs"]),
        swa_load_best=bool(cfg.get("swa_load_best", True)),
        swa_eval_freq=int(cfg.get("swa_eval_freq", 2)),
        clip_val=cfg.get("clip_val", None),
        comet_exp_name=cfg.get("comet_exp_name", "csiro"),
        verbose=bool(cfg.get("verbose", False)),
        tfms_fn = tfms,
        plot_imgs = plot_imgs
    )
    
    if overrides:
        base_kwargs.update(overrides)

    sweep_id = str(uuid.uuid4())[:4]
    outputs: list[dict[str, Any]] = []
    for sweep in sweeps or SWEEPS:
        kwargs = dict(base_kwargs)
        kwargs.update({k: v for k, v in sweep.items() if k != "tfms"})
        kwargs["sweep_config"] = str(sweep)
        if kwargs["comet_exp_name"] is not None:
            kwargs["comet_exp_name"] = f"{kwargs['comet_exp_name']}-{sweep_id}"

        fold_scores, mean, std = run_groupkfold_cv(**kwargs)
        outputs.append(dict(sweep_config=kwargs["sweep_config"], fold_scores=fold_scores, mean=mean, std=std))

    return outputs

