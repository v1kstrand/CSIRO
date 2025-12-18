from __future__ import annotations

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
)
from csiro.data import BiomassBaseCached, load_train_wide
from csiro.train import run_groupkfold_cv
from csiro.transforms import get_tfms, get_tfms_0


def _parse_dtype(s: str) -> torch.dtype:
    s = str(s).strip().lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


def _tfms_from_name(name: str):
    if name == "default":
        return get_tfms
    if name == "tfms0":
        return get_tfms_0
    raise ValueError(f"Unknown tfms: {name!r}")


def train_cv(
    *,
    csv: str | None = None,
    root: str = DEFAULT_DATA_ROOT,
    dino_repo: str = DEFAULT_DINO_REPO_DIR,
    dino_weights: str | None = None,
    model_size: str = DEFAULT_MODEL_SIZE,  # "b" == ViT-Base
    plus: str = DEFAULT_PLUS,
    dtype: str = DEFAULTS["dtype"],
    img_size: int = DEFAULTS["img_size"],
    epochs: int = DEFAULTS["epochs"],
    batch_size: int = DEFAULTS["batch_size"],
    wd: float = DEFAULTS["wd"],
    lr_start: float = DEFAULTS["lr_start"],
    lr_final: float = DEFAULTS["lr_final"],
    early_stopping: int = DEFAULTS["early_stopping"],
    head_hidden: int = DEFAULTS["head_hidden"],
    head_depth: int = DEFAULTS["head_depth"],
    head_drop: float = DEFAULTS["head_drop"],
    num_neck: int = DEFAULTS["num_neck"],
    swa_epochs: int = DEFAULTS["swa_epochs"],
    swa_lr: float | None = DEFAULTS["swa_lr"],
    swa_anneal_epochs: int = DEFAULTS["swa_anneal_epochs"],
    swa_load_best: bool = DEFAULTS["swa_load_best"],
    swa_eval_freq: int = DEFAULTS["swa_eval_freq"],
    tfms: str = "default",  # "default" | "tfms0"
    run_sweeps: bool = False,
    comet_project: str | None = None,
    n_splits: int = 5,
    group_col: str = "Sampling_Date",
    stratify_col: str = "State",
    seed: int = DEFAULTS["seed"],
    device: str | None = None,
    verbose: bool = True,
) -> Any:
    set_dtype(_parse_dtype(dtype))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if csv is None:
        import os
        csv = os.path.join(root, "train.csv")
    if dino_weights is None:
        dino_weights = dino_weights_path(repo_dir=dino_repo, model_size=model_size, plus=plus)

    sys.path.insert(0, str(dino_repo))
    wide_df = load_train_wide(str(csv), root=root)
    dataset = BiomassBaseCached(wide_df, img_size=int(img_size))

    backbone = torch.hub.load(
        str(dino_repo),
        dino_hub_name(model_size=str(model_size), plus=str(plus)),
        source="local",
        weights=str(dino_weights),
    )

    tfms_fn = _tfms_from_name(tfms)
    base_kwargs = dict(
        dataset=dataset,
        wide_df=wide_df,
        backbone=backbone,
        n_splits=int(n_splits),
        seed=int(seed),
        group_col=group_col,
        stratify_col=stratify_col,
        device=device,
        epochs=int(epochs),
        batch_size=int(batch_size),
        wd=float(wd),
        lr_start=float(lr_start),
        lr_final=float(lr_final),
        early_stopping=int(early_stopping),
        head_hidden=int(head_hidden),
        head_depth=int(head_depth),
        head_drop=float(head_drop),
        num_neck=int(num_neck),
        swa_epochs=int(swa_epochs),
        swa_lr=swa_lr,
        swa_anneal_epochs=int(swa_anneal_epochs),
        swa_load_best=bool(swa_load_best),
        swa_eval_freq=int(swa_eval_freq),
        comet_exp_name=comet_project,
        verbose=bool(verbose),
    )

    if not run_sweeps:
        return run_groupkfold_cv(tfms_fn=tfms_fn, sweep_config="single", **base_kwargs)

    sweep_id = str(uuid.uuid4())[:4]
    outputs: list[dict[str, Any]] = []
    for sweep in SWEEPS:
        kwargs = dict(base_kwargs)
        kwargs.update({k: v for k, v in sweep.items() if k != "tfms"})
        kwargs["tfms_fn"] = _tfms_from_name(str(sweep.get("tfms", "default")))
        kwargs["sweep_config"] = str(sweep)
        if kwargs["comet_exp_name"] is not None:
            kwargs["comet_exp_name"] = f"{kwargs['comet_exp_name']}-{sweep_id}"

        fold_scores, mean, std = run_groupkfold_cv(**kwargs)
        outputs.append(dict(sweep_config=kwargs["sweep_config"], fold_scores=fold_scores, mean=mean, std=std))
    return outputs
