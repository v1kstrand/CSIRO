from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

import torch

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_PKG_SRC))

from csiro_biomass.amp import set_dtype
from csiro_biomass.data import BiomassBaseCached, load_train_wide
from csiro_biomass.train import run_groupkfold_cv
from csiro_biomass.transforms import get_tfms, get_tfms_0

from experiment_config import DEFAULTS, SWEEPS


def _parse_dtype(s: str) -> torch.dtype:
    s = str(s).strip().lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


def train_cv(
    *,
    csv: str,
    dino_repo: str,
    dino_weights: str,
    root: str | None = None,
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
    """
    Programmatic (non-argparse) entrypoint equivalent to csiro/scripts/train_cv.py.

    Returns:
      - if run_sweeps=False: (fold_scores, mean, std)
      - if run_sweeps=True: list[dict] with per-sweep outputs
    """
    set_dtype(_parse_dtype(dtype))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sys.path.insert(0, dino_repo)
    wide_df = load_train_wide(csv, root=root)
    dataset = BiomassBaseCached(wide_df, img_size=int(img_size))

    backbone = torch.hub.load(dino_repo, "dinov3_vitb16", source="local", weights=dino_weights)

    tfms_fn = get_tfms if tfms == "default" else get_tfms_0
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
        kwargs.update(sweep)
        kwargs["sweep_config"] = str({k: v for k, v in sweep.items() if k != "tfms_fn"})
        if kwargs["comet_exp_name"] is not None:
            kwargs["comet_exp_name"] = f"{kwargs['comet_exp_name']}-{sweep_id}"
        fold_scores, mean, std = run_groupkfold_cv(**kwargs)
        outputs.append(
            dict(
                sweep_config=kwargs["sweep_config"],
                fold_scores=fold_scores,
                mean=mean,
                std=std,
            )
        )
    return outputs
