from __future__ import annotations

import argparse
import sys
import uuid

import torch

from csiro_biomass.amp import set_dtype
from csiro_biomass.data import BiomassBaseCached, load_train_wide
from csiro_biomass.train import run_groupkfold_cv
from csiro_biomass.transforms import get_tfms, get_tfms_0

from experiment_config import DEFAULTS, SWEEPS


def _parse_dtype(s: str) -> torch.dtype:
    s = s.strip().lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to train.csv")
    ap.add_argument("--root", default=None, help="Root dir joined to image_path")
    ap.add_argument("--dino-repo", required=True, help="Local path to dinov3 repo (for torch.hub.load)")
    ap.add_argument("--dino-weights", required=True, help="Path to dinov3 weights .pth")

    ap.add_argument("--dtype", default=DEFAULTS["dtype"], help="fp16|bf16|fp32")
    ap.add_argument("--img-size", type=int, default=DEFAULTS["img_size"])
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--wd", type=float, default=DEFAULTS["wd"])
    ap.add_argument("--lr-start", type=float, default=DEFAULTS["lr_start"])
    ap.add_argument("--lr-final", type=float, default=DEFAULTS["lr_final"])
    ap.add_argument("--early-stopping", type=int, default=DEFAULTS["early_stopping"])

    ap.add_argument("--head-hidden", type=int, default=DEFAULTS["head_hidden"])
    ap.add_argument("--head-depth", type=int, default=DEFAULTS["head_depth"])
    ap.add_argument("--head-drop", type=float, default=DEFAULTS["head_drop"])
    ap.add_argument("--num-neck", type=int, default=DEFAULTS["num_neck"])

    ap.add_argument("--swa-epochs", type=int, default=DEFAULTS["swa_epochs"])
    ap.add_argument("--swa-lr", type=float, default=DEFAULTS["swa_lr"])
    ap.add_argument("--swa-anneal-epochs", type=int, default=DEFAULTS["swa_anneal_epochs"])
    ap.add_argument("--swa-load-best", action="store_true", default=DEFAULTS["swa_load_best"])
    ap.add_argument("--swa-eval-freq", type=int, default=DEFAULTS["swa_eval_freq"])

    ap.add_argument("--tfms", choices=("default", "tfms0"), default="default")
    ap.add_argument("--run-sweeps", action="store_true", help="Run predefined sweeps from scripts/experiment_config.py")
    ap.add_argument("--comet-project", default=None, help="Comet project name (requires comet-ml installed)")

    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--group-col", default="Sampling_Date")
    ap.add_argument("--stratify-col", default="State")
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    args = ap.parse_args()

    set_dtype(_parse_dtype(args.dtype))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sys.path.insert(0, args.dino_repo)
    wide_df = load_train_wide(args.csv, root=args.root)
    dataset = BiomassBaseCached(wide_df, img_size=args.img_size)

    backbone = torch.hub.load(args.dino_repo, "dinov3_vitb16", source="local", weights=args.dino_weights)

    tfms_fn = get_tfms if args.tfms == "default" else get_tfms_0
    base_kwargs = dict(
        dataset=dataset,
        wide_df=wide_df,
        backbone=backbone,
        n_splits=args.n_splits,
        seed=args.seed,
        group_col=args.group_col,
        stratify_col=args.stratify_col,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        wd=args.wd,
        lr_start=args.lr_start,
        lr_final=args.lr_final,
        early_stopping=args.early_stopping,
        head_hidden=args.head_hidden,
        head_depth=args.head_depth,
        head_drop=args.head_drop,
        num_neck=args.num_neck,
        swa_epochs=args.swa_epochs,
        swa_lr=args.swa_lr,
        swa_anneal_epochs=args.swa_anneal_epochs,
        swa_load_best=args.swa_load_best,
        swa_eval_freq=args.swa_eval_freq,
        comet_exp_name=args.comet_project,
        verbose=True,
    )

    if not args.run_sweeps:
        fold_scores, mean, std = run_groupkfold_cv(tfms_fn=tfms_fn, sweep_config="single", **base_kwargs)
        print("CV summary")
        for i, s in enumerate(fold_scores.tolist()):
            print(f"  fold {i}: {s:.4f}")
        print(f"  mean ± std: {mean:.4f} ± {std:.4f}")
        return

    sweep_id = str(uuid.uuid4())[:4]
    for sweep in SWEEPS:
        kwargs = dict(base_kwargs)
        kwargs.update(sweep)
        kwargs["sweep_config"] = str({k: v for k, v in sweep.items() if k != "tfms_fn"})
        if kwargs["comet_exp_name"] is not None:
            kwargs["comet_exp_name"] = f"{kwargs['comet_exp_name']}-{sweep_id}"
        fold_scores, mean, std = run_groupkfold_cv(**kwargs)
        print(f"\nSweep: {kwargs['sweep_config']}")
        for i, s in enumerate(fold_scores.tolist()):
            print(f"  fold {i}: {s:.4f}")
        print(f"  mean ± std: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()

