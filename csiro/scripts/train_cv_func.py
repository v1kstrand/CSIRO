from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from csiro.config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DINO_ROOT,
    DEFAULTS,
    dino_hub_name,
    dino_weights_path_from_size,
    neck_num_heads_for,
)
from csiro.data import BiomassBaseCached, BiomassFullCached, BiomassTiledCached, load_train_wide
from csiro.train import run_groupkfold_cv

BASE_ARGS_PATH = os.getenv("BASE_ARGS_PATH")

def train_cv(
    *,
    csv: str | None = None,
    root: str = DEFAULT_DATA_ROOT,
    dino_repo: str = DEFAULT_DINO_ROOT,
    dino_weights: str | None = None,
    model_size: str | None = None,  # "b" == ViT-Base
    plus: str = "",
    overrides: dict[str, Any] | str | Path | None = None,
) -> Any:

    cfg: dict[str, Any] = dict(DEFAULTS)
    overrides = _load_overrides(overrides)
    if overrides:
        cfg.update(overrides)
    device = str(cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    if csv is None:
        csv = os.path.join(root, "train.csv")

    sys.path.insert(0, str(dino_repo))
    wide_df = load_train_wide(str(csv), root=str(root))
    dataset_cache: dict[str, Any] = {}
    backbone_cache: dict[tuple[str, str, str], Any] = {}

    cfg.update(
        dict(
            wide_df=wide_df,
            device=device,
        )
    )
    run_name = str(cfg.get("run_name", "")).strip()
    print(f"run_name: {run_name}", exp_name)
    if not run_name:
        raise ValueError("run_name must be set (used for artifact naming).")
    exp_name = str(cfg.get("comet_exp_name", "")).strip() 
    if exp_name and not run_name.startswith(exp_name):
        run_name = f"{exp_name}_{run_name}"
        
    cfg["run_name"] = run_name
    save_output_dir = cfg.get("save_output_dir", None)
    if save_output_dir:
        state_dir = os.path.join(str(save_output_dir), "states")
        os.makedirs(state_dir, exist_ok=True)
        state_path = os.path.join(state_dir, f"{run_name}_cv_state.pt")
        if not os.path.exists(state_path):
            torch.save(
                dict(
                    completed=False,
                    last_completed_fold=-1,
                    last_completed_model=-1,
                    fold_scores=[],
                    fold_model_scores=[],
                    states=[],
                    run_name_resolved=str(run_name),
                ),
                state_path,
            )

    if (
        "backbone_size" in cfg
        and "neck_num_heads" not in cfg
        and int(cfg.get("num_neck", 0)) > 0
    ):
        cfg["neck_num_heads"] = neck_num_heads_for(cfg["backbone_size"])
    cfg["config_name"] = str(cfg.get("run_name", "")).strip()

    sweep_model_size = str(
        cfg.get("backbone_size", model_size or cfg.get("backbone_size", "b"))
    )
    sweep_dino_weights = dino_weights
    if sweep_dino_weights is None:
        sweep_dino_weights = dino_weights_path_from_size(
            str(cfg.get("backbone_size", "b"))
        )
    if sweep_dino_weights is None:
        raise ValueError("Set DINO_B_WEIGHTS_PATH or DINO_L_WEIGHTS_PATH for the chosen backbone_size.")

    cache_key = (str(sweep_model_size), str(sweep_dino_weights), str(plus))
    if cache_key not in backbone_cache:
        print("INFO: model_size:", sweep_model_size)
        backbone_cache[cache_key] = torch.hub.load(
            str(dino_repo),
            dino_hub_name(model_size=str(sweep_model_size), plus=str(plus)),
            source="local",
            weights=str(sweep_dino_weights),
        )

    cfg["backbone"] = backbone_cache[cache_key]
    tiled_inp = bool(cfg.get("tiled_inp", cfg.get("tiled_inp", False)))
    model_name = str(cfg.get("model_name", cfg.get("model_name", ""))).strip().lower()
    full_rect = model_name in ("rect_full", "full_rect", "rect")
    if full_rect and tiled_inp:
        raise ValueError("rect_full model requires tiled_inp=False.")
    if full_rect:
        tiled_inp = False
    tile_geom_mode = str(cfg.get("tile_geom_mode", cfg.get("tile_geom_mode", "shared"))).strip().lower()
    if tile_geom_mode not in ("shared", "independent"):
        raise ValueError(f"tile_geom_mode must be 'shared' or 'independent' (got {tile_geom_mode})")
    use_shared_geom = tiled_inp and tile_geom_mode == "shared"
    img_preprocess = bool(cfg.get("img_preprocess", cfg.get("img_preprocess", False)))
    cache_key = "full_rect" if full_rect else ("shared" if use_shared_geom else ("tiled" if tiled_inp else "base"))
    if cache_key not in dataset_cache:
        if full_rect:
            dataset_cache[cache_key] = BiomassFullCached(
                wide_df,
                img_preprocess=img_preprocess,
            )
        elif use_shared_geom:
            dataset_cache[cache_key] = BiomassFullCached(
                wide_df,
                img_preprocess=img_preprocess,
            )
        elif tiled_inp:
            dataset_cache[cache_key] = BiomassTiledCached(
                wide_df,
                img_size=int(cfg["img_size"]),
                img_preprocess=img_preprocess,
            )
        else:
            dataset_cache[cache_key] = BiomassBaseCached(
                wide_df,
                img_size=int(cfg["img_size"]),
                img_preprocess=img_preprocess,
            )
    cfg["dataset"] = dataset_cache[cache_key]
    return run_groupkfold_cv(return_details=True, **cfg)


def _load_overrides(overrides: dict[str, Any] | str | Path | None) -> dict[str, Any] | None:
    if overrides is None:
        return None
    if isinstance(overrides, dict):
        return overrides
    path = Path(overrides)
    if not path.is_file():
        raise FileNotFoundError(f"overrides yaml not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("overrides yaml must be a mapping")
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--overrides", type=str, default="")
    args = parser.parse_args()
    overrides_path = args.overrides.strip()
    run_args = _load_overrides(overrides_path) if overrides_path else None

    if BASE_ARGS_PATH:
        overrides = _load_overrides(BASE_ARGS_PATH) or {}
        if run_args:
            overrides.update(run_args)
    else:
        overrides = run_args

    train_cv(overrides=overrides)
