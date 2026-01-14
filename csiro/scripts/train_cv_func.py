from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

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

def train_cv(
    *,
    csv: str | None = None,
    root: str = DEFAULT_DATA_ROOT,
    dino_repo: str = DEFAULT_DINO_ROOT,
    dino_weights: str | None = None,
    model_size: str | None = None,  # "b" == ViT-Base
    plus: str = "",
    overrides: dict[str, Any] | None = None,
    sweeps: dict = None
) -> Any:
    
    cfg: dict[str, Any] = dict(DEFAULTS)
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

    base_kwargs = dict(cfg)
    base_kwargs.update(
        dict(
            wide_df=wide_df,
            device=device,
        )
    )

    sweeps = sweeps or [base_kwargs]
    outputs: list[dict[str, Any]] = []
    for sweep in sweeps:
        kwargs = dict(base_kwargs)
        kwargs.update({k: v for k, v in sweep.items()})
        if (
            "backbone_size" in kwargs
            and "neck_num_heads" not in kwargs
            and int(kwargs.get("num_neck", 0)) > 0
        ):
            kwargs["neck_num_heads"] = neck_num_heads_for(kwargs["backbone_size"])
        name_src = dict(sweep)
        name_src.pop("cv_resume", None)
        kwargs["config_name"] = "".join(c for c in str(name_src) if c.isalnum() or c in "_-")[:80]

        sweep_model_size = str(
            kwargs.get("backbone_size", model_size or cfg.get("backbone_size", "b"))
        )
        sweep_dino_weights = dino_weights
        if sweep_dino_weights is None:
            sweep_dino_weights = dino_weights_path_from_size(
                str(kwargs.get("backbone_size", "b"))
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

        kwargs["backbone"] = backbone_cache[cache_key]
        tiled_inp = bool(kwargs.get("tiled_inp", cfg.get("tiled_inp", False)))
        model_name = str(kwargs.get("model_name", cfg.get("model_name", ""))).strip().lower()
        tile_geom_mode = str(kwargs.get("tile_geom_mode", cfg.get("tile_geom_mode", "shared"))).strip().lower()
        if tile_geom_mode not in ("shared", "independent"):
            raise ValueError(f"tile_geom_mode must be 'shared' or 'independent' (got {tile_geom_mode})")
        use_shared_geom = tiled_inp and tile_geom_mode == "shared"
        img_preprocess = bool(kwargs.get("img_preprocess", cfg.get("img_preprocess", False)))
        cache_key = "shared" if use_shared_geom else ("tiled" if tiled_inp else "base")
        if cache_key not in dataset_cache:
            if use_shared_geom:
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
        kwargs["dataset"] = dataset_cache[cache_key]
        result = run_groupkfold_cv(return_details=True, **kwargs)
        outputs.append(
            dict(
                config_name=kwargs["config_name"],
                result=result,
            )
        )

    return outputs
