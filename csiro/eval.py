from __future__ import annotations

from typing import Any
from pathlib import Path
import glob

import torch
from torch.utils.data import DataLoader

from .amp import autocast_context
from .config import DEFAULTS, default_num_workers, parse_dtype
from .model import DINOv3Regressor, TiledDINOv3Regressor
from .transforms import post_tfms


def _is_state_dict(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    keys = state.keys()
    return any(k in keys for k in ("head_hidden", "head_depth", "head_drop", "num_neck", "parts"))


def _flatten_states(states: Any) -> list[dict[str, Any]]:
    if _is_state_dict(states):
        return [states]
    if isinstance(states, dict) and "states" in states:
        states = states["states"]
    if isinstance(states, dict):
        states = list(states.values())
    if states and isinstance(states[0], list):
        states = [s for fold in states for s in fold]
    return list(states)


def _normalize_runs(states: Any) -> list[list[dict[str, Any]]]:
    if _is_state_dict(states):
        return [[states]]
    if isinstance(states, dict) and "seed_results" in states:
        states = states["seed_results"]
    if isinstance(states, dict) and "states" in states:
        return [_flatten_states(states["states"])]
    if isinstance(states, dict):
        runs: list[list[dict[str, Any]]] = []
        for _, v in states.items():
            if isinstance(v, dict) and "states" in v:
                v = v["states"]
            runs.append(_flatten_states(v))
        return runs
    if isinstance(states, list):
        if not states:
            return []
        if isinstance(states[0], dict):
            return [states]
        if isinstance(states[0], list):
            return [list(run) for run in states]
    return [_flatten_states(states)]


def _require_tiled_runs(runs: list[list[dict[str, Any]]]) -> None:
    for run in runs:
        for s in run:
            if not bool(s.get("tiled_inp", False)):
                raise ValueError("predict_ensemble_tiled requires tiled checkpoints (tiled_inp=True).")


def _agg_stack(xs: list[torch.Tensor], agg: str) -> torch.Tensor:
    if len(xs) == 1:
        return xs[0]
    agg = str(agg).lower()
    stacked = torch.stack(xs, dim=0)
    if agg == "mean":
        return stacked.mean(dim=0)
    if agg == "median":
        return stacked.median(dim=0).values
    raise ValueError(f"Unknown aggregation: {agg}")


def _agg_tta(p: torch.Tensor, agg: str) -> torch.Tensor:
    agg = str(agg).lower()
    if agg == "mean":
        return p.mean(dim=1)
    if agg == "median":
        return p.median(dim=1).values
    raise ValueError(f"Unknown aggregation: {agg}")


def _split_tta_batch(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    if x.ndim != 5:
        raise ValueError(f"Expected batched TTA [B,T,C,H,W], got {tuple(x.shape)}")
    b, t, c, h, w = x.shape
    return x.view(b * t, c, h, w), int(t)


def _ensure_tensor_batch(x, tfms) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        xs = [xi if torch.is_tensor(xi) else tfms(xi) for xi in x]
        return torch.stack(xs, dim=0)
    return tfms(x).unsqueeze(0)


def _get_tta_n(data) -> int:
    obj = data
    for _ in range(4):
        if hasattr(obj, "tta_n"):
            try:
                return int(getattr(obj, "tta_n"))
            except Exception:
                return 1
        if hasattr(obj, "dataset"):
            obj = getattr(obj, "dataset")
            continue
        if hasattr(obj, "base"):
            obj = getattr(obj, "base")
            continue
        break
    return 1


def _build_model_from_state(
    backbone,
    state: dict[str, Any],
    device: str | torch.device,
    backbone_dtype: torch.dtype | None = None,
):
    def _load_backbone_ln_state(backbone_obj, ln_state: dict[str, torch.Tensor] | None) -> None:
        if isinstance(ln_state, dict) and ln_state:
            backbone_obj.load_state_dict(ln_state, strict=False)

    use_tiled = bool(state.get("tiled_inp", False))
    model_cls = TiledDINOv3Regressor if use_tiled else DINOv3Regressor
    model = model_cls(
        backbone,
        hidden=int(state["head_hidden"]),
        drop=float(state["head_drop"]),
        depth=int(state["head_depth"]),
        num_neck=int(state["num_neck"]),
        backbone_dtype=backbone_dtype,
    ).to(device)
    parts = state.get("parts")
    if isinstance(parts, dict):
        for name in ("neck", "head", "norm"):
            part = getattr(model, name, None)
            if part is not None and name in parts:
                part.load_state_dict(parts[name], strict=True)
    else:
        model.load_state_dict(state, strict=False)
    _load_backbone_ln_state(model.backbone, state.get("backbone_ln"))
    return model


def load_states_from_pt(pt_path: str) -> Any:
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    return ckpt["states"] if isinstance(ckpt, dict) and "states" in ckpt else ckpt


def load_ensemble_states(pt_paths: list[str] | str) -> list[list[dict[str, Any]]]:
    if isinstance(pt_paths, (str, Path)):
        pt_paths = [str(pt_paths)]
    paths: list[str] = []
    for p in pt_paths:
        if any(ch in p for ch in "*?[]"):
            paths.extend(sorted(glob.glob(p)))
        else:
            paths.append(p)
    if not paths:
        raise ValueError("No checkpoint paths provided.")

    runs_all: list[list[dict[str, Any]]] = []
    for p in paths:
        states = load_states_from_pt(str(p))
        runs_all.extend(_normalize_runs(states))
    if not runs_all:
        raise ValueError("No states found in checkpoints.")
    return runs_all


def _require(state: dict[str, Any], key: str) -> Any:
    if key not in state:
        raise ValueError(f"Missing '{key}' in checkpoint state.")
    return state[key]


def load_dinov3_regressor_from_pt(
    pt_path: str,
    backbone,
    *,
    device: str | torch.device = "cuda",
    backbone_dtype: str | torch.dtype | None = None,
    state_idx: int = 0,
    seed: str | int | None = None,
) -> DINOv3Regressor:
    states_raw = load_states_from_pt(pt_path)
    seed_states = _extract_seed_states(states_raw)
    if not seed_states:
        raise ValueError("No states found in checkpoint.")
    if seed is None:
        seed = sorted(seed_states.keys())[0]
    seed_key = str(seed)
    if seed_key not in seed_states:
        raise KeyError(f"Seed '{seed}' not found in checkpoint.")
    states = seed_states[seed_key]
    if int(state_idx) >= len(states):
        raise IndexError(f"state_idx {state_idx} out of range for {len(states)} states.")

    state = states[int(state_idx)]
    if backbone_dtype is None:
        backbone_dtype = DEFAULTS["backbone_dtype"]
    if isinstance(backbone_dtype, str):
        backbone_dtype = parse_dtype(backbone_dtype)

    model = DINOv3Regressor(
        backbone,
        hidden=int(_require(state, "head_hidden")),
        drop=float(_require(state, "head_drop")),
        depth=int(_require(state, "head_depth")),
        num_neck=int(_require(state, "num_neck")),
        backbone_dtype=backbone_dtype,
    ).to(device)

    parts = state.get("parts")
    if isinstance(parts, dict):
        for name in ("neck", "head", "norm"):
            part = getattr(model, name, None)
            if part is not None and name in parts:
                part.load_state_dict(parts[name], strict=True)
    else:
        model.load_state_dict(state, strict=False)
    ln_state = state.get("backbone_ln")
    if isinstance(ln_state, dict) and ln_state:
        model.backbone.load_state_dict(ln_state, strict=False)

    if hasattr(model, "set_train"):
        model.set_train(False)
    model.eval()
    return model




@torch.no_grad()
def predict_ensemble(
    data,
    states: Any,
    backbone,
    *,
    batch_size: int = 128,
    num_workers: int | None = None,
    device: str | torch.device = "cuda",
    backbone_dtype: str | torch.dtype | None = None,
    trainable_dtype: str | torch.dtype | None = None,
    tta_agg: str = "mean",
    inner_agg: str = "mean",
    outer_agg: str = "mean",
) -> torch.Tensor:
    runs = _normalize_runs(states)

    if isinstance(data, DataLoader):
        dl = data
    else:
        num_workers = default_num_workers() if num_workers is None else int(num_workers)
        tta_n = _get_tta_n(data)
        if tta_n > 1:
            batch_size = max(1, int(batch_size) // int(tta_n))
        dl = DataLoader(
            data,
            shuffle=False,
            batch_size=int(batch_size),
            pin_memory=str(device).startswith("cuda"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

    if backbone_dtype is None:
        backbone_dtype = DEFAULTS["backbone_dtype"]
    if isinstance(backbone_dtype, str):
        backbone_dtype = parse_dtype(backbone_dtype)
    if trainable_dtype is None:
        trainable_dtype = DEFAULTS["trainable_dtype"]
    if isinstance(trainable_dtype, str):
        trainable_dtype = parse_dtype(trainable_dtype)

    tfms = post_tfms()
    outer_agg = str(outer_agg).lower()

    def _predict_with_models(models: list[DINOv3Regressor]) -> torch.Tensor:
        for model in models:
            if hasattr(model, "set_train"):
                model.set_train(False)
            model.eval()

        preds: list[torch.Tensor] = []
        with torch.inference_mode(), autocast_context(device, dtype=trainable_dtype):
            for batch in dl:
                if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                    x = batch[0]
                else:
                    x = batch

                x = _ensure_tensor_batch(x, tfms).to(device, non_blocking=True)

                preds_models: list[torch.Tensor] = []
                for model in models:
                    if x.ndim == 5:
                        x_tta, t = _split_tta_batch(x)
                        p_log = model(x_tta).float()
                        p = torch.expm1(p_log).clamp_min(0.0)
                        p = p.view(x.size(0), int(t), -1)
                        preds_models.append(_agg_tta(p, tta_agg))
                    elif x.ndim == 4:
                        p_log = model(x).float()
                        p = torch.expm1(p_log).clamp_min(0.0)
                        preds_models.append(p)
                    else:
                        raise ValueError(f"Expected batch [B,C,H,W] or [B,T,C,H,W], got {tuple(x.shape)}")

                p_ens = _agg_stack(preds_models, inner_agg)
                preds.append(p_ens.detach().cpu())

        return torch.cat(preds, dim=0)

    if outer_agg == "flatten":
        flat_states = [s for run in runs for s in run]
        models = [_build_model_from_state(backbone, s, device, backbone_dtype) for s in flat_states]
        return _predict_with_models(models)
    if outer_agg in ("mean", "median"):
        preds_runs: list[torch.Tensor] = []
        for run in runs:
            models = [_build_model_from_state(backbone, s, device, backbone_dtype) for s in run]
            preds_runs.append(_predict_with_models(models))
        return _agg_stack(preds_runs, outer_agg)
    raise ValueError(f"Unknown outer_agg: {outer_agg}")


@torch.no_grad()
def predict_ensemble_tiled(
    data,
    states: Any,
    backbone,
    *,
    batch_size: int = 128,
    num_workers: int | None = None,
    device: str | torch.device = "cuda",
    backbone_dtype: str | torch.dtype | None = None,
    trainable_dtype: str | torch.dtype | None = None,
    tta_agg: str = "mean",
    inner_agg: str = "mean",
    outer_agg: str = "mean",
) -> torch.Tensor:
    runs = _normalize_runs(states)
    _require_tiled_runs(runs)

    if isinstance(data, DataLoader):
        dl = data
    else:
        num_workers = default_num_workers() if num_workers is None else int(num_workers)
        tta_n = _get_tta_n(data)
        tile_n = 2
        if tta_n > 1 or tile_n > 1:
            batch_size = max(1, int(batch_size) // int(tile_n * max(1, tta_n)))
        dl = DataLoader(
            data,
            shuffle=False,
            batch_size=int(batch_size),
            pin_memory=str(device).startswith("cuda"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

    if backbone_dtype is None:
        backbone_dtype = DEFAULTS["backbone_dtype"]
    if isinstance(backbone_dtype, str):
        backbone_dtype = parse_dtype(backbone_dtype)
    if trainable_dtype is None:
        trainable_dtype = DEFAULTS["trainable_dtype"]
    if isinstance(trainable_dtype, str):
        trainable_dtype = parse_dtype(trainable_dtype)

    outer_agg = str(outer_agg).lower()

    def _predict_with_models(models: list[DINOv3Regressor]) -> torch.Tensor:
        for model in models:
            if hasattr(model, "set_train"):
                model.set_train(False)
            model.eval()

        preds: list[torch.Tensor] = []
        with torch.inference_mode(), autocast_context(device, dtype=trainable_dtype):
            for batch in dl:
                if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                    x = batch[0]
                else:
                    x = batch

                if not torch.is_tensor(x):
                    raise ValueError("predict_ensemble_tiled expects tensor batches.")
                x = x.to(device, non_blocking=True)

                preds_models: list[torch.Tensor] = []
                for model in models:
                    if x.ndim == 6:
                        b, t, tiles, c, h, w = x.shape
                        if tiles != 2:
                            raise ValueError(f"Expected tiles=2, got {tiles}.")
                        x_tta = x.view(b * t, tiles, c, h, w)
                        p_log = model(x_tta).float()
                        p = torch.expm1(p_log).clamp_min(0.0)
                        p = p.view(b, t, -1)
                        preds_models.append(_agg_tta(p, tta_agg))
                    elif x.ndim == 5:
                        p_log = model(x).float()
                        p = torch.expm1(p_log).clamp_min(0.0)
                        preds_models.append(p)
                    else:
                        raise ValueError(f"Expected [B,2,C,H,W] or [B,T,2,C,H,W], got {tuple(x.shape)}")

                p_ens = _agg_stack(preds_models, inner_agg)
                preds.append(p_ens.detach().cpu())

        return torch.cat(preds, dim=0)

    if outer_agg == "flatten":
        flat_states = [s for run in runs for s in run]
        models = [_build_model_from_state(backbone, s, device, backbone_dtype) for s in flat_states]
        return _predict_with_models(models)
    if outer_agg in ("mean", "median"):
        preds_runs: list[torch.Tensor] = []
        for run in runs:
            models = [_build_model_from_state(backbone, s, device, backbone_dtype) for s in run]
            preds_runs.append(_predict_with_models(models))
        return _agg_stack(preds_runs, outer_agg)
    raise ValueError(f"Unknown outer_agg: {outer_agg}")


def predict_ensemble_from_pt(
    data,
    pt_path: str,
    backbone,
    *,
    batch_size: int = 128,
    num_workers: int | None = None,
    device: str | torch.device = "cuda",
    backbone_dtype: str | torch.dtype | None = None,
    trainable_dtype: str | torch.dtype | None = None,
    tta_agg: str = "mean",
    inner_agg: str = "mean",
    outer_agg: str = "flatten",
) -> torch.Tensor:
    states = load_states_from_pt(pt_path)
    return predict_ensemble(
        data,
        states=states,
        backbone=backbone,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        backbone_dtype=backbone_dtype,
        trainable_dtype=trainable_dtype,
        tta_agg=tta_agg,
        inner_agg=inner_agg,
        outer_agg=outer_agg,
    )
