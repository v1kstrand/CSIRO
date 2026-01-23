from __future__ import annotations

from typing import Any
from pathlib import Path
import glob

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .amp import autocast_context
from .config import DEFAULTS, TARGETS, default_num_workers, parse_dtype, neck_num_heads_for
from .ensemble_utils import (
    _agg_stack,
    _agg_tta,
    _ensure_tensor_batch,
    _get_tta_n,
    _split_tta_batch,
)
from .metrics import eval_global_wr2
from .model import (
    TiledDINOv3Regressor,
    TiledDINOv3Regressor3,
    TiledDINOv3RegressorStitched3,
    FullDINOv3RegressorRect3,
)
from .transforms import post_tfms

def _normalize_pred_space(pred_space: str) -> str:
    s = str(pred_space).strip().lower()
    if s in ("log", "log1p"):
        return "log"
    if s in ("gram", "grams", "linear"):
        return "gram"
    raise ValueError(f"Unknown pred_space: {pred_space}")

def _pred_to_grams(pred: torch.Tensor, pred_space: str, *, clamp: bool = True) -> torch.Tensor:
    if pred_space == "gram":
        out = pred.float()
    else:
        out = torch.expm1(pred.float())
    return out.clamp_min(0.0) if clamp else out

try:
    _IDX_GREEN = TARGETS.index("Dry_Green_g")
    _IDX_CLOVER = TARGETS.index("Dry_Clover_g")
    _IDX_DEAD = TARGETS.index("Dry_Dead_g")
    _IDX_GDM = TARGETS.index("GDM_g")
    _IDX_TOTAL = TARGETS.index("Dry_Total_g")
except ValueError as exc:
    raise ValueError("TARGETS must include Dry_Green_g, Dry_Clover_g, Dry_Dead_g, GDM_g, Dry_Total_g.") from exc


def _postprocess_mass_balance(pred: torch.Tensor) -> torch.Tensor:
    if pred.size(-1) < 5:
        return pred
    out = pred.clone()
    green = out[..., _IDX_GREEN].clamp_min(0.0)
    clover = out[..., _IDX_CLOVER].clamp_min(0.0)
    dead = out[..., _IDX_DEAD].clamp_min(0.0)
    gdm = green + clover
    total = gdm + dead
    out[..., _IDX_GREEN] = green
    out[..., _IDX_CLOVER] = clover
    out[..., _IDX_DEAD] = dead
    out[..., _IDX_GDM] = gdm
    out[..., _IDX_TOTAL] = total
    return out

def _resolve_model_class(model_name: str | None, tiled_inp: bool) -> type[torch.nn.Module]:
    name = str(model_name or "").strip().lower()
    if not tiled_inp:
        if name in ("rect_full", "full_rect", "rect"):
            return FullDINOv3RegressorRect3
        raise ValueError("Non-tiled models are no longer supported. Set tiled_inp=True.")
    if not name or name in ("tiled_base", "tiled"):
        return TiledDINOv3Regressor
    if name in ("tiled_sum3", "tiled_mass3", "tiled_3sum", "tiled_3"):
        return TiledDINOv3Regressor3
    if name in ("tiled_stitch", "tiled_stitch3", "tiled_stitched"):
        return TiledDINOv3RegressorStitched3
    raise ValueError(f"Unknown model_name: {model_name}")


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
    def _is_fold_list(run: Any) -> bool:
        if not isinstance(run, list) or not run:
            return False
        fold_ids = set()
        for s in run:
            if not isinstance(s, dict) or "fold_idx" not in s:
                return False
            try:
                fold_ids.add(int(s["fold_idx"]))
            except Exception:
                return False
        return len(fold_ids) == 1

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
            if all(_is_fold_list(run) for run in states):
                flat = [s for run in states for s in run]
                return [flat]
            return [list(run) for run in states]
    return [_flatten_states(states)]


def _require_tiled_runs(runs: list[list[dict[str, Any]]]) -> None:
    for run in runs:
        for s in run:
            if not bool(s.get("tiled_inp", False)):
                raise ValueError("predict_ensemble_tiled requires tiled checkpoints (tiled_inp=True).")


def _avg_state_parts(states: list[dict[str, dict[str, torch.Tensor]]]) -> dict[str, dict[str, torch.Tensor]]:
    if not states:
        raise ValueError("Cannot average empty state list.")
    out: dict[str, dict[str, torch.Tensor]] = {}
    for name in states[0].keys():
        if not isinstance(states[0][name], dict):
            continue
        out[name] = {}
        keys = states[0][name].keys()
        for k in keys:
            vals = [s[name][k] for s in states]
            if not torch.is_tensor(vals[0]):
                out[name][k] = vals[0]
                continue
            if vals[0].is_floating_point():
                stack = torch.stack([v.float() for v in vals], dim=0)
                out[name][k] = stack.mean(dim=0).to(dtype=vals[0].dtype)
            else:
                out[name][k] = vals[0]
    return out


def _resolve_kweights_state(state: dict[str, Any], k_weights: int) -> dict[str, Any]:
    if int(k_weights) <= 0:
        return state
    parts = state.get("parts")
    if not isinstance(parts, dict):
        return state
    last_k_states = parts.get("last_k_states")
    if not isinstance(last_k_states, list) or not last_k_states:
        return state
    k = min(int(k_weights), len(last_k_states))
    if k <= 0:
        return state
    latest_first = bool(parts.get("last_k_latest_first", False))
    use_states = last_k_states[:k] if latest_first else last_k_states[-k:]
    avg_parts = _avg_state_parts(use_states)
    new_state = dict(state)
    new_parts = dict(parts)
    new_parts.update(avg_parts)
    new_state["parts"] = new_parts
    return new_state


 


def _build_model_from_state(
    backbone,
    state: dict[str, Any],
    device: str | torch.device,
    backbone_dtype: torch.dtype | None = None,
):
    use_tiled = bool(state.get("tiled_inp", False))
    model_name = str(state.get("model_name", "")).strip().lower()
    model_cls = _resolve_model_class(model_name or None, use_tiled)
    backbone_size = str(state.get("backbone_size", DEFAULTS.get("backbone_size", "b")))
    neck_num_heads = int(state.get("neck_num_heads", neck_num_heads_for(backbone_size)))
    pred_space = _normalize_pred_space(state.get("pred_space", DEFAULTS.get("pred_space", "log")))
    head_style = str(state.get("head_style", DEFAULTS.get("head_style", "single"))).strip().lower()
    if pred_space == "gram" and model_cls is TiledDINOv3Regressor:
        raise ValueError("pred_space='gram' is only supported for the 3-output model variants.")
    model_kwargs = dict(
        backbone=backbone,
        hidden=int(state["head_hidden"]),
        drop=float(state["head_drop"]),
        depth=int(state["head_depth"]),
        num_neck=int(state["num_neck"]),
        neck_num_heads=int(neck_num_heads),
        backbone_dtype=backbone_dtype,
        pred_space=pred_space,
    )
    if model_cls in (TiledDINOv3Regressor3, TiledDINOv3RegressorStitched3, FullDINOv3RegressorRect3):
        model_kwargs["head_style"] = head_style
        if model_cls in (TiledDINOv3RegressorStitched3, FullDINOv3RegressorRect3):
            out_format = str(state.get("out_format", DEFAULTS.get("out_format", "cat_cls"))).strip().lower()
            model_kwargs["out_format"] = out_format
            model_kwargs["neck_rope"] = bool(state.get("neck_rope", DEFAULTS.get("neck_rope", True)))
            if model_cls is TiledDINOv3RegressorStitched3:
                model_kwargs["neck_pool"] = bool(state.get("neck_pool", DEFAULTS.get("neck_pool", False)))
            model_kwargs["rope_rescale"] = state.get("rope_rescale", DEFAULTS.get("rope_rescale", None))
            model_kwargs["neck_drop"] = float(state.get("neck_drop", DEFAULTS.get("neck_drop", 0.0)))
            model_kwargs["drop_path"] = state.get("drop_path", DEFAULTS.get("drop_path", None))
            model_kwargs["neck_ffn"] = bool(state.get("neck_ffn", DEFAULTS.get("neck_ffn", True)))
            model_kwargs["neck_layer_scale"] = state.get(
                "neck_layer_scale",
                DEFAULTS.get("neck_layer_scale", None),
            )
    model = model_cls(**model_kwargs).to(device)
    model.model_name = model_name or ("tiled_base" if use_tiled else "base")
    model.pred_space = pred_space
    parts = state.get("parts")
    if isinstance(parts, dict):
        for name in ("neck", "head", "norm"):
            part = getattr(model, name, None)
            if part is not None and name in parts:
                part.load_state_dict(parts[name], strict=True)
    else:
        model.load_state_dict(state, strict=False)
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
    k_weights: int = 0,
) -> torch.Tensor:
    runs = _normalize_runs(states)
    if int(k_weights) > 0:
        runs = [[_resolve_kweights_state(s, int(k_weights)) for s in run] for run in runs]

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
    backbone.eval()

    tfms = post_tfms()
    outer_agg = str(outer_agg).lower()

    def _predict_with_models(models: list[torch.nn.Module]) -> torch.Tensor:
        for model in models:
            model.eval()

        preds: list[torch.Tensor] = []
        ctx = autocast_context(device, dtype=trainable_dtype)
        for batch in dl:
            if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                x = batch[0]
            else:
                x = batch

            x = _ensure_tensor_batch(x, tfms).to(device, non_blocking=True)

            preds_models: list[torch.Tensor] = []
            for model in models:
                with torch.no_grad(), ctx:
                    if x.ndim == 5:
                        x_tta, t = _split_tta_batch(x)
                        p_raw = model(x_tta).float()
                        pred_space = getattr(model, "pred_space", "log")
                        p = _pred_to_grams(p_raw, pred_space, clamp=True)
                        p = p.view(x.size(0), int(t), -1)
                        preds_models.append(_agg_tta(p, tta_agg))
                    elif x.ndim == 4:
                        p_raw = model(x).float()
                        pred_space = getattr(model, "pred_space", "log")
                        p = _pred_to_grams(p_raw, pred_space, clamp=True)
                        preds_models.append(p)
                    else:
                        raise ValueError(f"Expected batch [B,C,H,W] or [B,T,C,H,W], got {tuple(x.shape)}")

            p_ens = _agg_stack(preds_models, inner_agg)
            p_ens = _postprocess_mass_balance(p_ens)
            preds.append(p_ens.detach().cpu())

        return torch.cat(preds, dim=0)

    if outer_agg == "flatten":
        flat_states = [s for run in runs for s in run]
        models = [_build_model_from_state(backbone, s, device, backbone_dtype) for s in flat_states]
        preds = _predict_with_models(models)
        return _postprocess_mass_balance(preds)
    if outer_agg in ("mean", "median"):
        preds_runs: list[torch.Tensor] = []
        for run in runs:
            models = [_build_model_from_state(backbone, s, device, backbone_dtype) for s in run]
            preds_runs.append(_predict_with_models(models))
        preds = _agg_stack(preds_runs, outer_agg)
        return _postprocess_mass_balance(preds)
    raise ValueError(f"Unknown outer_agg: {outer_agg}")


def _make_cv_iter(wide_df, cv_cfg: int | None) -> list[tuple[Any, Any]]:
    if cv_cfg is None:
        cv_cfg = int(DEFAULTS.get("cv_cfg", 1))
    cv_cfg = int(cv_cfg)
    groups = wide_df["Sampling_Date"].values
    if cv_cfg == 1:
        gkf = GroupKFold(n_splits=5, shuffle=True, random_state=126015)
        return list(gkf.split(wide_df, groups=groups))
    if cv_cfg == 2:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=82947501)
        strat = wide_df["State"].astype(str).values
        return list(sgkf.split(wide_df, y=strat, groups=groups))
    raise ValueError(f"Unsupported cv_cfg: {cv_cfg}")


def eval_fold_ensemble_from_pt(
    *,
    pt_path: str,
    fold_idx: int,
    dataset,
    backbone,
    wide_df=None,
    va_idx=None,
    cv_cfg: int | None = None,
    batch_size: int = 128,
    num_workers: int | None = None,
    device: str | torch.device = "cuda",
    backbone_dtype: str | torch.dtype | None = None,
    trainable_dtype: str | torch.dtype | None = None,
    tta_agg: str = "mean",
    inner_agg: str = "mean",
    k_weights: int = 0,
    tiled_inp: bool | None = None,
) -> list[dict[str, Any]]:
    states = load_states_from_pt(pt_path)
    runs = _normalize_runs(states)
    if not runs:
        raise ValueError("No runs found in checkpoint.")

    if va_idx is None:
        if wide_df is None:
            raise ValueError("Provide wide_df or va_idx.")
        cv_iter = _make_cv_iter(wide_df, cv_cfg)
        for i, (_, v_idx) in enumerate(cv_iter):
            if int(i) == int(fold_idx):
                va_idx = v_idx
                break
        if va_idx is None:
            raise ValueError(f"Fold {fold_idx} not found in CV iterator.")

    num_workers = default_num_workers() if num_workers is None else int(num_workers)
    tta_n = _get_tta_n(dataset)
    tile_n = 2 if tiled_inp is True else 1
    if tiled_inp is None and runs and runs[0]:
        tile_n = 2 if bool(runs[0][0].get("tiled_inp", False)) else 1
    bs_eff = max(1, int(batch_size) // int(tile_n * max(1, tta_n)))

    subset = Subset(dataset, va_idx)
    dl = DataLoader(
        subset,
        shuffle=False,
        batch_size=int(bs_eff),
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

    loss_weights = DEFAULTS.get("loss_weights", (1.0, 1.0, 1.0, 0.0, 0.0))
    w_vec = torch.as_tensor(loss_weights, dtype=torch.float32)

    def _eval_ensemble(models: list[torch.nn.Module]) -> float:
        for model in models:
            model.eval()
        w5 = w_vec.to(device).view(1, -1)
        ss_res = torch.zeros((), device=device)
        sum_w = torch.zeros((), device=device)
        sum_wy = torch.zeros((), device=device)
        sum_wy2 = torch.zeros((), device=device)
        ctx = autocast_context(device, dtype=trainable_dtype)
        for batch in dl:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x, y_log = batch[0], batch[1]
            else:
                raise ValueError("Validation loader must yield (x, y_log).")
            x = x.to(device, non_blocking=True)
            y_log = y_log.to(device, non_blocking=True)

            preds_models: list[torch.Tensor] = []
            for model in models:
                with torch.no_grad(), ctx:
                    if tile_n == 2:
                        if x.ndim != 5:
                            raise ValueError(f"Expected tiled batch [B,2,C,H,W], got {tuple(x.shape)}")
                        p_raw = model(x).float()
                        pred_space = getattr(model, "pred_space", "log")
                        p = _pred_to_grams(p_raw, pred_space, clamp=True)
                        preds_models.append(p)
                    elif x.ndim == 5:
                        x_tta, t = _split_tta_batch(x)
                        p_raw = model(x_tta).float()
                        pred_space = getattr(model, "pred_space", "log")
                        p = _pred_to_grams(p_raw, pred_space, clamp=True)
                        p = p.view(x.size(0), int(t), -1)
                        preds_models.append(_agg_tta(p, tta_agg))
                    elif x.ndim == 4:
                        p_raw = model(x).float()
                        pred_space = getattr(model, "pred_space", "log")
                        p = _pred_to_grams(p_raw, pred_space, clamp=True)
                        preds_models.append(p)
                    else:
                        raise ValueError(f"Expected batch [B,C,H,W] or [B,T,C,H,W], got {tuple(x.shape)}")

            p_ens = _agg_stack(preds_models, inner_agg)
            p_ens = _postprocess_mass_balance(p_ens)
            y = torch.expm1(y_log.float())
            diff = y - p_ens
            w = w5.expand_as(y)
            ss_res += (w * diff * diff).sum()
            sum_w += w.sum()
            sum_wy += (w * y).sum()
            sum_wy2 += (w * y * y).sum()

        mu = sum_wy / (sum_w + 1e-12)
        ss_tot = sum_wy2 - sum_w * mu * mu
        return float((1.0 - ss_res / (ss_tot + 1e-12)).item())

    results: list[dict[str, Any]] = []
    for run_idx, run in enumerate(runs):
        fold_states = [s for s in run if int(s.get("fold_idx", -1)) == int(fold_idx)]
        if not fold_states:
            raise ValueError(f"Missing fold {fold_idx} in checkpoint states.")
        if int(k_weights) > 0:
            fold_states = [_resolve_kweights_state(s, int(k_weights)) for s in fold_states]
        models = [_build_model_from_state(backbone, s, device, backbone_dtype) for s in fold_states]
        per_model_scores = [float(eval_global_wr2(m, dl, w_vec, device=device)) for m in models]
        ens_score = _eval_ensemble(models)
        results.append(
            dict(
                run_idx=int(run_idx),
                fold_idx=int(fold_idx),
                n_models=int(len(models)),
                per_model_scores=per_model_scores,
                ensemble_score=float(ens_score),
            )
        )

    return results


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
    k_weights: int = 0,
) -> torch.Tensor:
    runs = _normalize_runs(states)
    if int(k_weights) > 0:
        runs = [[_resolve_kweights_state(s, int(k_weights)) for s in run] for run in runs]
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
    backbone.eval()

    outer_agg = str(outer_agg).lower()

    def _predict_with_models(models: list[torch.nn.Module]) -> torch.Tensor:
        for model in models:
            model.eval()

        preds: list[torch.Tensor] = []
        ctx = autocast_context(device, dtype=trainable_dtype)
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
                with torch.no_grad(), ctx:
                    if x.ndim == 6:
                        b, t, tiles, c, h, w = x.shape
                        if tiles != 2:
                            raise ValueError(f"Expected tiles=2, got {tiles}.")
                        x_tta = x.view(b * t, tiles, c, h, w)
                        p_raw = model(x_tta).float()
                        pred_space = getattr(model, "pred_space", "log")
                        p = _pred_to_grams(p_raw, pred_space, clamp=True)
                        p = p.view(b, t, -1)
                        preds_models.append(_agg_tta(p, tta_agg))
                    elif x.ndim == 5:
                        p_raw = model(x).float()
                        pred_space = getattr(model, "pred_space", "log")
                        p = _pred_to_grams(p_raw, pred_space, clamp=True)
                        preds_models.append(p)
                    else:
                        raise ValueError(f"Expected [B,2,C,H,W] or [B,T,2,C,H,W], got {tuple(x.shape)}")

            p_ens = _agg_stack(preds_models, inner_agg)
            p_ens = _postprocess_mass_balance(p_ens)
            preds.append(p_ens.detach().cpu())

        return torch.cat(preds, dim=0)

    if outer_agg == "flatten":
        flat_states = [s for run in runs for s in run]
        models = [_build_model_from_state(backbone, s, device, backbone_dtype) for s in flat_states]
        preds = _predict_with_models(models)
        return _postprocess_mass_balance(preds)
    if outer_agg in ("mean", "median"):
        preds_runs: list[torch.Tensor] = []
        for run in runs:
            models = [_build_model_from_state(backbone, s, device, backbone_dtype) for s in run]
            preds_runs.append(_predict_with_models(models))
        preds = _agg_stack(preds_runs, outer_agg)
        return _postprocess_mass_balance(preds)
    raise ValueError(f"Unknown outer_agg: {outer_agg}")
