from __future__ import annotations

from typing import Any
from pathlib import Path
import glob

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

from .amp import autocast_context
from .config import DEFAULTS, DEFAULT_LOSS_WEIGHTS, TARGETS, default_num_workers, parse_dtype, neck_num_heads_for
from .ensemble_utils import (
    _agg_stack,
    _agg_tta,
    _ensure_tensor_batch,
    _get_tta_n,
    _split_tta_batch,
)
from .model import TiledDINOv3Regressor, TiledDINOv3Regressor3
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
        raise ValueError("Non-tiled models are no longer supported. Set tiled_inp=True.")
    if not name or name in ("tiled_base", "tiled"):
        return TiledDINOv3Regressor
    if name in ("tiled_sum3", "tiled_mass3", "tiled_3sum", "tiled_3"):
        return TiledDINOv3Regressor3
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


def _trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for name in ("neck", "head", "norm"):
        part = getattr(model, name, None)
        if part is not None:
            for p in part.parameters():
                if p.requires_grad:
                    params.append(p)
    return params


def _resolve_param_spec(
    model: torch.nn.Module,
    param_spec: Any | None,
) -> list[torch.nn.Parameter]:
    if param_spec is None:
        return _trainable_params(model)
    if callable(param_spec):
        params = list(param_spec(model))
    elif isinstance(param_spec, (list, tuple)):
        if not param_spec:
            return []
        if all(isinstance(x, str) for x in param_spec):
            named = dict(model.named_parameters())
            params = []
            for key in param_spec:
                if key in named:
                    params.append(named[key])
                    continue
                part = getattr(model, str(key), None)
                if part is None:
                    raise ValueError(f"Unknown param spec '{key}'.")
                if isinstance(part, torch.nn.Module):
                    params.extend([p for p in part.parameters() if p.requires_grad])
                else:
                    raise ValueError(f"Param spec '{key}' is not a module or parameter name.")
        else:
            raise ValueError("param_spec list must contain only strings or be a callable.")
    else:
        raise ValueError("param_spec must be None, a callable, or a list of strings.")
    return [p for p in params if p.requires_grad]


def _parse_ttt(ttt: dict[str, Any] | tuple[int, float, float] | None) -> tuple[int, float, float]:
    if ttt is None:
        ttt = DEFAULTS.get("ttt", {})
    if isinstance(ttt, dict):
        return int(ttt.get("steps", 0)), float(ttt.get("lr", 1e-4)), float(ttt.get("beta", 0.0))
    if isinstance(ttt, (list, tuple)) and len(ttt) == 3:
        return int(ttt[0]), float(ttt[1]), float(ttt[2])
    raise ValueError("ttt must be a dict or (steps, lr, beta) tuple.")


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
    if model_cls is TiledDINOv3Regressor3:
        model_kwargs["head_style"] = head_style
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
    ttt: dict[str, Any] | tuple[int, float, float] | None = None,
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
    ttt_steps, ttt_lr, ttt_beta = _parse_ttt(ttt)
    use_ttt = ttt_steps > 0 and ttt_lr > 0.0

    def _predict_with_models(models: list[torch.nn.Module]) -> torch.Tensor:
        for model in models:
            if hasattr(model, "set_train"):
                model.set_train(False)
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
                snap = None
                if use_ttt:
                    params = _trainable_params(model)
                    if params:
                        snap = [p.detach().clone() for p in params]
                        if hasattr(model, "set_train"):
                            model.set_train(True)
                        for _ in range(int(ttt_steps)):
                            with autocast_context(device, dtype=trainable_dtype):
                                if x.ndim == 5:
                                    x_tta, _ = _split_tta_batch(x)
                                    p1 = model(x_tta)
                                    p2 = model(x_tta)
                                else:
                                    p1 = model(x)
                                    p2 = model(x)
                                loss = ((p1.float() - p2.float()) ** 2).mean()
                                if ttt_beta > 0.0:
                                    reg = torch.zeros((), device=x.device)
                                    for p, s in zip(params, snap):
                                        reg = reg + (p - s).pow(2).mean()
                                    loss = loss + float(ttt_beta) * reg
                            grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
                            with torch.no_grad():
                                for p, g in zip(params, grads):
                                    if g is not None:
                                        p.add_(g, alpha=-float(ttt_lr))
                        if hasattr(model, "set_train"):
                            model.set_train(False)
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
                if snap is not None:
                    with torch.no_grad():
                        for p, s in zip(_trainable_params(model), snap):
                            p.copy_(s)

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
    ttt: dict[str, Any] | tuple[int, float, float] | None = None,
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
    ttt_steps, ttt_lr, ttt_beta = _parse_ttt(ttt)
    use_ttt = ttt_steps > 0 and ttt_lr > 0.0

    def _predict_with_models(models: list[torch.nn.Module]) -> torch.Tensor:
        for model in models:
            if hasattr(model, "set_train"):
                model.set_train(False)
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
                snap = None
                if use_ttt:
                    params = _trainable_params(model)
                    if params:
                        snap = [p.detach().clone() for p in params]
                        if hasattr(model, "set_train"):
                            model.set_train(True)
                        for _ in range(int(ttt_steps)):
                            with autocast_context(device, dtype=trainable_dtype):
                                if x.ndim == 6:
                                    b, t, tiles, c, h, w = x.shape
                                    if tiles != 2:
                                        raise ValueError(f"Expected tiles=2, got {tiles}.")
                                    x_tta = x.view(b * t, tiles, c, h, w)
                                    p1 = model(x_tta)
                                    p2 = model(x_tta)
                                else:
                                    p1 = model(x)
                                    p2 = model(x)
                                loss = ((p1.float() - p2.float()) ** 2).mean()
                                if ttt_beta > 0.0:
                                    reg = torch.zeros((), device=x.device)
                                    for p, s in zip(params, snap):
                                        reg = reg + (p - s).pow(2).mean()
                                    loss = loss + float(ttt_beta) * reg
                            grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
                            with torch.no_grad():
                                for p, g in zip(params, grads):
                                    if g is not None:
                                        p.add_(g, alpha=-float(ttt_lr))
                        if hasattr(model, "set_train"):
                            model.set_train(False)
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
                if snap is not None:
                    with torch.no_grad():
                        for p, s in zip(_trainable_params(model), snap):
                            p.copy_(s)

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


def _wr2_stats_init(device: str | torch.device) -> dict[str, torch.Tensor]:
    return dict(
        ss_res=torch.zeros((), device=device),
        sum_w=torch.zeros((), device=device),
        sum_wy=torch.zeros((), device=device),
        sum_wy2=torch.zeros((), device=device),
    )


def _wr2_stats_update(
    stats: dict[str, torch.Tensor],
    y: torch.Tensor,
    p: torch.Tensor,
    w5: torch.Tensor,
) -> None:
    w = w5.expand_as(y)
    diff = y - p
    stats["ss_res"] += (w * diff * diff).sum()
    stats["sum_w"] += w.sum()
    stats["sum_wy"] += (w * y).sum()
    stats["sum_wy2"] += (w * y * y).sum()


def _wr2_from_stats(stats: dict[str, torch.Tensor]) -> float:
    mu = stats["sum_wy"] / (stats["sum_w"] + 1e-12)
    ss_tot = stats["sum_wy2"] - stats["sum_w"] * mu * mu
    return (1.0 - stats["ss_res"] / (ss_tot + 1e-12)).item()


def _make_cv_iter(wide_df, cv_params: dict[str, Any] | None) -> list[tuple[Any, Any]]:
    if cv_params is None:
        cv_params = DEFAULTS.get("cv_params", {})
    if not isinstance(cv_params, dict):
        raise ValueError("cv_params must be a dict.")
    mode = str(cv_params.get("mode", "gkf")).lower()
    if mode != "gkf":
        raise ValueError(f"Unsupported cv mode: {mode}")
    if "cv_seed" not in cv_params:
        raise ValueError("cv_params must include 'cv_seed'.")
    n_splits = int(cv_params.get("n_splits", DEFAULTS["cv_params"]["n_splits"]))
    groups = wide_df["Sampling_Date"].values
    gkf = GroupKFold(n_splits=int(n_splits), shuffle=True, random_state=int(cv_params["cv_seed"]))
    return list(gkf.split(wide_df, groups=groups))


def ttt_sweep_cv(
    dataset,
    wide_df,
    backbone,
    *,
    pt_paths: list[str] | str | None = None,
    states: Any | None = None,
    cv_params: dict[str, Any] | None = None,
    sweeps: list[dict[str, Any]] | None = None,
    batch_size: int = 64,
    num_workers: int | None = None,
    device: str | torch.device = "cuda",
    backbone_dtype: str | torch.dtype | None = None,
    trainable_dtype: str | torch.dtype | None = None,
    tta_agg: str = "mean",
    inner_agg: str = "mean",
    outer_agg: str = "mean",
) -> list[dict[str, Any]]:
    if sweeps is None or not sweeps:
        raise ValueError("sweeps must be a non-empty list.")
    if states is None:
        if pt_paths is None:
            raise ValueError("Provide states or pt_paths.")
        runs = load_ensemble_states(pt_paths)
    else:
        runs = _normalize_runs(states)
    if not runs:
        raise ValueError("No runs found for TTT sweep.")
    _require_tiled_runs(runs)

    if num_workers is None:
        num_workers = default_num_workers()
    if backbone_dtype is None:
        backbone_dtype = DEFAULTS["backbone_dtype"]
    if isinstance(backbone_dtype, str):
        backbone_dtype = parse_dtype(backbone_dtype)
    if trainable_dtype is None:
        trainable_dtype = DEFAULTS["trainable_dtype"]
    if isinstance(trainable_dtype, str):
        trainable_dtype = parse_dtype(trainable_dtype)

    cv_iter = _make_cv_iter(wide_df, cv_params)
    w_vec = torch.as_tensor(DEFAULT_LOSS_WEIGHTS, dtype=torch.float32, device=device).view(1, -1)

    outer_agg = str(outer_agg).lower()
    inner_agg = str(inner_agg).lower()

    def _forward_model(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 6:
            b, t, tiles, c, h, w = x.shape
            if tiles != 2:
                raise ValueError(f"Expected tiles=2, got {tiles}.")
            x_tta = x.view(b * t, tiles, c, h, w)
            p_raw = model(x_tta).float()
            pred_space = getattr(model, "pred_space", "log")
            p = _pred_to_grams(p_raw, pred_space, clamp=True)
            p = p.view(b, t, -1)
            return _agg_tta(p, tta_agg)
        if x.ndim == 5:
            p_raw = model(x).float()
            pred_space = getattr(model, "pred_space", "log")
            return _pred_to_grams(p_raw, pred_space, clamp=True)
        raise ValueError(f"Expected tiled batch [B,2,C,H,W] or [B,T,2,C,H,W], got {tuple(x.shape)}")

    def _eval_fold(
        fold_runs: list[list[dict[str, Any]]],
        va_idx,
        fold_idx: int,
        task,
        param_spec,
        steps: int,
        lr: float,
        beta: float,
        bs: int,
    ) -> tuple[float, float, float, float]:
        subset = Subset(dataset, va_idx)
        tta_n = _get_tta_n(subset)
        tile_n = 2
        bs_eff = max(1, int(bs) // int(tile_n * max(1, tta_n)))
        dl = DataLoader(
            subset,
            shuffle=False,
            batch_size=int(bs_eff),
            pin_memory=str(device).startswith("cuda"),
            num_workers=int(num_workers),
            persistent_workers=(num_workers > 0),
        )

        def _make_models(states_list: list[dict[str, Any]]) -> list[torch.nn.Module]:
            return [_build_model_from_state(backbone, s, device, backbone_dtype) for s in states_list]

        def _make_params(models: list[torch.nn.Module]) -> list[list[torch.nn.Parameter]]:
            return [_resolve_param_spec(m, param_spec) for m in models]

        use_ttt = steps > 0 and lr > 0.0
        stats_base = _wr2_stats_init(device)
        stats_ttt = _wr2_stats_init(device)
        ssl_sum = 0.0
        reg_sum = 0.0
        ssl_n = 0

        if outer_agg == "flatten":
            flat_states = [s for run in fold_runs for s in run]
            models = _make_models(flat_states)
            params_list = _make_params(models)

            for model in models:
                if hasattr(model, "set_train"):
                    model.set_train(False)
                model.eval()

            for batch_idx, batch in enumerate(dl):
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    x, y_log = batch[0], batch[1]
                else:
                    raise ValueError("TTT sweep requires (x, y_log) batches.")
                x = x.to(device, non_blocking=True)
                y_log = y_log.to(device, non_blocking=True)
                y = torch.expm1(y_log.float())

                base_preds_models: list[torch.Tensor] = []
                ttt_preds_models: list[torch.Tensor] = []
                for model_idx, (model, params) in enumerate(zip(models, params_list)):
                    with torch.no_grad(), autocast_context(device, dtype=trainable_dtype):
                        p_base = _forward_model(model, x)
                    base_preds_models.append(p_base)

                    if use_ttt and params:
                        snap = [p.detach().clone() for p in params]
                        for step_idx in range(int(steps)):
                            ctx = dict(
                                device=device,
                                trainable_dtype=trainable_dtype,
                                backbone_dtype=backbone_dtype,
                                fold_idx=int(fold_idx),
                                model_idx=int(model_idx),
                                batch_idx=int(batch_idx),
                                step_idx=int(step_idx),
                            )
                            with torch.enable_grad(), autocast_context(device, dtype=trainable_dtype):
                                loss_task = task(model, x, ctx)
                                if isinstance(loss_task, (tuple, list)):
                                    loss_task = loss_task[0]
                                loss_task = loss_task.float()
                                reg_val = torch.zeros((), device=x.device)
                                if beta > 0.0:
                                    for p, s in zip(params, snap):
                                        reg_val = reg_val + (p - s).pow(2).mean()
                                loss = loss_task + float(beta) * reg_val
                            ssl_sum += float(loss_task.detach().item())
                            reg_sum += float((float(beta) * reg_val).detach().item())
                            ssl_n += 1
                            grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
                            with torch.no_grad():
                                for p, g in zip(params, grads):
                                    if g is not None:
                                        p.add_(g, alpha=-float(lr))
                        if hasattr(model, "set_train"):
                            model.set_train(False)
                        model.eval()
                        with torch.no_grad(), autocast_context(device, dtype=trainable_dtype):
                            p_ttt = _forward_model(model, x)
                        with torch.no_grad():
                            for p, s in zip(params, snap):
                                p.copy_(s)
                    else:
                        p_ttt = p_base
                    ttt_preds_models.append(p_ttt)

                p_base = _agg_stack(base_preds_models, inner_agg)
                p_ttt = _agg_stack(ttt_preds_models, inner_agg)
                _wr2_stats_update(stats_base, y, p_base, w_vec)
                _wr2_stats_update(stats_ttt, y, p_ttt, w_vec)

            mean_ssl = ssl_sum / max(ssl_n, 1)
            mean_reg = reg_sum / max(ssl_n, 1)
            return _wr2_from_stats(stats_base), _wr2_from_stats(stats_ttt), float(mean_ssl), float(mean_reg)

        run_models: list[list[torch.nn.Module]] = []
        run_params: list[list[list[torch.nn.Parameter]]] = []
        for run_states in fold_runs:
            models = _make_models(run_states)
            for model in models:
                if hasattr(model, "set_train"):
                    model.set_train(False)
                model.eval()
            run_models.append(models)
            run_params.append(_make_params(models))

        for batch_idx, batch in enumerate(dl):
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x, y_log = batch[0], batch[1]
            else:
                raise ValueError("TTT sweep requires (x, y_log) batches.")
            x = x.to(device, non_blocking=True)
            y_log = y_log.to(device, non_blocking=True)
            y = torch.expm1(y_log.float())

            run_base: list[torch.Tensor] = []
            run_ttt: list[torch.Tensor] = []
            for run_idx, (models, params_list) in enumerate(zip(run_models, run_params)):
                base_preds_models: list[torch.Tensor] = []
                ttt_preds_models: list[torch.Tensor] = []
                for model_idx, (model, params) in enumerate(zip(models, params_list)):
                    with torch.no_grad(), autocast_context(device, dtype=trainable_dtype):
                        p_base = _forward_model(model, x)
                    base_preds_models.append(p_base)

                    if use_ttt and params:
                        snap = [p.detach().clone() for p in params]
                        for step_idx in range(int(steps)):
                            ctx = dict(
                                device=device,
                                trainable_dtype=trainable_dtype,
                                backbone_dtype=backbone_dtype,
                                fold_idx=int(fold_idx),
                                model_idx=int(model_idx),
                                run_idx=int(run_idx),
                                batch_idx=int(batch_idx),
                                step_idx=int(step_idx),
                            )
                            with torch.enable_grad(), autocast_context(device, dtype=trainable_dtype):
                                loss_task = task(model, x, ctx)
                                if isinstance(loss_task, (tuple, list)):
                                    loss_task = loss_task[0]
                                loss_task = loss_task.float()
                                reg_val = torch.zeros((), device=x.device)
                                if beta > 0.0:
                                    for p, s in zip(params, snap):
                                        reg_val = reg_val + (p - s).pow(2).mean()
                                loss = loss_task + float(beta) * reg_val
                            ssl_sum += float(loss_task.detach().item())
                            reg_sum += float((float(beta) * reg_val).detach().item())
                            ssl_n += 1
                            grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
                            with torch.no_grad():
                                for p, g in zip(params, grads):
                                    if g is not None:
                                        p.add_(g, alpha=-float(lr))
                        if hasattr(model, "set_train"):
                            model.set_train(False)
                        model.eval()
                        with torch.no_grad(), autocast_context(device, dtype=trainable_dtype):
                            p_ttt = _forward_model(model, x)
                        with torch.no_grad():
                            for p, s in zip(params, snap):
                                p.copy_(s)
                    else:
                        p_ttt = p_base
                    ttt_preds_models.append(p_ttt)

                run_base.append(_agg_stack(base_preds_models, inner_agg))
                run_ttt.append(_agg_stack(ttt_preds_models, inner_agg))

            p_base = _agg_stack(run_base, outer_agg)
            p_ttt = _agg_stack(run_ttt, outer_agg)
            _wr2_stats_update(stats_base, y, p_base, w_vec)
            _wr2_stats_update(stats_ttt, y, p_ttt, w_vec)

        mean_ssl = ssl_sum / max(ssl_n, 1)
        mean_reg = reg_sum / max(ssl_n, 1)
        return _wr2_from_stats(stats_base), _wr2_from_stats(stats_ttt), float(mean_ssl), float(mean_reg)

    results: list[dict[str, Any]] = []
    for sweep in tqdm(sweeps, desc="TTT sweeps"):
        task = sweep.get("task")
        if task is None:
            raise ValueError("Each sweep must include a 'task'.")
        name = str(sweep.get("name", getattr(task, "name", task.__class__.__name__)))
        param_spec = sweep.get("params")
        steps = int(sweep.get("steps", 0))
        lr = float(sweep.get("lr", 0.0))
        beta = float(sweep.get("beta", 0.0))
        bs = int(sweep.get("batch_size", batch_size))

        fold_base: list[float] = []
        fold_ttt: list[float] = []
        fold_delta: list[float] = []
        fold_ssl: list[float] = []
        fold_reg: list[float] = []

        fold_iter = tqdm(cv_iter, desc=f"{name} folds", leave=False)
        for fold_idx, (_, va_idx) in enumerate(fold_iter):
            fold_runs: list[list[dict[str, Any]]] = []
            for run in runs:
                fold_states = [s for s in run if int(s.get("fold_idx", -1)) == int(fold_idx)]
                if not fold_states:
                    raise ValueError(f"Missing fold {fold_idx} in checkpoint states.")
                fold_runs.append(fold_states)

            base_score, ttt_score, ssl_mean, reg_mean = _eval_fold(
                fold_runs,
                va_idx,
                int(fold_idx),
                task,
                param_spec,
                steps,
                lr,
                beta,
                bs,
            )
            fold_base.append(float(base_score))
            fold_ttt.append(float(ttt_score))
            fold_delta.append(float(ttt_score - base_score))
            fold_ssl.append(float(ssl_mean))
            fold_reg.append(float(reg_mean))

        mean_base = float(sum(fold_base) / max(len(fold_base), 1))
        mean_ttt = float(sum(fold_ttt) / max(len(fold_ttt), 1))
        mean_delta = float(sum(fold_delta) / max(len(fold_delta), 1))
        mean_ssl = float(sum(fold_ssl) / max(len(fold_ssl), 1))
        mean_reg = float(sum(fold_reg) / max(len(fold_reg), 1))

        results.append(
            dict(
                name=name,
                steps=steps,
                lr=lr,
                beta=beta,
                batch_size=int(bs),
                inner_agg=str(inner_agg),
                outer_agg=str(outer_agg),
                fold_base=fold_base,
                fold_ttt=fold_ttt,
                fold_delta=fold_delta,
                fold_ssl_loss=fold_ssl,
                fold_reg_loss=fold_reg,
                mean_base=mean_base,
                mean_ttt=mean_ttt,
                mean_delta=mean_delta,
                mean_ssl_loss=mean_ssl,
                mean_reg_loss=mean_reg,
            )
        )

    return results
