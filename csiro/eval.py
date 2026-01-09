from __future__ import annotations

from typing import Any
from pathlib import Path
import glob

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold

from .amp import autocast_context
from .config import DEFAULTS, DEFAULT_LOSS_WEIGHTS, default_num_workers, parse_dtype, neck_num_heads_for
from .ensemble_utils import (
    _agg_stack,
    _agg_tta,
    _ensure_tensor_batch,
    _get_tta_n,
    _split_tta_batch,
)
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
    model_cls = TiledDINOv3Regressor if use_tiled else DINOv3Regressor
    backbone_size = str(state.get("backbone_size", DEFAULTS.get("backbone_size", "b")))
    neck_num_heads = int(state.get("neck_num_heads", neck_num_heads_for(backbone_size)))
    model = model_cls(
        backbone,
        hidden=int(state["head_hidden"]),
        drop=float(state["head_drop"]),
        depth=int(state["head_depth"]),
        num_neck=int(state["num_neck"]),
        neck_num_heads=int(neck_num_heads),
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
    backbone_size = str(state.get("backbone_size", DEFAULTS.get("backbone_size", "b")))
    neck_num_heads = int(state.get("neck_num_heads", neck_num_heads_for(backbone_size)))

    model = DINOv3Regressor(
        backbone,
        hidden=int(_require(state, "head_hidden")),
        drop=float(_require(state, "head_drop")),
        depth=int(_require(state, "head_depth")),
        num_neck=int(_require(state, "num_neck")),
        neck_num_heads=int(neck_num_heads),
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
    if hasattr(model, "set_train"):
        model.set_train(False)
    model.eval()
    return model




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

    def _predict_with_models(models: list[DINOv3Regressor]) -> torch.Tensor:
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
                if snap is not None:
                    with torch.no_grad():
                        for p, s in zip(_trainable_params(model), snap):
                            p.copy_(s)

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

    def _predict_with_models(models: list[DINOv3Regressor]) -> torch.Tensor:
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
                if snap is not None:
                    with torch.no_grad():
                        for p, s in zip(_trainable_params(model), snap):
                            p.copy_(s)

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
    ttt: dict[str, Any] | tuple[int, float, float] | None = None,
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
        ttt=ttt,
    )


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
            p_log = model(x_tta).float()
            p = torch.expm1(p_log).clamp_min(0.0)
            p = p.view(b, t, -1)
            return _agg_tta(p, tta_agg)
        if x.ndim == 5:
            p_log = model(x).float()
            return torch.expm1(p_log).clamp_min(0.0)
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
    ) -> tuple[float, float]:
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
                                loss = task(model, x, ctx)
                                if isinstance(loss, (tuple, list)):
                                    loss = loss[0]
                                loss = loss.float()
                                if beta > 0.0:
                                    reg = torch.zeros((), device=x.device)
                                    for p, s in zip(params, snap):
                                        reg = reg + (p - s).pow(2).mean()
                                    loss = loss + float(beta) * reg
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

            return _wr2_from_stats(stats_base), _wr2_from_stats(stats_ttt)

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
                                loss = task(model, x, ctx)
                                if isinstance(loss, (tuple, list)):
                                    loss = loss[0]
                                loss = loss.float()
                                if beta > 0.0:
                                    reg = torch.zeros((), device=x.device)
                                    for p, s in zip(params, snap):
                                        reg = reg + (p - s).pow(2).mean()
                                    loss = loss + float(beta) * reg
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

        return _wr2_from_stats(stats_base), _wr2_from_stats(stats_ttt)

    results: list[dict[str, Any]] = []
    for sweep in sweeps:
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

        for fold_idx, (_, va_idx) in enumerate(cv_iter):
            fold_runs: list[list[dict[str, Any]]] = []
            for run in runs:
                fold_states = [s for s in run if int(s.get("fold_idx", -1)) == int(fold_idx)]
                if not fold_states:
                    raise ValueError(f"Missing fold {fold_idx} in checkpoint states.")
                fold_runs.append(fold_states)

            base_score, ttt_score = _eval_fold(
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

        mean_base = float(sum(fold_base) / max(len(fold_base), 1))
        mean_ttt = float(sum(fold_ttt) / max(len(fold_ttt), 1))
        mean_delta = float(sum(fold_delta) / max(len(fold_delta), 1))

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
                mean_base=mean_base,
                mean_ttt=mean_ttt,
                mean_delta=mean_delta,
            )
        )

    return results
