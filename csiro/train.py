from __future__ import annotations

import copy
import math
import os
from typing import Any, Callable

import numpy as np
import torch
import torchvision.transforms as T
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .amp import autocast_context, grad_scaler
from .config import TARGETS, default_num_workers, DEFAULT_LOSS_WEIGHTS, DEFAULTS, parse_dtype, neck_num_heads_for
from .data import TransformView, TiledSharedTransformView, TiledTransformView
from .ensemble_utils import (
    _agg_stack,
    _agg_tta,
    _ensure_tensor_batch,
    _get_tta_n,
    _split_tta_batch,
)
from .losses import NegativityPenaltyLoss, WeightedMSELoss, WeightedSmoothL1Loss
from .metrics import eval_global_wr2
from .model import (
    TiledDINOv3Regressor,
    TiledDINOv3Regressor3,
    TiledDINOv3RegressorStitched3,
    FullDINOv3RegressorRect3,
)
from .transforms import base_train_comp, post_tfms
from .utils import build_color_jitter_sweep, filter_kwargs

def cos_sin_lr(ep: int, epochs: int, lr_start: float, lr_final: float) -> float:
    if epochs <= 1:
        return lr_final
    t = (ep - 1) / (epochs - 1)
    return lr_final + 0.5 * (lr_start - lr_final) * (1.0 + math.cos(math.pi * t))

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


def set_optimizer_lr(opt, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = lr


def _trainable_blocks(m: torch.nn.Module) -> list[torch.nn.Module]:
    parts: list[torch.nn.Module] = []
    for name in ("neck", "head", "patch_head", "norm"):
        part = getattr(m, name, None)
        if part is not None:
            parts.append(part)
    return parts


def _trainable_params_list(m: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for b in _trainable_blocks(m):
        for p in b.parameters():
            if p.requires_grad:
                params.append(p)
    return params


def _save_parts(m: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    state: dict[str, dict[str, torch.Tensor]] = {}
    for name in ("neck", "head", "patch_head", "norm"):
        part = getattr(m, name, None)
        if part is not None:
            state[name] = {k: v.detach().cpu() for k, v in part.state_dict().items()}
    return state


def _load_parts(m: torch.nn.Module, state: dict[str, dict[str, torch.Tensor]]) -> None:
    for name in ("neck", "head", "patch_head", "norm"):
        part = getattr(m, name, None)
        if part is not None and name in state:
            part.load_state_dict(state[name], strict=True)


def _avg_states(states: list[dict[str, dict[str, torch.Tensor]]]) -> dict[str, dict[str, torch.Tensor]]:
    if not states:
        raise ValueError("Cannot average empty state list.")
    out: dict[str, dict[str, torch.Tensor]] = {}
    for name in states[0].keys():
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
            model_kwargs["neck_drop"] = float(state.get("neck_drop", DEFAULTS.get("neck_drop", 0.0)))
            model_kwargs["drop_path"] = state.get("drop_path", DEFAULTS.get("drop_path", None))
            model_kwargs["rope_rescale"] = state.get("rope_rescale", DEFAULTS.get("rope_rescale", None))
            model_kwargs["neck_ffn"] = bool(state.get("neck_ffn", DEFAULTS.get("neck_ffn", True)))
            if model_cls is TiledDINOv3RegressorStitched3:
                model_kwargs["neck_pool"] = bool(state.get("neck_pool", DEFAULTS.get("neck_pool", False)))
                model_kwargs["norm_bb_out"] = bool(state.get("norm_bb_out", DEFAULTS.get("norm_bb_out", False)))
    model = model_cls(**model_kwargs).to(device)
    model.model_name = model_name or ("tiled_base" if use_tiled else "base")
    _load_parts(model, state["parts"])
    return model

def train_one_fold(
    *,
    ds_tr_view,
    ds_va_view,
    backbone,
    tr_idx,
    va_idx,
    wd: float = 1e-4,
    fold_idx: int = 0,
    epochs: int = 5,
    warmup_steps: int | None = None,
    lr_start: float = 3e-4,
    lr_final: float = 5e-5,
    batch_size: int = 128,
    clip_val: float | None = 3.0,
    device: str = "cuda",
    save_path: str | None = None,
    verbose: bool = False,
    early_stopping: int = 6,
    head_hidden: int = 1024,
    head_depth: int = 2,
    head_drop: float = 0.1,
    num_neck: int = 0,
    neck_num_heads: int | None = None,
    num_workers: int | None = None,
    backbone_dtype: str | torch.dtype | None = None,
    trainable_dtype: str | torch.dtype | None = None,
    comet_exp: Any | None = None,
    skip_log_first_n: int = 5,
    curr_fold: int = 0,
    model_idx: int = 0,
    return_state: bool = False,
    attempt_idx: int | None = None,
    attempt_max: int | None = None,
    tiled_inp: bool = False,
    val_freq: int = 1,
    backbone_size: str | None = None,
    mixup: tuple[float, float] | dict[str, float] | None = None,
    model_name: str | None = None,
    head_style: str | None = None,
    pred_space: str | None = None,
    loss_weights: list[float] | tuple[float, ...] | None = None,
    huber_beta: float | None = None,
    tau_neg: float | None = None,
    tau_patch: float | None = None,
    out_format: str | None = None,
    neck_rope: bool | None = None,
    neck_pool: bool | None = None,
    norm_bb_out: bool | None = None,
    rope_rescale: float | None = None,
    neck_drop: float | None = None,
    drop_path: dict[str, float] | None = None,
    neck_ffn: bool | None = None,
    top_k_weights: int | None = None,
    last_k_weights: int | None = None,
    k_weights_val: bool | None = None,
) -> float | dict[str, Any]:
    tr_subset = Subset(ds_tr_view, tr_idx)
    va_subset = Subset(ds_va_view, va_idx)

    num_workers = default_num_workers() if num_workers is None else int(num_workers)
    tile_n = 2 if tiled_inp else 1
    train_bs = max(1, int(batch_size) // int(tile_n))
    val_bs = int(train_bs)
    tta_n = _get_tta_n(ds_va_view)
    if tta_n > 1:
        val_bs = max(1, int(val_bs) // int(tta_n))
    dl_kwargs = dict(
        batch_size=int(train_bs),
        pin_memory=str(device).startswith("cuda"),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    dl_tr = DataLoader(tr_subset, shuffle=True, **dl_kwargs)
    dl_va = DataLoader(va_subset, shuffle=False, **{**dl_kwargs, "batch_size": int(val_bs)})

    if backbone_dtype is None:
        backbone_dtype = parse_dtype(DEFAULTS["backbone_dtype"])
    elif isinstance(backbone_dtype, str):
        backbone_dtype = parse_dtype(backbone_dtype)
    if trainable_dtype is None:
        trainable_dtype = parse_dtype(DEFAULTS["trainable_dtype"])
    elif isinstance(trainable_dtype, str):
        trainable_dtype = parse_dtype(trainable_dtype)

    if mixup is None:
        mixup = DEFAULTS.get("mixup", (0.0, 0.0))
    mixup_p = 0.0
    mixup_p3 = 0.0
    mixup_alpha = 0.0
    if isinstance(mixup, dict):
        mixup_p = float(mixup.get("p", mixup.get("p_mix", 0.0)))
        mixup_p3 = float(mixup.get("p3", mixup.get("p3_mix", 0.0)))
        mixup_alpha = float(mixup.get("alpha", 0.0))
    elif isinstance(mixup, (list, tuple)) and len(mixup) == 2:
        mixup_p = float(mixup[0])
        mixup_alpha = float(mixup[1])
    else:
        raise ValueError("mixup must be a (p, alpha) tuple or dict with p/p3/alpha.")
    if not (0.0 <= mixup_p <= 1.0):
        raise ValueError(f"mixup p must be in [0,1] (got {mixup_p}).")
    if not (0.0 <= mixup_p3 <= 1.0):
        raise ValueError(f"mixup p3 must be in [0,1] (got {mixup_p3}).")
    if mixup_alpha < 0.0:
        raise ValueError(f"mixup alpha must be >= 0 (got {mixup_alpha}).")

    if backbone_size is None:
        backbone_size = str(DEFAULTS.get("backbone_size", "b"))
    if neck_num_heads is None:
        neck_num_heads = int(neck_num_heads_for(backbone_size))

    if model_name is None:
        model_name = str(DEFAULTS.get("model_name", "")).strip()
    model_cls = _resolve_model_class(model_name or None, tiled_inp)
    if pred_space is None:
        pred_space = DEFAULTS.get("pred_space", "log")
    pred_space = _normalize_pred_space(pred_space)
    if pred_space == "gram" and model_cls is TiledDINOv3Regressor:
        raise ValueError("pred_space='gram' is only supported for the 3-output model variants.")
    if head_style is None:
        head_style = DEFAULTS.get("head_style", "single")
    head_style = str(head_style).strip().lower()
    model_kwargs = dict(
        backbone=backbone,
        hidden=int(head_hidden),
        drop=float(head_drop),
        depth=int(head_depth),
        num_neck=int(num_neck),
        neck_num_heads=int(neck_num_heads),
        backbone_dtype=backbone_dtype,
        pred_space=pred_space,
    )
    if model_cls is TiledDINOv3Regressor3:
        model_kwargs["head_style"] = head_style
    if model_cls in (TiledDINOv3RegressorStitched3, FullDINOv3RegressorRect3):
        model_kwargs["head_style"] = head_style
        if out_format is None:
            out_format = DEFAULTS.get("out_format", "cat_cls")
        model_kwargs["out_format"] = str(out_format).strip().lower()
        if neck_rope is None:
            neck_rope = bool(DEFAULTS.get("neck_rope", True))
        model_kwargs["neck_rope"] = bool(neck_rope)
        if model_cls is TiledDINOv3RegressorStitched3:
            if neck_pool is None:
                neck_pool = bool(DEFAULTS.get("neck_pool", False))
            model_kwargs["neck_pool"] = bool(neck_pool)
            if norm_bb_out is None:
                norm_bb_out = bool(DEFAULTS.get("norm_bb_out", False))
            model_kwargs["norm_bb_out"] = bool(norm_bb_out)
        if rope_rescale is None:
            rope_rescale = DEFAULTS.get("rope_rescale", None)
        model_kwargs["rope_rescale"] = rope_rescale
        if neck_drop is None:
            neck_drop = float(DEFAULTS.get("neck_drop", 0.0))
        model_kwargs["neck_drop"] = float(neck_drop)
        if neck_ffn is None:
            neck_ffn = bool(DEFAULTS.get("neck_ffn", True))
        model_kwargs["neck_ffn"] = bool(neck_ffn)
        if drop_path is None:
            drop_path = DEFAULTS.get("drop_path", None)
        model_kwargs["drop_path"] = drop_path
    model = model_cls(**model_kwargs).to(device)
    model.init()
    model.model_name = model_name or ("tiled_base" if tiled_inp else "base")

    if loss_weights is None:
        loss_weights = DEFAULTS.get("loss_weights", DEFAULT_LOSS_WEIGHTS)
    w_loss = torch.as_tensor(loss_weights, dtype=torch.float32)
    eval_w = torch.as_tensor(DEFAULT_LOSS_WEIGHTS, dtype=torch.float32, device=device)
    if pred_space == "gram":
        if huber_beta is None:
            huber_beta = float(DEFAULTS.get("huber_beta", 1.0))
        criterion = WeightedSmoothL1Loss(weights=w_loss, beta=float(huber_beta)).to(device)
    else:
        criterion = WeightedMSELoss(weights=w_loss).to(device)
    if tau_neg is None:
        tau_neg = float(DEFAULTS.get("tau_neg", 0.0))
    neg_criterion = None
    if float(tau_neg) > 0.0:
        neg_criterion = NegativityPenaltyLoss(tau_neg=float(tau_neg), pred_space=pred_space).to(device)

    if tau_patch is None:
        tau_patch = float(DEFAULTS.get("tau_patch", 0.0))
    tau_patch = float(tau_patch)
    use_patch_aux = (
        tau_patch > 0.0
        and pred_space == "log"
        and isinstance(model, TiledDINOv3RegressorStitched3)
    )
    if use_patch_aux and str(getattr(model, "head_style", "single")).strip().lower() != "single":
        raise ValueError("patch aux is only supported for head_style='single'.")
    
    trainable_params = _trainable_params_list(model)
    
    opt = torch.optim.AdamW(trainable_params, lr=float(lr_start), weight_decay=float(wd))
    scaler = grad_scaler(device, dtype=trainable_dtype)

    best_score = -1e9
    best_state = None
    best_opt_state = None
    patience = 0
    if top_k_weights is None:
        top_k_weights = int(DEFAULTS.get("top_k_weights", 0))
    top_k = max(0, int(top_k_weights))
    if last_k_weights is None:
        last_k_weights = int(DEFAULTS.get("last_k_weights", 0))
    last_k = max(0, int(last_k_weights))
    if int(top_k) > 0 and int(last_k) > 0:
        raise ValueError("top_k_weights and last_k_weights are mutually exclusive; set one to 0.")
    topk_list: list[tuple[float, dict[str, dict[str, torch.Tensor]]]] = []
    recent_states: list[dict[str, dict[str, torch.Tensor]]] = []
    best_window_states: list[dict[str, dict[str, torch.Tensor]]] = []
    if k_weights_val is None:
        k_weights_val = bool(DEFAULTS.get("k_weights_val", False))
    k_weights_val = bool(k_weights_val)
    recent_states: list[dict[str, dict[str, torch.Tensor]]] = []
    best_window_states: list[dict[str, dict[str, torch.Tensor]]] = []

    if warmup_steps is None:
        warmup_steps = int(DEFAULTS.get("warmup_steps", 0))
    warmup_steps = max(0, int(warmup_steps))

    val_freq = max(1, int(val_freq))
    attempt_idx = int(attempt_idx) if attempt_idx is not None else 1
    attempt_max = int(attempt_max) if attempt_max is not None else 1
    p_bar = tqdm(range(1, int(epochs) + 1))

    for ep in p_bar:
        warmup_epochs = min(int(warmup_steps), int(epochs))
        if int(epochs) <= int(warmup_epochs):
            raise ValueError("epochs must be > warmup_steps to run at least one evaluation.")
        if warmup_epochs > 0 and int(ep) <= int(warmup_epochs):
            lr = float(lr_start) * (float(ep) / float(warmup_epochs))
        else:
            cosine_epochs = max(int(epochs) - int(warmup_epochs), 1)
            cosine_step = int(ep) - int(warmup_epochs)
            lr = cos_sin_lr(int(cosine_step) + 1, int(cosine_epochs), float(lr_start), float(lr_final))
        set_optimizer_lr(opt, lr)

        model.train()
        running = 0.0
        running_patch = 0.0
        n_seen = 0

        for bi, (x, y_log) in enumerate(dl_tr):
            x = x.to(device, non_blocking=True)
            y_log = y_log.to(device, non_blocking=True)
            y_target = y_log
            if pred_space == "gram":
                y_target = torch.expm1(y_log.float())
            if mixup_p > 0.0 and mixup_alpha > 0.0 and int(x.size(0)) > 1:
                if torch.rand((), device=x.device).item() < mixup_p:
                    bs = int(x.size(0))
                    use_k3 = mixup_p3 > 0.0 and bs > 2 and torch.rand((), device=x.device).item() < mixup_p3
                    if use_k3:
                        perm1 = torch.randperm(bs, device=x.device)
                        perm2 = torch.randperm(bs, device=x.device)
                        x2 = x[perm1]
                        x3 = x[perm2]
                        weights = torch.distributions.Dirichlet(
                            torch.full((3,), float(mixup_alpha), device=x.device)
                        ).sample((bs,))
                        w1, w2, w3 = weights[:, 0], weights[:, 1], weights[:, 2]
                        w_shape = [bs] + [1] * (x.ndim - 1)
                        x = x * w1.view(w_shape) + x2 * w2.view(w_shape) + x3 * w3.view(w_shape)

                        w_y = weights.view(bs, 3)
                        if pred_space == "log":
                            y_lin = torch.expm1(y_log.float()).clamp_min(0.0)
                            y2_lin = torch.expm1(y_log[perm1].float()).clamp_min(0.0)
                            y3_lin = torch.expm1(y_log[perm2].float()).clamp_min(0.0)
                            y_mix = y_lin * w_y[:, [0]] + y2_lin * w_y[:, [1]] + y3_lin * w_y[:, [2]]
                            y_target = torch.log1p(y_mix)
                        else:
                            y2 = y_target[perm1]
                            y3 = y_target[perm2]
                            y_target = y_target * w_y[:, [0]] + y2 * w_y[:, [1]] + y3 * w_y[:, [2]]
                    else:
                        perm = torch.randperm(bs, device=x.device)
                        x2 = x[perm]
                        y2 = y_target[perm]
                        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample((bs,)).to(x.device)
                        lam_x = lam.view([bs] + [1] * (x.ndim - 1))
                        x = x * lam_x + x2 * (1.0 - lam_x)

                        lam_y = lam.view(bs, 1)
                        if pred_space == "log":
                            y_lin = torch.expm1(y_log.float()).clamp_min(0.0)
                            y2_lin = torch.expm1(y2.float()).clamp_min(0.0)
                            y_mix = y_lin * lam_y + y2_lin * (1.0 - lam_y)
                            y_target = torch.log1p(y_mix)
                        else:
                            y_target = y_target * lam_y + y2 * (1.0 - lam_y)

            opt.zero_grad(set_to_none=True)
            with autocast_context(device, dtype=trainable_dtype):
                if use_patch_aux:
                    pred, patch_pred = model.forward_with_patch(x)
                    loss_main = criterion(pred, y_target)
                    loss_patch = criterion(patch_pred, y_target)
                else:
                    pred = model(x)
                    loss_main = criterion(pred, y_target)
                loss = loss_main
                if use_patch_aux:
                    loss = loss + (float(tau_patch) * loss_patch)
                if neg_criterion is not None:
                    loss = loss + neg_criterion(pred)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if clip_val and clip_val > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(clip_val))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if clip_val and clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(clip_val))
                opt.step()

            bs = int(x.size(0))
            running += float(loss_main.detach().item()) * bs
            if use_patch_aux:
                running_patch += float(loss_patch.detach().item()) * bs
            n_seen += bs

        train_loss = running / max(int(n_seen), 1)
        train_patch = running_patch / max(int(n_seen), 1) if use_patch_aux else None
        do_eval = (val_freq == 1) or (int(ep) and int(ep) % int(val_freq) == 0)
        if int(ep) <= int(warmup_epochs):
            do_eval = False
        score = None
        if do_eval:
            model.eval()
            score_curr = float(eval_global_wr2(model, dl_va, eval_w, device=device))
            if top_k > 0:
                state_k = _save_parts(model)
                topk_list.append((float(score_curr), state_k))
                topk_list.sort(key=lambda x: float(x[0]), reverse=True)
                if len(topk_list) > int(top_k):
                    topk_list.pop(-1)
            if last_k > 0:
                state_k = _save_parts(model)
                recent_states.append(state_k)
                if len(recent_states) > int(last_k):
                    recent_states.pop(0)
            score = score_curr
            if k_weights_val and (top_k > 0 or last_k > 0):
                if top_k > 0 and topk_list:
                    state_eval = _avg_states([s for _, s in topk_list])
                elif last_k > 0 and recent_states:
                    state_eval = _avg_states(recent_states)
                else:
                    state_eval = _save_parts(model)
                orig_state = _save_parts(model)
                _load_parts(model, state_eval)
                score = float(eval_global_wr2(model, dl_va, eval_w, device=device))
                _load_parts(model, orig_state)

        if comet_exp is not None and int(ep) > int(skip_log_first_n):
            p = {f"x_train_loss_cv{curr_fold}_m{model_idx}": float(train_loss)}
            if train_patch is not None:
                p[f"x_train_patch_cv{curr_fold}_m{model_idx}"] = float(train_patch)
            if score is not None:
                p[f"x_val_wR2_cv{curr_fold}_m{model_idx}"] = float(score)
            comet_exp.log_metrics(p, step=int(ep))

        if score is not None and score > best_score:
            best_score = float(score)
            patience = 0
            best_state = _save_parts(model)
            best_opt_state = copy.deepcopy(opt.state_dict())
            if last_k > 0 and recent_states:
                best_window_states = list(recent_states)
        else:
            if int(ep) > int(warmup_epochs):
                patience += 1

        s1 = (
            f"Best score: {best_score:.4f} | Patience: {patience:02d}/{int(early_stopping):02d} | "
            f"try={attempt_idx}/{attempt_max} | lr: {lr:6.4f}"
        )
        if score is None:
            s2 = (
                f"[fold {fold_idx} | model {int(model_idx)}] | train_loss={train_loss:.4f} | "
                f"val_wR2=skip | {s1}"
            )
        else:
            s2 = (
                f"[fold {fold_idx} | model {int(model_idx)}] | train_loss={train_loss:.4f} | "
                f"val_wR2={score:.4f} | {s1}"
            )
        if verbose:
            print(s2)
        p_bar.set_postfix_str(s2)

        if patience >= int(early_stopping):
            p_bar.set_postfix_str(s2 + " | Early stopping")
            break

    p_bar.close()

    if best_state is None:
        best_state = _save_parts(model)
        best_score = float(best_score)
    final_state = best_state
    if top_k > 0 and topk_list:
        final_state = _avg_states([s for _, s in topk_list])
    elif last_k > 0 and best_window_states:
        final_state = _avg_states(best_window_states)
        if isinstance(final_state, dict):
            final_state["last_k_states"] = best_window_states
    if save_path and final_state is not None:
        torch.save(final_state, save_path)
    if return_state:
        return {
            "score": float(best_score),
            "best_score": float(best_score),
            "state": final_state,
            "best_state": best_state,
            "best_opt_state": best_opt_state,
            "opt_state": copy.deepcopy(opt.state_dict()),
        }
    return float(best_score)


def eval_global_wr2_ensemble(
    models: list[torch.nn.Module],
    dl_va,
    w_vec: torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    trainable_dtype: str | torch.dtype | None = None,
    tta_agg: str = "mean",
    inner_agg: str = "mean",
    tiled_inp: bool = False,
    comet_exp: Any | None = None,
    curr_fold: int | None = None,
    log_key: str = "1ENS_wR2",
) -> float:
    for model in models:
        if hasattr(model, "set_train"):
            model.eval()
        model.eval()

    w5 = w_vec.to(device).view(1, -1)
    ss_res = torch.zeros((), device=device)
    sum_w = torch.zeros((), device=device)
    sum_wy = torch.zeros((), device=device)
    sum_wy2 = torch.zeros((), device=device)

    if trainable_dtype is None:
        trainable_dtype = parse_dtype(DEFAULTS["trainable_dtype"])
    elif isinstance(trainable_dtype, str):
        trainable_dtype = parse_dtype(trainable_dtype)

    ctx = autocast_context(device, dtype=trainable_dtype)

    for batch in dl_va:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y_log = batch[0], batch[1]
        else:
            raise ValueError("Validation loader must yield (x, y_log).")

        x = x.to(device, non_blocking=True)
        y_log = y_log.to(device, non_blocking=True)

        preds_models: list[torch.Tensor] = []
        for model in models:
            with torch.no_grad(), ctx:
                if tiled_inp:
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
    score = (1.0 - ss_res / (ss_tot + 1e-12)).item()
    if comet_exp is not None:
        try:
            comet_exp.log_metrics({str(f"{log_key}_cv{curr_fold}"): float(score)})
        except Exception:
            pass
    return float(score)




def run_groupkfold_cv(
    *,
    dataset,
    wide_df,
    backbone_dtype: str | torch.dtype | None = None,
    trainable_dtype: str | torch.dtype | None = None,
    comet_exp_name: str | None = None,
    config_name: str = "",
    img_size: int | None = None,
    return_details: bool = False,
    save_output_dir: str | None = None,
    cv_cfg: int | None = None,
    max_folds: int | None = None,
    **train_kwargs,
):
    if "cv_params" in train_kwargs:
        raise ValueError("cv_params is no longer supported; use cv_cfg instead.")
    if cv_cfg is None:
        cv_cfg = int(DEFAULTS.get("cv_cfg", 1))
    cv_cfg = int(cv_cfg)
    if max_folds is None:
        max_folds = DEFAULTS.get("max_folds", None)
    max_folds = None if max_folds is None else int(max_folds)
    cv_resume = bool(train_kwargs.pop("cv_resume", DEFAULTS.get("cv_resume", False)))

    if cv_cfg == 1:
        n_splits = 5
        gkf = GroupKFold(n_splits=int(n_splits), shuffle=True, random_state=126015)
        groups = wide_df["Sampling_Date"].values
        cv_iter = gkf.split(wide_df, groups=groups)
    elif cv_cfg == 2:
        n_splits = 5
        sgkf = StratifiedGroupKFold(n_splits=int(n_splits), shuffle=True, random_state=82947501)
        groups = wide_df["Sampling_Date"].values
        strat = wide_df["State"].astype(str).values
        cv_iter = sgkf.split(wide_df, y=strat, groups=groups)
    else:
        raise ValueError(f"Unknown cv_cfg: {cv_cfg}")

    org_train_kwargs = train_kwargs.copy()
    n_models = int(train_kwargs.pop("n_models", DEFAULTS["n_models"]))
    inp_train_kwargs = filter_kwargs(train_one_fold, org_train_kwargs)
    bcs_range = train_kwargs.pop("bcs_range", DEFAULTS["bcs_range"])
    hue_range = train_kwargs.pop("hue_range", DEFAULTS["hue_range"])
    cutout_p = float(train_kwargs.pop("cutout", DEFAULTS.get("cutout", 0.0)))
    to_gray_p = float(train_kwargs.pop("to_gray", DEFAULTS.get("to_gray", 0.0)))
    train_kwargs.pop("rdrop", None)
    val_bs_override = train_kwargs.pop("val_bs", DEFAULTS.get("val_bs", None))
    tiled_inp = bool(train_kwargs.pop("tiled_inp", DEFAULTS.get("tiled_inp", False)))
    tile_geom_mode = str(train_kwargs.pop("tile_geom_mode", DEFAULTS.get("tile_geom_mode", "shared"))).strip().lower()
    if tile_geom_mode not in ("shared", "independent"):
        raise ValueError(f"tile_geom_mode must be 'shared' or 'independent' (got {tile_geom_mode})")
    run_name = str(train_kwargs.get("run_name", DEFAULTS.get("run_name", ""))).strip()
    if not run_name:
        raise ValueError("run_name must be set (used for artifact naming).")
    model_name = str(train_kwargs.get("model_name", DEFAULTS.get("model_name", "")))
    full_rect = str(model_name).strip().lower() in ("rect_full", "full_rect", "rect")
    if full_rect and tiled_inp:
        raise ValueError("rect_full model requires tiled_inp=False.")
    if full_rect:
        tiled_inp = False
    out_format = str(train_kwargs.get("out_format", DEFAULTS.get("out_format", "cat_cls"))).strip().lower()
    pred_space = _normalize_pred_space(train_kwargs.get("pred_space", DEFAULTS.get("pred_space", "log")))
    head_style = str(train_kwargs.get("head_style", DEFAULTS.get("head_style", "single"))).strip().lower()
    jitter_tfms = build_color_jitter_sweep(
        int(n_models),
        bcs_range=tuple(bcs_range),
        hue_range=tuple(hue_range),
    )
    train_tfms_list = [T.Compose([base_train_comp, t]) for t in jitter_tfms]
    rect_train_tfms_list = [
        T.Compose([T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5), t]) for t in jitter_tfms
    ]
    train_post_ops = [post_tfms()]
    if cutout_p > 0.0:
        train_post_ops.append(T.RandomErasing(p=float(cutout_p)))
    if to_gray_p > 0.0:
        train_post_ops.append(T.RandomGrayscale(p=float(to_gray_p)))
    train_post = T.Compose(train_post_ops)
    use_shared_geom = tiled_inp and tile_geom_mode == "shared"
    img_size_use = int(img_size or DEFAULTS.get("img_size", 512))
    rect_resize = (
        T.Resize((int(img_size_use), int(img_size_use) * 2), antialias=True) if full_rect else None
    )
    if tiled_inp:
        if use_shared_geom:
            ds_va_view = TiledSharedTransformView(
                dataset,
                geom_tfms=None,
                img_size=img_size_use,
                post=post_tfms(),
            )
        else:
            ds_va_view = TiledTransformView(dataset, post_tfms())
    elif full_rect:
        ds_va_view = TransformView(dataset, T.Compose([rect_resize, post_tfms()]))
    else:
        ds_va_view = TransformView(dataset, post_tfms())

    model_name = str(train_kwargs.get("model_name", DEFAULTS.get("model_name", "")))
    safe_name = "".join(c for c in str(run_name).strip() if c.isalnum() or c in "_-")
    if not safe_name:
        raise ValueError("run_name must contain at least one alnum/_/- character.")
    cv_state_path = None
    save_output_path = None
    if save_output_dir is not None:
        state_dir = os.path.join(save_output_dir, "states")
        complete_dir = os.path.join(save_output_dir, "complete")
        cv_state_path = os.path.join(state_dir, f"{safe_name}_cv_state.pt")
        save_output_path = os.path.join(complete_dir, f"{safe_name}.pt")

    val_min_score = float(train_kwargs.get("val_min_score", DEFAULTS.get("val_min_score", 0.0)))
    val_num_retry = int(train_kwargs.get("val_num_retry", DEFAULTS.get("val_num_retry", 1)))
    val_num_retry = max(1, int(val_num_retry))

    fold_scores: list[float] = []
    fold_model_scores: list[list[float]] = []
    fold_states: list[list[dict[str, Any]]] = []
    start_fold = 0
    start_model = 0
    state = None

    if cv_resume:
        if cv_state_path is None:
            raise ValueError("cv_resume=True requires save_output_dir.")
        if os.path.exists(cv_state_path):
            state = torch.load(cv_state_path, map_location="cpu", weights_only=False)
            if state.get("completed", False):
                raise ValueError("Refusing to resume: CV run is already marked completed.")
            fold_scores[:] = [float(x) for x in state.get("fold_scores", [])]
            fold_model_scores[:] = [list(map(float, xs)) for xs in state.get("fold_model_scores", [])]
            fold_states[:] = list(state.get("states", []))
            last_completed = state.get("last_completed_fold")
            last_completed_model = state.get("last_completed_model")
            if last_completed is None:
                last_completed = len(fold_states) - 1
            if last_completed_model is None:
                if fold_states:
                    last_completed_model = len(fold_states[int(last_completed)]) - 1
                else:
                    last_completed_model = -1
            if int(last_completed) < 0:
                start_fold = 0
                start_model = 0
            else:
                start_fold = int(last_completed)
                start_model = int(last_completed_model) + 1
            if start_model >= int(n_models):
                start_fold += 1
                start_model = 0
            print(f"INFO: Resuming from fold {start_fold}, model {start_model}")
            
    exp_key = comet_exp = None
    if comet_exp_name is not None:
        import comet_ml  # type: ignore
        
        if cv_resume and isinstance(state, dict):
            exp_key = state.get("exp_key")

        comet_exp = comet_ml.start(
            api_key=os.getenv("COMET_API_KEY"),
            project_name=comet_exp_name,
            experiment_key=exp_key,
        )
        if hasattr(comet_exp, "get_key"):
            try:
                exp_key = comet_exp.get_key()
            except Exception:
                exp_key = None
        for k, v in org_train_kwargs.items():
            if isinstance(v, (int, float, str)):
                comet_exp.log_parameter(k, v)
            else:
                comet_exp.log_parameter(k, str(v)[:40])

    def _save_cv_state(completed: bool, last_fold: int, last_model: int) -> None:
        if cv_state_path is None:
            return
        os.makedirs(os.path.dirname(cv_state_path), exist_ok=True)
        torch.save(
            dict(
                completed=bool(completed),
                last_completed_fold=int(last_fold),
                last_completed_model=int(last_model),
                fold_scores=fold_scores,
                fold_model_scores=fold_model_scores,
                states=fold_states,
                exp_key=exp_key,
            ),
            cv_state_path,
        )
    try:
        exp_name = safe_name
        if comet_exp is not None:
            exp_name = comet_exp_name + "_" + exp_name
            comet_exp.set_name(exp_name)

        for fold_idx, (tr_idx, va_idx) in enumerate(cv_iter):
            if max_folds is not None and int(fold_idx) >= int(max_folds):
                break
            if int(fold_idx) < int(start_fold):
                continue
            model_scores: list[float] = []
            model_states: list[dict[str, Any]] = []
            model_states_best: list[dict[str, Any]] = []
            if int(fold_idx) < len(fold_states):
                model_states = list(fold_states[int(fold_idx)])
            if int(fold_idx) < len(fold_model_scores):
                model_scores = list(fold_model_scores[int(fold_idx)])
            for model_idx in range(int(n_models)):
                if int(fold_idx) == int(start_fold) and int(model_idx) < int(start_model):
                    continue
                train_tfms = rect_train_tfms_list[int(model_idx)] if full_rect else train_tfms_list[int(model_idx)]
                if tiled_inp:
                    if use_shared_geom:
                        ds_tr_view = TiledSharedTransformView(
                            dataset,
                            geom_tfms="safe",
                            img_size=img_size_use,
                            post=train_post,
                        )
                    else:
                        ds_tr_view = TiledTransformView(
                            dataset,
                            T.Compose([train_tfms, train_post]),
                        )
                elif full_rect:
                    ds_tr_view = TransformView(
                        dataset,
                        T.Compose([train_tfms, rect_resize, train_post]),
                    )
                else:
                    ds_tr_view = TransformView(dataset, T.Compose([train_tfms, train_post]))
                result = train_one_fold(
                    ds_tr_view=ds_tr_view,
                    ds_va_view=ds_va_view,
                    tr_idx=tr_idx,
                    va_idx=va_idx,
                    fold_idx=int(fold_idx),
                    comet_exp=comet_exp,
                    curr_fold=int(fold_idx),
                    model_idx=int(model_idx),
                    return_state=True,
                    **inp_train_kwargs,
                )
                if isinstance(result, float) and math.isnan(result):
                    return
                best_attempt = None
                best_attempt_score = -1e9
                attempts = 0
                while attempts < int(val_num_retry):
                    if attempts > 0:
                        result = train_one_fold(
                            ds_tr_view=ds_tr_view,
                            ds_va_view=ds_va_view,
                            tr_idx=tr_idx,
                            va_idx=va_idx,
                            fold_idx=int(fold_idx),
                            comet_exp=comet_exp,
                            curr_fold=int(fold_idx),
                            model_idx=int(model_idx),
                            attempt_idx=int(attempts) + 1,
                            attempt_max=int(val_num_retry),
                            return_state=True,
                            **inp_train_kwargs,
                        )
                        if isinstance(result, float) and math.isnan(result):
                            return
                    attempts += 1
                    score = float(result["score"])
                    if math.isnan(score):
                        return
                    if score > best_attempt_score:
                        best_attempt = result
                        best_attempt_score = score
                    if score >= float(val_min_score):
                        break
                if best_attempt is None:
                    return
                result = best_attempt
                result["val_failed"] = bool(float(result["score"]) < float(val_min_score))
                result["val_attempts"] = int(attempts)
                result["val_min_score"] = float(val_min_score)

                model_scores.append(float(result["score"]))
                best_parts = result.get("best_state", result["state"])
                model_states.append(
                    dict(
                        fold_idx=int(fold_idx),
                        model_idx=int(model_idx),
                        tiled_inp=bool(tiled_inp),
                        model_name=str(model_name or ("tiled_base" if tiled_inp else "base")),
                        out_format=str(out_format),
                        pred_space=str(pred_space),
                        head_style=str(head_style),
                        neck_rope=bool(train_kwargs.get("neck_rope", DEFAULTS.get("neck_rope", True))),
                        rope_rescale=train_kwargs.get("rope_rescale", DEFAULTS.get("rope_rescale", None)),
                        neck_drop=float(train_kwargs.get("neck_drop", DEFAULTS.get("neck_drop", 0.0))),
                        neck_pool=bool(train_kwargs.get("neck_pool", DEFAULTS.get("neck_pool", False))),
                        neck_ffn=bool(train_kwargs.get("neck_ffn", DEFAULTS.get("neck_ffn", True))),
                        drop_path=copy.deepcopy(train_kwargs.get("drop_path", DEFAULTS.get("drop_path", None))),
                        backbone_size=str(train_kwargs.get("backbone_size", DEFAULTS.get("backbone_size", "b"))),
                        parts=result["state"],
                        head_hidden=int(train_kwargs["head_hidden"]),
                        head_depth=int(train_kwargs["head_depth"]),
                        head_drop=float(train_kwargs["head_drop"]),
                        num_neck=int(train_kwargs["num_neck"]),
                        neck_num_heads=int(
                            train_kwargs.get(
                                "neck_num_heads",
                                neck_num_heads_for(
                                    str(train_kwargs.get("backbone_size", DEFAULTS.get("backbone_size", "b")))
                                ),
                            )
                        ),
                        img_size=None if img_size is None else int(img_size),
                        score=float(result["score"]),
                        best_score=float(result["best_score"]),
                        val_failed=bool(result.get("val_failed", False)),
                        val_attempts=int(result.get("val_attempts", 1)),
                        val_min_score=float(result.get("val_min_score", val_min_score)),
                    )
                )
                model_states_best.append(
                    dict(
                        fold_idx=int(fold_idx),
                        model_idx=int(model_idx),
                        tiled_inp=bool(tiled_inp),
                        model_name=str(model_name or ("tiled_base" if tiled_inp else "base")),
                        out_format=str(out_format),
                        pred_space=str(pred_space),
                        head_style=str(head_style),
                        neck_rope=bool(train_kwargs.get("neck_rope", DEFAULTS.get("neck_rope", True))),
                        rope_rescale=train_kwargs.get("rope_rescale", DEFAULTS.get("rope_rescale", None)),
                        neck_drop=float(train_kwargs.get("neck_drop", DEFAULTS.get("neck_drop", 0.0))),
                        neck_pool=bool(train_kwargs.get("neck_pool", DEFAULTS.get("neck_pool", False))),
                        neck_ffn=bool(train_kwargs.get("neck_ffn", DEFAULTS.get("neck_ffn", True))),
                        drop_path=copy.deepcopy(train_kwargs.get("drop_path", DEFAULTS.get("drop_path", None))),
                        backbone_size=str(train_kwargs.get("backbone_size", DEFAULTS.get("backbone_size", "b"))),
                        parts=best_parts,
                        head_hidden=int(train_kwargs["head_hidden"]),
                        head_depth=int(train_kwargs["head_depth"]),
                        head_drop=float(train_kwargs["head_drop"]),
                        num_neck=int(train_kwargs["num_neck"]),
                        neck_num_heads=int(
                            train_kwargs.get(
                                "neck_num_heads",
                                neck_num_heads_for(
                                    str(train_kwargs.get("backbone_size", DEFAULTS.get("backbone_size", "b")))
                                ),
                            )
                        ),
                        img_size=None if img_size is None else int(img_size),
                        score=float(result["score"]),
                        best_score=float(result["best_score"]),
                        val_failed=bool(result.get("val_failed", False)),
                        val_attempts=int(result.get("val_attempts", 1)),
                        val_min_score=float(result.get("val_min_score", val_min_score)),
                    )
                )
                if int(fold_idx) < len(fold_states):
                    fold_states[int(fold_idx)] = model_states
                else:
                    fold_states.append(model_states)
                if int(fold_idx) < len(fold_model_scores):
                    fold_model_scores[int(fold_idx)] = model_scores
                else:
                    fold_model_scores.append(model_scores)
                _save_cv_state(False, int(fold_idx), int(model_idx))

            if int(fold_idx) < len(fold_model_scores):
                fold_model_scores[int(fold_idx)] = model_scores
            else:
                fold_model_scores.append(model_scores)
            if int(fold_idx) < len(fold_states):
                fold_states[int(fold_idx)] = model_states
            else:
                fold_states.append(model_states)

            va_subset = Subset(ds_va_view, va_idx)
            num_workers = train_kwargs.get("num_workers", None)
            num_workers = default_num_workers() if num_workers is None else int(num_workers)
            tile_n = 2 if tiled_inp else 1
            if val_bs_override is None:
                val_bs = max(1, int(train_kwargs["batch_size"]) // int(tile_n))
            else:
                val_bs = max(1, int(val_bs_override))
            dl_va = DataLoader(
                va_subset,
                shuffle=False,
                batch_size=int(val_bs),
                pin_memory=str(train_kwargs.get("device", "cuda")).startswith("cuda"),
                num_workers=num_workers,
                persistent_workers=(num_workers > 0),
            )

            if backbone_dtype is None:
                backbone_dtype = train_kwargs.get("backbone_dtype", DEFAULTS["backbone_dtype"])
            if isinstance(backbone_dtype, str):
                backbone_dtype = parse_dtype(backbone_dtype)
            top_k = int(train_kwargs.get("top_k_weights", DEFAULTS.get("top_k_weights", 0)))
            models_best = [
                _build_model_from_state(train_kwargs["backbone"], s, train_kwargs["device"], backbone_dtype)
                for s in model_states_best
            ]
            models_final = [
                _build_model_from_state(train_kwargs["backbone"], s, train_kwargs["device"], backbone_dtype)
                for s in model_states
            ]
            criterion = WeightedMSELoss().to(train_kwargs["device"])
            fold_score_best = eval_global_wr2_ensemble(
                models_best,
                dl_va,
                criterion.w,
                device=train_kwargs["device"],
                trainable_dtype=trainable_dtype,
                comet_exp=comet_exp,
                curr_fold=int(fold_idx),
                tiled_inp=bool(tiled_inp),
                log_key="1ENS_wR2",
            )
            if int(top_k) > 0:
                fold_score_kmean = eval_global_wr2_ensemble(
                    models_final,
                    dl_va,
                    criterion.w,
                    device=train_kwargs["device"],
                    trainable_dtype=trainable_dtype,
                    comet_exp=comet_exp,
                    curr_fold=int(fold_idx),
                    tiled_inp=bool(tiled_inp),
                    log_key="1ENS_kmean_wR2",
                )
                fold_scores.append(float(fold_score_kmean))
            else:
                fold_scores.append(float(fold_score_best))
            _save_cv_state(False, int(fold_idx), int(n_models) - 1)

        total_folds = int(max_folds) if max_folds is not None else int(n_splits)
        if fold_scores:
            last_fold = int(min(len(fold_scores), total_folds) - 1)
            _save_cv_state(len(fold_scores) >= total_folds, last_fold, int(n_models) - 1)
    finally:
        if comet_exp is not None:
            if fold_scores:
                total_folds = int(max_folds) if max_folds is not None else int(n_splits)
                if len(fold_scores) < int(total_folds):
                    comet_exp.end()
                    return
                fold_scores_np = np.asarray(fold_scores, dtype=np.float32)
                comet_exp.log_metric("0cv_mean", fold_scores_np.mean())
                comet_exp.log_metric("0cv_std", fold_scores_np.std(ddof=0))
            comet_exp.end()

    scores = np.asarray(fold_scores, dtype=np.float32)
    if save_output_path is not None:
        os.makedirs(os.path.dirname(save_output_path), exist_ok=True)
        torch.save(
            {
                "fold_scores": scores,
                "fold_model_scores": fold_model_scores,
                "mean": float(scores.mean()),
                "std": float(scores.std(ddof=0)),
                "states": fold_states,
            },
            save_output_path,
        )
        if cv_state_path is not None and os.path.exists(cv_state_path):
            try:
                os.remove(cv_state_path)
            except OSError:
                pass
        
    if return_details:
        return {
            "fold_scores": scores,
            "fold_model_scores": fold_model_scores,
            "mean": float(scores.mean()),
            "std": float(scores.std(ddof=0)),
            "states": fold_states,
        }
