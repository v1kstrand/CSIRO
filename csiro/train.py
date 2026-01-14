from __future__ import annotations

import copy
import math
import os
from typing import Any, Callable
import uuid

import numpy as np
import torch
import torchvision.transforms as T
from sklearn.model_selection import GroupKFold
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
from .model import TiledDINOv3Regressor, TiledDINOv3Regressor3, TiledDINOv3RegressorStitched3
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
    for name in ("neck", "head", "norm"):
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
    for name in ("neck", "head", "norm"):
        part = getattr(m, name, None)
        if part is not None:
            state[name] = {k: v.detach().cpu() for k, v in part.state_dict().items()}
    return state


def _load_parts(m: torch.nn.Module, state: dict[str, dict[str, torch.Tensor]]) -> None:
    for name in ("neck", "head", "norm"):
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
    if model_cls in (TiledDINOv3Regressor3, TiledDINOv3RegressorStitched3):
        model_kwargs["head_style"] = head_style
        if model_cls is TiledDINOv3RegressorStitched3:
            out_format = str(state.get("out_format", DEFAULTS.get("out_format", "cat_cls"))).strip().lower()
            model_kwargs["out_format"] = out_format
            model_kwargs["neck_rope"] = bool(state.get("neck_rope", DEFAULTS.get("neck_rope", True)))
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
    tiled_inp: bool = False,
    val_freq: int = 1,
    backbone_size: str | None = None,
    mixup: tuple[float, float] | None = None,
    model_name: str | None = None,
    head_style: str | None = None,
    pred_space: str | None = None,
    loss_weights: list[float] | tuple[float, ...] | None = None,
    huber_beta: float | None = None,
    tau_neg: float | None = None,
    out_format: str | None = None,
    neck_rope: bool | None = None,
    top_k_weights: int | None = None,
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
    if not isinstance(mixup, (list, tuple)) or len(mixup) != 2:
        raise ValueError("mixup must be a (p, alpha) tuple.")
    mixup_p = float(mixup[0])
    mixup_alpha = float(mixup[1])

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
    if model_cls is TiledDINOv3RegressorStitched3:
        if out_format is None:
            out_format = DEFAULTS.get("out_format", "cat_cls")
        model_kwargs["out_format"] = str(out_format).strip().lower()
        if neck_rope is None:
            neck_rope = bool(DEFAULTS.get("neck_rope", True))
        model_kwargs["neck_rope"] = bool(neck_rope)
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
    topk_list: list[tuple[float, dict[str, dict[str, torch.Tensor]]]] = []

    val_freq = max(1, int(val_freq))
    p_bar = tqdm(range(1, int(epochs) + 1))

    for ep in p_bar:
        lr = cos_sin_lr(int(ep), int(epochs), float(lr_start), float(lr_final))
        set_optimizer_lr(opt, lr)

        model.set_train(True)
        running = 0.0
        n_seen = 0

        for bi, (x, y_log) in enumerate(dl_tr):
            x = x.to(device, non_blocking=True)
            y_log = y_log.to(device, non_blocking=True)
            y_target = y_log
            if pred_space == "gram":
                y_target = torch.expm1(y_log.float())
            if mixup_p > 0.0 and mixup_alpha > 0.0 and int(x.size(0)) > 1:
                if torch.rand((), device=x.device).item() < mixup_p:
                    perm = torch.randperm(int(x.size(0)), device=x.device)
                    x2 = x[perm]
                    y2 = y_target[perm]
                    lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample((x.size(0),)).to(x.device)
                    lam_x = lam.view([x.size(0)] + [1] * (x.ndim - 1))
                    x = x * lam_x + x2 * (1.0 - lam_x)

                    lam_y = lam.view(x.size(0), 1)
                    if pred_space == "log":
                        y_lin = torch.expm1(y_log.float()).clamp_min(0.0)
                        y2_lin = torch.expm1(y2.float()).clamp_min(0.0)
                        y_mix = y_lin * lam_y + y2_lin * (1.0 - lam_y)
                        y_target = torch.log1p(y_mix)
                    else:
                        y_target = y_target * lam_y + y2 * (1.0 - lam_y)

            opt.zero_grad(set_to_none=True)
            with autocast_context(device, dtype=trainable_dtype):
                pred = model(x)
                loss_main = criterion(pred, y_target)
                loss = loss_main
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
            n_seen += bs

        train_loss = running / max(int(n_seen), 1)
        do_eval = (val_freq == 1) or (int(ep) and int(ep) % int(val_freq) == 0)
        score = None
        if do_eval:
            model.set_train(False)
            score = float(eval_global_wr2(model, dl_va, eval_w, device=device))
            if top_k > 0:
                state_k = _save_parts(model)
                topk_list.append((float(score), state_k))
                topk_list.sort(key=lambda x: float(x[0]), reverse=True)
                if len(topk_list) > int(top_k):
                    topk_list.pop(-1)

        if comet_exp is not None and int(ep) > int(skip_log_first_n):
            p = {f"x_train_loss_cv{curr_fold}_m{model_idx}": float(train_loss)}
            if score is not None:
                p[f"x_val_wR2_cv{curr_fold}_m{model_idx}"] = float(score)
            comet_exp.log_metrics(p, step=int(ep))

        if score is not None and score > best_score:
            best_score = float(score)
            patience = 0
            best_state = _save_parts(model)
            best_opt_state = copy.deepcopy(opt.state_dict())
        else:
            patience += 1

        s1 = f"Best score: {best_score:.4f} | Patience: {patience:02d}/{int(early_stopping):02d} | lr: {lr:6.4f}"
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
            model.set_train(False)
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
    n_models: int = 1,
    img_size: int | None = None,
    return_details: bool = False,
    save_output_dir: str | None = None,
    cv_params: dict[str, Any] | None = None,
    max_folds: int | None = None,
    **train_kwargs,
):
    if cv_params is None:
        cv_params = DEFAULTS.get("cv_params")
    if not isinstance(cv_params, dict):
        raise ValueError("cv_params must be a dict.")
    if "mode" not in cv_params:
        raise ValueError("cv_params must include 'mode'.")

    split_mode = str(cv_params["mode"]).lower()
    n_splits = int(cv_params.get("n_splits", DEFAULTS["cv_params"]["n_splits"]))
    if max_folds is None:
        max_folds = cv_params.get("max_folds", DEFAULTS.get("max_folds", None))
    max_folds = None if max_folds is None else int(max_folds)
    cv_resume = bool(train_kwargs.pop("cv_resume", DEFAULTS.get("cv_resume", False)))
    cv_seed: int | None = None

    if split_mode == "gkf":
        if "cv_seed" not in cv_params:
            raise ValueError("cv_params must include 'cv_seed' for mode='gkf'.")
        cv_seed = int(cv_params["cv_seed"])
        gkf = GroupKFold(n_splits=int(n_splits), shuffle=True, random_state=int(cv_seed))
        groups = wide_df["Sampling_Date"].values
        cv_iter = gkf.split(wide_df, groups=groups)
    else:
        raise ValueError(f"Unknown cv mode: {cv_params['mode']}")

    org_train_kwargs = train_kwargs.copy()
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
    model_name = str(train_kwargs.get("model_name", DEFAULTS.get("model_name", "")))
    out_format = str(train_kwargs.get("out_format", DEFAULTS.get("out_format", "cat_cls"))).strip().lower()
    pred_space = _normalize_pred_space(train_kwargs.get("pred_space", DEFAULTS.get("pred_space", "log")))
    head_style = str(train_kwargs.get("head_style", DEFAULTS.get("head_style", "single"))).strip().lower()
    jitter_tfms = build_color_jitter_sweep(
        int(n_models),
        bcs_range=tuple(bcs_range),
        hue_range=tuple(hue_range),
    )
    train_tfms_list = [T.Compose([base_train_comp, t]) for t in jitter_tfms]
    train_post_ops = [post_tfms()]
    if cutout_p > 0.0:
        train_post_ops.append(T.RandomErasing(p=float(cutout_p)))
    if to_gray_p > 0.0:
        train_post_ops.append(T.RandomGrayscale(p=float(to_gray_p)))
    train_post = T.Compose(train_post_ops)
    use_shared_geom = tiled_inp and tile_geom_mode == "shared"
    img_size_use = int(img_size or DEFAULTS.get("img_size", 512))
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
    else:
        ds_va_view = TransformView(dataset, post_tfms())

    cv_state_path = None
    if save_output_dir is not None:
        cv_state_path = os.path.join(save_output_dir, f"{config_name}_cv_state.pt")

    fold_scores: list[float] = []
    fold_model_scores: list[list[float]] = []
    fold_states: list[list[dict[str, Any]]] = []
    start_fold = 0
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
            if last_completed is None:
                last_completed = len(fold_states) - 1
            start_fold = int(last_completed) + 1
            print(f"INFO: Resuming from fold {start_fold}")
            
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

    def _save_cv_state(completed: bool, last_fold: int) -> None:
        if cv_state_path is None:
            return
        os.makedirs(save_output_dir, exist_ok=True)
        torch.save(
            dict(
                completed=bool(completed),
                last_completed_fold=int(last_fold),
                fold_scores=fold_scores,
                fold_model_scores=fold_model_scores,
                states=fold_states,
                exp_key=exp_key,
            ),
            cv_state_path,
        )
    try:
        uid = "_" + str(uuid.uuid4())[:3]
        exp_name = config_name + uid
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
            for model_idx in range(int(n_models)):
                train_tfms = train_tfms_list[int(model_idx)]
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
                    )
                )

            fold_model_scores.append(model_scores)
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
            _save_cv_state(False, int(fold_idx))

        total_folds = int(max_folds) if max_folds is not None else int(n_splits)
        if fold_scores:
            last_fold = int(min(len(fold_scores), total_folds) - 1)
            _save_cv_state(len(fold_scores) >= total_folds, last_fold)
    finally:
        if comet_exp is not None:
            if fold_scores:
                fold_scores_np = np.asarray(fold_scores, dtype=np.float32)
                comet_exp.log_metric("0cv_mean", fold_scores_np.mean())
                comet_exp.log_metric("0cv_std", fold_scores_np.std(ddof=0))
            comet_exp.end()

    scores = np.asarray(fold_scores, dtype=np.float32)
    if save_output_dir is not None:
        os.makedirs(save_output_dir, exist_ok=True)
        save_output_path = os.path.join(save_output_dir, exp_name + ".pt")
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


