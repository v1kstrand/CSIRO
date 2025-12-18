from __future__ import annotations

import copy
import math
import os
from typing import Any, Callable

import numpy as np
import torch
import torchvision.transforms as T
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .amp import autocast_context, grad_scaler
from .config import DEFAULT_SEED, default_num_workers
from .data import TransformView
from .losses import WeightedMSELoss
from .metrics import eval_global_wr2
from .model import DINOv3Regressor
from .transforms import get_post_tfms_imagenet, get_train_tfms


def cos_sin_lr(ep: int, epochs: int, lr_start: float, lr_final: float) -> float:
    if epochs <= 1:
        return lr_final
    t = (ep - 1) / (epochs - 1)
    return lr_final + 0.5 * (lr_start - lr_final) * (1.0 + math.cos(math.pi * t))


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
    plot_imgs: bool = False,
    early_stopping: int = 6,
    head_hidden: int = 1024,
    head_depth: int = 2,
    head_drop: float = 0.1,
    num_neck: int = 0,
    num_workers: int | None = None,
    comet_exp: Any | None = None,
    skip_log_first_n: int = 5,
    curr_fold: int = 0,
    swa_epochs: int = 15,
    swa_lr: float | None = None,
    swa_anneal_epochs: int = 10,
    swa_load_best: bool = True,
    swa_eval_freq: int = 2,
) -> float:
    tr_subset = Subset(ds_tr_view, tr_idx)
    va_subset = Subset(ds_va_view, va_idx)

    num_workers = default_num_workers() if num_workers is None else int(num_workers)
    dl_kwargs = dict(
        batch_size=int(batch_size),
        pin_memory=str(device).startswith("cuda"),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    dl_tr = DataLoader(tr_subset, shuffle=True, **dl_kwargs)
    dl_va = DataLoader(va_subset, shuffle=False, **dl_kwargs)

    if plot_imgs:
        from .viz import show_nxn_grid

        show_nxn_grid(dataloader=dl_tr, n=4)
        return float("nan")

    model = DINOv3Regressor(
        backbone,
        hidden=int(head_hidden),
        drop=float(head_drop),
        depth=int(head_depth),
        num_neck=int(num_neck),
    ).to(device)
    model.init()

    criterion = WeightedMSELoss().to(device)
    trainable_params = _trainable_params_list(model)
    opt = torch.optim.AdamW(trainable_params, lr=float(lr_start), weight_decay=float(wd))
    scaler = grad_scaler(device)

    if swa_lr is None:
        swa_lr = float(lr_final)

    best_score = -1e9
    best_state = None
    best_opt_state = None
    patience = 0

    p_bar = tqdm(range(1, int(epochs) + 1))
    for ep in p_bar:
        lr = cos_sin_lr(int(ep), int(epochs), float(lr_start), float(lr_final))
        set_optimizer_lr(opt, lr)

        model.set_train(True)
        running = 0.0
        n_seen = 0

        for x, y_log in dl_tr:
            x = x.to(device, non_blocking=True)
            y_log = y_log.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast_context(device):
                p_log = model(x)
                loss = criterion(p_log, y_log)

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
            running += float(loss.detach().item()) * bs
            n_seen += bs

        train_loss = running / max(int(n_seen), 1)
        model.set_train(False)
        score = float(eval_global_wr2(model, dl_va, criterion.w, device=device))

        if comet_exp is not None and int(ep) > int(skip_log_first_n):
            comet_exp.log_metrics(
                {f"train_loss_{curr_fold}": float(train_loss), f"val_wR2_{curr_fold}": float(score)},
                step=int(ep),
            )

        if score > best_score:
            best_score = score
            patience = 0
            best_state = _save_parts(model)
            best_opt_state = copy.deepcopy(opt.state_dict())
        else:
            patience += 1

        s1 = f"Best score: {best_score:.4f} | Patience: {patience:02d}/{int(early_stopping):02d} | lr: {lr:6.4f}"
        s2 = f"[fold {fold_idx}] | train_loss={train_loss:.4f} | val_wR2={score:.4f} | {s1}"
        if verbose:
            print(s2)
        p_bar.set_postfix_str(s2)

        if patience >= int(early_stopping):
            p_bar.set_postfix_str(s2 + " | Early stopping -> SWA phase")
            break

    p_bar.close()

    if (int(swa_epochs) <= 0) or (best_state is None):
        if save_path and best_state is not None:
            torch.save(best_state, save_path)
        return float(best_score)

    if swa_load_best:
        _load_parts(model, best_state)
        if best_opt_state is not None:
            opt.load_state_dict(best_opt_state)

    swa_model = AveragedModel(model).to(device)
    swa_sched = SWALR(
        opt,
        swa_lr=float(swa_lr),
        anneal_epochs=int(swa_anneal_epochs),
        anneal_strategy="cos",
    )

    p_bar = tqdm(range(1, int(swa_epochs) + 1))
    swa_score = None
    for k in p_bar:
        model.set_train(True)
        running = 0.0
        swa_n_seen = 0

        for x, y_log in dl_tr:
            x = x.to(device, non_blocking=True)
            y_log = y_log.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast_context(device):
                p_log = model(x)
                loss = criterion(p_log, y_log)

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
            running += float(loss.detach().item()) * bs
            swa_n_seen += bs

        swa_loss = running / max(int(swa_n_seen), 1)
        swa_sched.step()
        swa_model.update_parameters(model)

        if comet_exp is not None:
            comet_exp.log_metrics({f"swa_train_loss_{curr_fold}": float(swa_loss)}, step=int(k))

        s2 = f"[fold {fold_idx}] | swa_loss={swa_loss:.4f}"
        if verbose:
            print(s2)
        p_bar.set_postfix_str(s2)

        if int(swa_eval_freq) > 0 and (int(k) % int(swa_eval_freq) == 0):
            swa_score = float(eval_global_wr2(swa_model, dl_va, criterion.w, device=device))
            if comet_exp is not None:
                comet_exp.log_metrics({f"swa_wR2_{curr_fold}": float(swa_score)}, step=int(k))

    p_bar.close()

    if swa_score is None or int(swa_eval_freq) <= 0 or (int(k) % int(swa_eval_freq) != 0):
        swa_score = float(eval_global_wr2(swa_model, dl_va, criterion.w, device=device))

    if save_path:
        torch.save(_save_parts(swa_model.module), save_path)

    return float(swa_score)


def run_groupkfold_cv(
    *,
    dataset,
    wide_df,
    n_splits: int = 5,
    seed: int = DEFAULT_SEED,
    group_col: str = "Sampling_Date",
    stratify_col: str = "State",
    train_tfms_fn: Callable[[], T.Compose] | None = None,
    post_tfms_fn: Callable[[], T.Compose] | None = None,
    tfms_fn: Callable[[], T.Compose] | None = None,
    comet_exp_name: str | None = None,
    sweep_config: str = "",
    **train_kwargs,
):
    sgkf = StratifiedGroupKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    X = wide_df
    y = wide_df[stratify_col].values
    groups = wide_df[group_col].values

    if train_tfms_fn is None:
        train_tfms_fn = tfms_fn or get_train_tfms
    if post_tfms_fn is None:
        post_tfms_fn = get_post_tfms_imagenet

    ds_tr_view = TransformView(dataset, T.Compose([train_tfms_fn(), post_tfms_fn()]))
    ds_va_view = TransformView(dataset, post_tfms_fn())

    comet_exp = None
    if comet_exp_name is not None:
        try:
            import comet_ml  # type: ignore

            comet_exp = comet_ml.start(
                api_key=os.getenv("COMET_API_KEY"),
                project_name=comet_exp_name,
                experiment_key=None,
            )
            for k, v in train_kwargs.items():
                if isinstance(v, (int, float, str)):
                    comet_exp.log_parameter(k, v)
            
        except Exception as e:
            raise ImportError(f"{e}") from e

    fold_scores: list[float] = []
    try:
        if comet_exp is not None:
            import uuid

            comet_exp.set_name(comet_exp_name + "_" + sweep_config + "_" + str(uuid.uuid4())[:3])

        for fold_idx, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups)):
            score = train_one_fold(
                ds_tr_view=ds_tr_view,
                ds_va_view=ds_va_view,
                tr_idx=tr_idx,
                va_idx=va_idx,
                fold_idx=int(fold_idx),
                comet_exp=comet_exp,
                curr_fold=int(fold_idx),
                **train_kwargs,
            )
            fold_scores.append(float(score))
    finally:
        if comet_exp is not None:
            comet_exp.end()

    scores = np.asarray(fold_scores, dtype=np.float32)
    return scores, float(scores.mean()), float(scores.std(ddof=0))

