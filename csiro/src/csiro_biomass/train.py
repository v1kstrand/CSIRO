from __future__ import annotations

import math
import os
from typing import Any, Callable

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm.auto import tqdm
import torchvision.transforms as T

from .amp import DTYPE, autocast_context, grad_scaler
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


def _trainable_params(m: torch.nn.Module):
    for b in _trainable_blocks(m):
        yield from b.parameters()


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
    fold_idx: int = 0,
    epochs: int = 5,
    lr_start: float = 3e-4,
    lr_final: float = 5e-5,
    wd: float = 1e-2,
    batch_size: int = 128,
    device: str = "cuda",
    save_path: str | None = None,
    verbose: bool = True,
    plot_imgs: bool = False,
    head_hidden: int = 1024,
    head_drop: float = 0.1,
    head_depth: int = 2,
    num_neck: int = 0,
    early_stopping: int = 6,
    num_workers: int | None = None,
    comet_exp: Any | None = None,
    skip_log_first_n: int = 5,
    curr_fold: int = 0,
    swa_epochs: int = 20,
    swa_lr: float | None = None,
    swa_anneal_epochs: int = 15,
    swa_load_best: bool = False,
    swa_eval_freq: int = 2,
):
    tr_subset = Subset(ds_tr_view, tr_idx)
    va_subset = Subset(ds_va_view, va_idx)

    num_workers = default_num_workers() if num_workers is None else int(num_workers)
    dl_kwargs = dict(
        batch_size=batch_size,
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

    criterion = WeightedMSELoss().to(device)
    model = DINOv3Regressor(
        backbone, hidden=head_hidden, drop=head_drop, depth=head_depth, num_neck=num_neck
    ).to(device)
    model.init()
    opt = torch.optim.AdamW(_trainable_params(model), lr=lr_start, weight_decay=wd)
    scaler = grad_scaler(device)

    best_score = -1e9
    patience = 0
    best_state: dict[str, dict[str, torch.Tensor]] | None = None
    p_bar = tqdm(range(1, epochs + 1))
    for ep in p_bar:
        lr = cos_sin_lr(ep, epochs, lr_start, lr_final)
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
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            bs = x.size(0)
            running += float(loss.detach().item()) * int(bs)
            n_seen += bs

        train_loss = running / max(int(n_seen), 1)
        score = eval_global_wr2(model, dl_va, criterion.w, device=device)

        if comet_exp is not None and ep > int(skip_log_first_n):
            comet_exp.log_metrics(
                {f"train_loss_{curr_fold}": float(train_loss), f"val_wR2_{curr_fold}": float(score)},
                step=int(ep),
            )

        if score > best_score:
            patience = 0
            best_score = score
            best_state = _save_parts(model)
        else:
            patience += 1

        if verbose:
            p_bar.set_postfix_str(
                f"[fold {fold_idx}] ep {ep:02d} train={train_loss:.4f} val_wR2={score:.4f} best={best_score:.4f} pat={patience}/{early_stopping}"
            )

        if patience >= early_stopping:
            break

    p_bar.close()

    if (swa_epochs <= 0) or (best_state is None):
        if save_path and best_state is not None:
            torch.save(best_state, save_path)
        return float(best_score)

    if swa_lr is None:
        swa_lr = float(lr_final)

    if swa_load_best:
        _load_parts(model, best_state)

    swa_model = AveragedModel(model).to(device)
    swa_sched = SWALR(opt, swa_lr=float(swa_lr), anneal_epochs=int(swa_anneal_epochs), anneal_strategy="cos")

    swa_score = float(best_score)
    p_bar = tqdm(range(1, int(swa_epochs) + 1))
    for k in p_bar:
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
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            bs = x.size(0)
            running += float(loss.detach().item()) * int(bs)
            n_seen += bs

        swa_loss = running / max(int(n_seen), 1)
        swa_sched.step()
        swa_model.update_parameters(model)

        if comet_exp is not None:
            comet_exp.log_metrics({f"swa_train_loss_{curr_fold}": float(swa_loss)}, step=int(k))

        p_bar.set_postfix_str(f"[fold {fold_idx}] swa_loss={swa_loss:.4f}")
        if int(swa_eval_freq) > 0 and (k % int(swa_eval_freq)) == 0:
            swa_score = float(eval_global_wr2(swa_model, dl_va, criterion.w, device=device))
            if comet_exp is not None:
                comet_exp.log_metrics({f"swa_wR2_{curr_fold}": float(swa_score)}, step=int(k))

    p_bar.close()
    if int(swa_eval_freq) <= 0 or (k % int(swa_eval_freq)) != 0:
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
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
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
        except Exception as e:
            raise ImportError(
                "comet_exp_name was provided but comet_ml could not be imported; install comet-ml or pass comet_exp_name=None."
            ) from e

    fold_scores = []
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
                fold_idx=fold_idx,
                comet_exp=comet_exp,
                curr_fold=fold_idx,
                **train_kwargs,
            )
            fold_scores.append(score)
    finally:
        if comet_exp is not None:
            comet_exp.end()

    fold_scores = np.asarray(fold_scores, dtype=np.float32)
    return fold_scores, float(fold_scores.mean()), float(fold_scores.std(ddof=0))
