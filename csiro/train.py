from __future__ import annotations

import copy
import math
import os
from typing import Any, Callable
import uuid

import numpy as np
import torch
import torchvision.transforms as T
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .amp import autocast_context, grad_scaler
from .config import DEFAULT_SEED, default_num_workers, DEFAULT_LOSS_WEIGHTS, TARGETS
from .data import TransformView
from .losses import WeightedMSELoss, WeightedSmoothL1Loss, std_balanced_weights
from .metrics import eval_global_wr2
from .model import DINOv3Regressor
from .transforms import post_tfms, train_tfms


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


def _ensure_tensor_batch(x, tfms) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        xs = [xi if torch.is_tensor(xi) else tfms(xi) for xi in x]
        return torch.stack(xs, dim=0)
    return tfms(x).unsqueeze(0)


def _build_model_from_state(backbone, state: dict[str, Any], device: str | torch.device):
    model = DINOv3Regressor(
        backbone,
        hidden=int(state["head_hidden"]),
        drop=float(state["head_drop"]),
        depth=int(state["head_depth"]),
        num_neck=int(state["num_neck"]),
    ).to(device)
    _load_parts(model, state["parts"])
    return model

def _set_swa_lr(swa_lr_start, swa_lr_final, lr_final):
    if swa_lr_start is None and swa_lr_final is None:
        swa_lr_start = float(lr_final)
        swa_lr_final = float(lr_final)
    elif swa_lr_start is None:
        swa_lr_start = float(swa_lr_final)
    elif swa_lr_final is None:
        swa_lr_final = float(swa_lr_start)

    return swa_lr_start, swa_lr_final


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
    swa_lr_start: float | None = None,
    swa_lr_final: float | None = None,
    swa_anneal_epochs: int = 10,
    swa_load_best: bool = True,
    swa_eval_freq: int = 2,
    model_idx: int = 0,
    return_state: bool = False,
    wide_df = None,
    w_std_alpha: float = -1.,
    smooth_l1_beta: float = -1.
) -> float | dict[str, Any]:
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
    
    w_loss = torch.as_tensor(DEFAULT_LOSS_WEIGHTS, dtype=torch.float32)
    eval_w = torch.as_tensor(DEFAULT_LOSS_WEIGHTS, dtype=torch.float32, device=device)
    if w_std_alpha >= 0:
        if wide_df is None:
            raise ValueError("wide_df is required when w_std_alpha >= 0.")
        y_tr = wide_df.iloc[tr_idx][TARGETS].to_numpy(dtype=np.float32)
        y_tr_t = torch.from_numpy(y_tr)                                       # float32 CPU
        y_tr_log = torch.log1p(y_tr_t)                                        # [N_tr, 5]

        std_t = y_tr_log.std(dim=0, unbiased=False)
        w_loss = std_balanced_weights(w_loss, std_t, alpha=float(w_std_alpha))
    
    if smooth_l1_beta < 0:
        criterion = WeightedMSELoss(weights=w_loss).to(device)
    else:
        criterion = WeightedSmoothL1Loss(weights=w_loss, beta=float(smooth_l1_beta)).to(device)
    
    trainable_params = _trainable_params_list(model)
    
    opt = torch.optim.AdamW(trainable_params, lr=float(lr_start), weight_decay=float(wd))
    scaler = grad_scaler(device)

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
        score = float(eval_global_wr2(model, dl_va, eval_w, device=device))

        if comet_exp is not None and int(ep) > int(skip_log_first_n):
            p = {f"x_train_loss_cv{curr_fold}_m{model_idx}": float(train_loss), f"x_val_wR2_cv{curr_fold}_m{model_idx}": float(score)}
            comet_exp.log_metrics(p, step=int(ep))

        if score > best_score:
            best_score = score
            patience = 0
            best_state = _save_parts(model)
            best_opt_state = copy.deepcopy(opt.state_dict())
        else:
            patience += 1

        s1 = f"Best score: {best_score:.4f} | Patience: {patience:02d}/{int(early_stopping):02d} | lr: {lr:6.4f}"
        s2 = (
            f"[fold {fold_idx} | model {int(model_idx)}] | train_loss={train_loss:.4f} | "
            f"val_wR2={score:.4f} | {s1}"
        )
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
        if return_state:
            return {
                "score": float(best_score),
                "best_score": float(best_score),
                "swa_score": None,
                "state": best_state,
                "best_state": best_state,
                "best_opt_state": best_opt_state,
                "opt_state": copy.deepcopy(opt.state_dict()),
                "used_swa": False,
            }
        return float(best_score)

    if swa_load_best:
        _load_parts(model, best_state)
        if best_opt_state is not None:
            opt.load_state_dict(best_opt_state)

    swa_model = AveragedModel(model).to(device)

    swa_lr_start, swa_lr_final = _set_swa_lr(swa_lr_start, swa_lr_final, lr_final)
    total_swa_epochs = int(swa_epochs)
    anneal_epochs = min(max(int(swa_anneal_epochs), 0), total_swa_epochs)
    p_bar = tqdm(range(1, total_swa_epochs + 1))
    swa_score = None
    for k in p_bar:
        if anneal_epochs <= 0:
            swa_lr = float(swa_lr_final)
        elif int(k) <= anneal_epochs:
            swa_lr = cos_sin_lr(int(k), int(anneal_epochs), float(swa_lr_start), float(swa_lr_final))
        else:
            swa_lr = float(swa_lr_final)
        set_optimizer_lr(opt, float(swa_lr))
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
        swa_model.update_parameters(model)

        if comet_exp is not None:
            comet_exp.log_metrics({f"x_swa_train_loss_cv{curr_fold}_m{model_idx}": float(swa_loss)}, step=int(k))
            
        s2 = f"[fold {fold_idx} | model {int(model_idx)}] | swa_loss={swa_loss:.4f}"
        if verbose:
            print(s2)
        if int(swa_eval_freq) > 0 and (int(k) % int(swa_eval_freq) == 0):
            swa_score = float(eval_global_wr2(swa_model, dl_va, eval_w, device=device))
            if comet_exp is not None:
                comet_exp.log_metrics({f"swa_wR2_cv{curr_fold}_m{model_idx}": float(swa_score)}, step=int(k))
                p_bar.set_postfix_str(s2 + f" | swa_wR2={swa_score:.4f}")
        else:
            p_bar.set_postfix_str(s2)

    p_bar.close()

    if swa_score is None or int(swa_eval_freq) <= 0 or (int(k) % int(swa_eval_freq) != 0):
        swa_score = float(eval_global_wr2(swa_model, dl_va, eval_w, device=device))

    swa_state = _save_parts(swa_model.module)
    if save_path:
        torch.save(swa_state, save_path)

    if return_state:
        return {
            "score": float(swa_score),
            "best_score": float(best_score),
            "swa_score": float(swa_score),
            "state": swa_state,
            "best_state": best_state,
            "best_opt_state": best_opt_state,
            "opt_state": copy.deepcopy(opt.state_dict()),
            "used_swa": True,
        }
    return float(swa_score)


@torch.no_grad()
def eval_global_wr2_ensemble(
    models: list[torch.nn.Module],
    dl_va,
    w_vec: torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    tta_rot90: bool = False,
    tta_agg: str = "mean",
    ens_agg: str = "mean",
    comet_exp: Any | None = None,
    curr_fold: int | None = None,
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

    n_rots = 4 if tta_rot90 else 1
    with torch.inference_mode(), autocast_context(device):
        for batch in dl_va:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x, y_log = batch[0], batch[1]
            else:
                raise ValueError("Validation loader must yield (x, y_log).")

            x = x.to(device, non_blocking=True)
            y_log = y_log.to(device, non_blocking=True)

            preds_models: list[torch.Tensor] = []
            for model in models:
                preds_rots: list[torch.Tensor] = []
                for k in range(n_rots):
                    x_rot = x if k == 0 else torch.rot90(x, k, dims=(-2, -1))
                    p_log = model(x_rot).float()
                    p = torch.expm1(p_log).clamp_min(0.0)
                    preds_rots.append(p)
                preds_models.append(_agg_stack(preds_rots, tta_agg))

            p_ens = _agg_stack(preds_models, ens_agg)

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
            comet_exp.log_metrics({str(f"1ENS_wR2_cv{curr_fold}"): float(score)})
        except Exception:
            pass
    return float(score)


@torch.no_grad()
def predict_ensemble(
    data,
    states: list[dict[str, Any]],
    backbone,
    *,
    batch_size: int = 128,
    num_workers: int | None = None,
    device: str | torch.device = "cuda",
    tta_rot90: bool = True,
    tta_agg: str = "mean",
    ens_agg: str = "mean",
) -> torch.Tensor:
    if states and isinstance(states[0], list):
        states = [s for fold in states for s in fold]

    if isinstance(data, DataLoader):
        dl = data
    else:
        num_workers = default_num_workers() if num_workers is None else int(num_workers)
        dl = DataLoader(
            data,
            shuffle=False,
            batch_size=int(batch_size),
            pin_memory=str(device).startswith("cuda"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

    models = [_build_model_from_state(backbone, s, device) for s in states]
    for model in models:
        if hasattr(model, "set_train"):
            model.set_train(False)
        model.eval()

    tfms = post_tfms()
    n_rots = 4 if tta_rot90 else 1
    preds: list[torch.Tensor] = []
    with torch.inference_mode(), autocast_context(device):
        for batch in dl:
            if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                x = batch[0]
            else:
                x = batch

            x = _ensure_tensor_batch(x, tfms).to(device, non_blocking=True)

            preds_models: list[torch.Tensor] = []
            for model in models:
                preds_rots: list[torch.Tensor] = []
                for k in range(n_rots):
                    x_rot = x if k == 0 else torch.rot90(x, k, dims=(-2, -1))
                    p_log = model(x_rot).float()
                    p = torch.expm1(p_log).clamp_min(0.0)
                    preds_rots.append(p)
                preds_models.append(_agg_stack(preds_rots, tta_agg))

            p_ens = _agg_stack(preds_models, ens_agg)
            preds.append(p_ens.detach().cpu())

    return torch.cat(preds, dim=0)


def run_groupkfold_cv(
    *,
    dataset,
    wide_df,
    n_splits: int = 5,
    seed: int = DEFAULT_SEED,
    group_col: str = "Sampling_Date",
    stratify_col: str = "State",
    tfms_fn: Callable[[], T.Compose] | None = None,
    comet_exp_name: str | None = None,
    config_name: str = "",
    n_models: int = 1,
    img_size: int | None = None,
    return_details: bool = False,
    save_output_dir: str | None = None,
    **train_kwargs,
):
    sgkf = StratifiedGroupKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    X = wide_df
    y = wide_df[stratify_col].values
    groups = wide_df[group_col].values

    if tfms_fn is None:
        tfms_fn = train_tfms

    ds_tr_view = TransformView(dataset, T.Compose([tfms_fn(), post_tfms()]))
    ds_va_view = TransformView(dataset, post_tfms())

    comet_exp = None
    if comet_exp_name is not None:
        import comet_ml  # type: ignore
        
        if "uid" in comet_exp_name:
            uid = "_" + str(uuid.uuid4())[:3]
            comet_exp_name = comet_exp_name.replace("uid", "") + uid

        comet_exp = comet_ml.start(
            api_key=os.getenv("COMET_API_KEY"),
            project_name=comet_exp_name,
            experiment_key=None,
        )
        for k, v in train_kwargs.items():
            if isinstance(v, (int, float, str)):
                comet_exp.log_parameter(k, v)

    fold_scores: list[float] = []
    fold_model_scores: list[list[float]] = []
    fold_states: list[list[dict[str, Any]]] = []
    try:
        uid = "_" + str(uuid.uuid4())[:3]
        exp_name = config_name + uid
        if comet_exp is not None:
            exp_name = comet_exp_name + "_" + exp_name
            comet_exp.set_name(exp_name)

        for fold_idx, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups)):
            model_scores: list[float] = []
            model_states: list[dict[str, Any]] = []
            for model_idx in range(int(n_models)):
                result = train_one_fold(
                    wide_df=wide_df,
                    ds_tr_view=ds_tr_view,
                    ds_va_view=ds_va_view,
                    tr_idx=tr_idx,
                    va_idx=va_idx,
                    fold_idx=int(fold_idx),
                    comet_exp=comet_exp,
                    curr_fold=int(fold_idx),
                    model_idx=int(model_idx),
                    return_state=True,
                    **train_kwargs,
                )
                if result == float("nan"):
                    return
                model_scores.append(float(result["score"]))
                model_states.append(
                    dict(
                        fold_idx=int(fold_idx),
                        model_idx=int(model_idx),
                        parts=result["state"],
                        head_hidden=int(train_kwargs["head_hidden"]),
                        head_depth=int(train_kwargs["head_depth"]),
                        head_drop=float(train_kwargs["head_drop"]),
                        num_neck=int(train_kwargs["num_neck"]),
                        img_size=None if img_size is None else int(img_size),
                        score=float(result["score"]),
                        best_score=float(result["best_score"]),
                        swa_score=result["swa_score"],
                        best_state=result["best_state"],
                        used_swa=bool(result["used_swa"]),
                    )
                )

            fold_model_scores.append(model_scores)
            fold_states.append(model_states)

            va_subset = Subset(ds_va_view, va_idx)
            num_workers = train_kwargs.get("num_workers", None)
            num_workers = default_num_workers() if num_workers is None else int(num_workers)
            dl_va = DataLoader(
                va_subset,
                shuffle=False,
                batch_size=int(train_kwargs["batch_size"]),
                pin_memory=str(train_kwargs.get("device", "cuda")).startswith("cuda"),
                num_workers=num_workers,
                persistent_workers=(num_workers > 0),
            )

            models = [_build_model_from_state(train_kwargs["backbone"], s, train_kwargs["device"]) for s in model_states]
            criterion = WeightedMSELoss().to(train_kwargs["device"])
            fold_score = eval_global_wr2_ensemble(
                models,
                dl_va,
                criterion.w,
                device=train_kwargs["device"],
                comet_exp=comet_exp,
                curr_fold=int(fold_idx),
                
            )
            fold_scores.append(float(fold_score))
            
            
    finally:
        if comet_exp is not None:
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
        
    if return_details:
        return {
            "fold_scores": scores,
            "fold_model_scores": fold_model_scores,
            "mean": float(scores.mean()),
            "std": float(scores.std(ddof=0)),
            "states": fold_states,
        }
