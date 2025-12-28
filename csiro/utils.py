from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import time

from .amp import autocast_context
from .config import (
    DEFAULTS,
    DEFAULT_LOSS_WEIGHTS,
    DEFAULT_DATA_ROOT,
    DEFAULT_DINO_REPO_DIR,
    DEFAULT_MODEL_SIZE,
    DEFAULT_PLUS,
    DINO_WEIGHTS_PATH,
    default_num_workers,
    dino_hub_name,
    parse_dtype,
)
from .data import BiomassBaseCached, TransformView, load_train_wide
from .losses import WeightedMSELoss
from .model import DINOv3Regressor
from .transforms import base_train_comp, post_tfms, train_tfms
from .amp import grad_scaler


def _ensure_tensor_batch(x, tfms) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        xs = [xi if torch.is_tensor(xi) else tfms(xi) for xi in x]
        return torch.stack(xs, dim=0)
    return tfms(x).unsqueeze(0)


def _build_model_from_state(
    backbone,
    state: dict[str, Any],
    device: str | torch.device,
    backbone_dtype: torch.dtype | None = None,
) -> DINOv3Regressor:
    model = DINOv3Regressor(
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
    return model


def _normalize_states(states: Any) -> list[list[dict[str, Any]]]:
    if isinstance(states, (str, bytes)):
        states = torch.load(states, map_location="cpu", weights_only=False)
    if isinstance(states, dict) and "states" in states:
        states = states["states"]
    if not isinstance(states, list) or not states:
        raise ValueError("states must be a non-empty list or a checkpoint dict with 'states'.")
    if isinstance(states[0], dict):
        return [states]
    if isinstance(states[0], list):
        return states
    return [list(states)]


def _pairwise_stats(preds: torch.Tensor) -> dict[str, float]:
    m = int(preds.shape[0])
    if m < 2:
        return dict(mean_pairwise_corr=1.0, mean_pairwise_mae=0.0, mean_model_std=0.0)
    flat = preds.reshape(m, -1).float()
    flat = flat - flat.mean(dim=1, keepdim=True)
    denom = flat.norm(dim=1, keepdim=True)
    corr = (flat @ flat.t()) / (denom @ denom.t() + 1e-12)
    off = corr[~torch.eye(m, dtype=torch.bool)]
    mean_corr = float(off.mean().item()) if off.numel() else 1.0

    mae_sum = 0.0
    count = 0
    for i in range(m):
        for j in range(i + 1, m):
            mae_sum += float((preds[i] - preds[j]).abs().mean().item())
            count += 1
    mean_mae = float(mae_sum / max(count, 1))

    mean_std = float(preds.std(dim=0, unbiased=False).mean().item())
    return dict(mean_pairwise_corr=mean_corr, mean_pairwise_mae=mean_mae, mean_model_std=mean_std)


def load_train_dataset_simple(
    *,
    csv: str | None = None,
    root: str | None = DEFAULT_DATA_ROOT,
    img_size: int | None = None,
    cache_images: bool = True,
):
    if csv is None:
        if root is None:
            raise ValueError("Set root or csv to load the training dataset.")
        csv = os.path.join(root, "train.csv")
    if img_size is None:
        img_size = int(DEFAULTS["img_size"])

    wide_df = load_train_wide(str(csv), root=str(root) if root is not None else None)
    dataset = BiomassBaseCached(wide_df, img_size=int(img_size), cache_images=bool(cache_images))
    return wide_df, dataset


def preview_augments(
    tfms_list: list[T.Compose],
    *,
    dataset=None,
    k: int = 4,
    seed: int = 0,
    show_titles: bool = False,
):
    if not tfms_list:
        raise ValueError("tfms_list must contain at least one transform.")
    if dataset is None:
        _, dataset = load_train_dataset_simple()

    total = len(dataset)
    g = torch.Generator().manual_seed(int(seed))
    idxs = torch.randperm(int(total), generator=g)[: int(k)].tolist()

    aug_tfms = [T.Compose([base_train_comp, t]) for t in tfms_list]
    n_cols = len(aug_tfms)
    fig, axes = plt.subplots(int(k), int(n_cols), figsize=(3.2 * n_cols, 3.2 * int(k)))
    if int(k) == 1:
        axes = [axes]
    for r, idx in enumerate(idxs):
        sample = dataset[int(idx)]
        img = sample[0] if isinstance(sample, (tuple, list)) else sample
        if not isinstance(img, Image.Image):
            raise TypeError("Dataset must return PIL images for preview_augments.")
        for c, tfm in enumerate(aug_tfms):
            aug = tfm(img)
            ax = axes[r][c] if int(k) > 1 else axes[c]
            ax.imshow(aug)
            ax.axis("off")
            if show_titles:
                ax.set_title(f"t{c+1}")
    plt.tight_layout()
    return fig


def build_color_jitter_sweep(
    n: int,
    *,
    bcs_range: tuple[float, float],
    hue_range: tuple[float, float],
) -> list[T.Compose]:
    n = int(n)
    if n <= 0:
        raise ValueError("n must be >= 1.")
    b0, b1 = float(bcs_range[0]), float(bcs_range[1])
    h0, h1 = float(hue_range[0]), float(hue_range[1])
    if n == 1:
        bcs_vals = [b0]
        hue_vals = [h0]
    else:
        bcs_vals = torch.linspace(b0, b1, n).tolist()
        hue_vals = torch.linspace(h0, h1, n).tolist()

    tfms_list: list[T.Compose] = []
    for bcs, hue in zip(bcs_vals, hue_vals):
        tfms_list.append(
            T.Compose(
                [
                    T.ColorJitter(
                        brightness=float(bcs),
                        contrast=float(bcs),
                        saturation=float(bcs),
                        hue=float(hue),
                    )
                ]
            )
        )
    return tfms_list


def _trainable_params_list(m: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for name in ("neck", "head", "norm"):
        part = getattr(m, name, None)
        if part is None:
            continue
        for p in part.parameters():
            if p.requires_grad:
                params.append(p)
    return params


def health_check_throughput(
    *,
    dataset=None,
    backbone=None,
    warmup: int = 2,
    epochs: int = 1,
    batch_size: int = 32,
    num_workers: int | None = None,
    device: str | torch.device = "cuda",
    backbone_dtype: str | torch.dtype | None = None,
    head_hidden: int | None = None,
    head_depth: int | None = None,
    head_drop: float | None = None,
    num_neck: int | None = None,
    lr: float | None = None,
    wd: float | None = None,
    no_transform: bool = False,
) -> dict[str, float]:
    if dataset is None:
        _, dataset = load_train_dataset_simple()

    sample = dataset[0]
    if not no_transform:
        tfms = T.Compose([train_tfms(), post_tfms()])
        if isinstance(sample, (tuple, list)):
            img0 = sample[0] if sample else None
            if isinstance(img0, Image.Image):
                dataset = TransformView(dataset, tfms)
        elif isinstance(sample, Image.Image):
            dataset = TransformView(dataset, tfms)
    else:
        if isinstance(sample, (tuple, list)):
            img0 = sample[0] if sample else None
            if isinstance(img0, Image.Image):
                raise ValueError("no_transform=True requires a tensor dataset (got PIL images).")
        elif isinstance(sample, Image.Image):
            raise ValueError("no_transform=True requires a tensor dataset (got PIL images).")

    num_workers = default_num_workers() if num_workers is None else int(num_workers)
    dl = DataLoader(
        dataset,
        shuffle=True,
        batch_size=int(batch_size),
        pin_memory=str(device).startswith("cuda"),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    if backbone_dtype is None:
        backbone_dtype = DEFAULTS["backbone_dtype"]
    if isinstance(backbone_dtype, str):
        backbone_dtype = parse_dtype(backbone_dtype)

    if backbone is None:
        dino_repo = DEFAULT_DINO_REPO_DIR
        model_size = DEFAULT_MODEL_SIZE
        plus = DEFAULT_PLUS
        dino_weights = DINO_WEIGHTS_PATH
        if dino_repo is None:
            raise ValueError("Set DEFAULT_DINO_REPO_DIR in config/env or pass backbone.")
        if dino_weights is None:
            raise ValueError("Set DINO_WEIGHTS_PATH in config/env or pass backbone.")
        backbone = torch.hub.load(
            str(dino_repo),
            dino_hub_name(model_size=str(model_size), plus=str(plus)),
            source="local",
            weights=str(dino_weights),
        )

    model = DINOv3Regressor(
        backbone,
        hidden=int(DEFAULTS["head_hidden"] if head_hidden is None else head_hidden),
        drop=float(DEFAULTS["head_drop"] if head_drop is None else head_drop),
        depth=int(DEFAULTS["head_depth"] if head_depth is None else head_depth),
        num_neck=int(DEFAULTS["num_neck"] if num_neck is None else num_neck),
        backbone_dtype=backbone_dtype,
    ).to(device)
    model.init()
    if hasattr(model, "set_train"):
        model.set_train(True)
    model.train()

    trainable_params = _trainable_params_list(model)
    lr = float(DEFAULTS["lr_start"] if lr is None else lr)
    wd = float(DEFAULTS["wd"] if wd is None else wd)
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
    scaler = grad_scaler(device)
    criterion = WeightedMSELoss(weights=torch.as_tensor(DEFAULT_LOSS_WEIGHTS, dtype=torch.float32)).to(device)

    it = iter(dl)
    for _ in range(int(warmup)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("Dataset must yield (x, y_log) for health_check_throughput.")
        x, y_log = batch[0], batch[1]
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

    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)

    times: list[float] = []
    n_seen = 0
    for _ in range(int(max(1, epochs))):
        it = iter(dl)
        for batch in it:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("Dataset must yield (x, y_log) for health_check_throughput.")
            x, y_log = batch[0], batch[1]
            t0 = time.perf_counter()
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
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            n_seen += int(x.size(0))

    if not times:
        return dict(samples_per_sec=0.0, mean_step_time=0.0, median_step_time=0.0, total_time=0.0)

    t = torch.tensor(times, dtype=torch.float64)
    mean_step = float(t.mean().item())
    med_step = float(t.median().item())
    total = float(t.sum().item())
    sps = float(n_seen / max(total, 1e-12))
    return dict(
        samples_per_sec=sps,
        mean_step_time=mean_step,
        median_step_time=med_step,
        total_time=total,
        epochs=int(max(1, epochs)),
        batch_size=int(batch_size),
    )


def analyze_dataloader_perf(
    *,
    dataset=None,
    batch_size: int = 32,
    num_workers: int | None = None,
    device: str | torch.device | None = None,
    epochs: int = 1,
    max_batches_per_epoch: int | None = None,
    warmup: int = 5,
    seed: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    include_transfer: bool = True,
) -> dict[str, float]:
    if dataset is None:
        _, dataset = load_train_dataset_simple()

    tfms = T.Compose([train_tfms(), post_tfms()])
    sample = dataset[0]
    if isinstance(sample, (tuple, list)):
        img0 = sample[0] if sample else None
        if isinstance(img0, Image.Image):
            dataset = TransformView(dataset, tfms)
    elif isinstance(sample, Image.Image):
        dataset = TransformView(dataset, tfms)

    num_workers = default_num_workers() if num_workers is None else int(num_workers)
    if pin_memory is None:
        pin_memory = str(device).startswith("cuda") if device is not None else False
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    dl = DataLoader(
        dataset,
        shuffle=False,
        batch_size=int(batch_size),
        pin_memory=bool(pin_memory),
        num_workers=num_workers,
        persistent_workers=bool(persistent_workers),
    )

    g = torch.Generator().manual_seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    device = torch.device(device) if device is not None else None
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)

    if int(warmup) > 0:
        it_warm = iter(dl)
        for _ in range(int(warmup)):
            try:
                next(it_warm)
            except StopIteration:
                break

    batch_times: list[float] = []
    epoch_times: list[float] = []
    time_between: list[float] = []
    n_seen = 0
    epochs = int(max(1, epochs))

    for _ in range(int(epochs)):
        it = iter(dl)
        epoch_t0 = time.perf_counter()
        prev_end: float | None = None
        for bi, batch in enumerate(it):
            if max_batches_per_epoch is not None and bi >= int(max_batches_per_epoch):
                break
            t0 = time.perf_counter()
            if prev_end is not None:
                time_between.append(t0 - prev_end)
            if include_transfer and device is not None:
                if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                    x = batch[0]
                else:
                    x = batch
                if torch.is_tensor(x):
                    x = x.to(device, non_blocking=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            batch_times.append(t1 - t0)
            prev_end = t1
            if isinstance(batch, (tuple, list)) and len(batch) >= 1 and torch.is_tensor(batch[0]):
                n_seen += int(batch[0].size(0))
            elif torch.is_tensor(batch):
                n_seen += int(batch.size(0))
            else:
                n_seen += int(batch_size)

        if prev_end is None:
            epoch_times.append(0.0)
        else:
            epoch_times.append(prev_end - epoch_t0)

    if not batch_times:
        return dict(
            mean_batch_time=0.0,
            median_batch_time=0.0,
            samples_per_sec=0.0,
            total_time=0.0,
            avg_epoch_time=0.0,
            avg_time_between=0.0,
        )

    times = torch.tensor(batch_times, dtype=torch.float64)
    mean_bt = float(times.mean().item())
    med_bt = float(times.median().item())
    sps = float(n_seen / max(times.sum().item(), 1e-12))
    total_time = float(sum(epoch_times))
    avg_epoch = float(total_time / max(len(epoch_times), 1))
    avg_between = float(sum(time_between) / max(len(time_between), 1))
    return dict(
        mean_batch_time=mean_bt,
        median_batch_time=med_bt,
        samples_per_sec=sps,
        total_time=total_time,
        avg_epoch_time=avg_epoch,
        avg_time_between=avg_between,
        epochs=int(epochs),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        include_transfer=bool(include_transfer),
    )


@torch.no_grad()
def analyze_ensemble_redundancy(
    ensemble_states: Any,
    *,
    backbone=None,
    dataset=None,
    n_samples: int = 64,
    batch_size: int = 32,
    num_workers: int | None = None,
    device: str | torch.device = "cuda",
    seed: int = 0,
    backbone_dtype: str | torch.dtype | None = None,
    tta_rot90: bool = False,
    tta_agg: str = "mean",
    return_preds: bool = False,
    dino_repo: str | None = None,
    dino_weights: str | None = None,
    model_size: str | None = None,
    plus: str | None = None,
) -> dict[str, Any]:
    fold_states = _normalize_states(ensemble_states)

    if backbone_dtype is None:
        backbone_dtype = DEFAULTS["backbone_dtype"]
    if isinstance(backbone_dtype, str):
        backbone_dtype = parse_dtype(backbone_dtype)

    if backbone is None:
        dino_repo = DEFAULT_DINO_REPO_DIR if dino_repo is None else dino_repo
        model_size = DEFAULT_MODEL_SIZE if model_size is None else model_size
        plus = DEFAULT_PLUS if plus is None else plus
        dino_weights = DINO_WEIGHTS_PATH if dino_weights is None else dino_weights
        if dino_repo is None:
            raise ValueError("Set DEFAULT_DINO_REPO_DIR in config/env or pass dino_repo.")
        if dino_weights is None:
            raise ValueError("Set DINO_WEIGHTS_PATH in config/env or pass dino_weights.")
        backbone = torch.hub.load(
            str(dino_repo),
            dino_hub_name(model_size=str(model_size), plus=str(plus)),
            source="local",
            weights=str(dino_weights),
        )

    if dataset is None:
        _, dataset = load_train_dataset_simple()

    num_workers = default_num_workers() if num_workers is None else int(num_workers)
    tfms = post_tfms()
    n_rots = 4 if tta_rot90 else 1

    if isinstance(dataset, DataLoader):
        dl = dataset
        idxs = None
    else:
        sample = dataset[0]
        if isinstance(sample, (tuple, list)):
            img0 = sample[0] if sample else None
            if isinstance(img0, Image.Image):
                dataset = TransformView(dataset, tfms)
        elif isinstance(sample, Image.Image):
            dataset = TransformView(dataset, tfms)

        total = len(dataset)
        g = torch.Generator().manual_seed(int(seed))
        if int(n_samples) >= int(total):
            idxs = list(range(int(total)))
        else:
            idxs = torch.randperm(int(total), generator=g)[: int(n_samples)].tolist()
        subset = Subset(dataset, idxs)
        dl = DataLoader(
            subset,
            shuffle=False,
            batch_size=int(batch_size),
            pin_memory=str(device).startswith("cuda"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

    fold_stats: list[dict[str, float]] = []
    fold_preds: list[torch.Tensor] = []

    with torch.inference_mode(), autocast_context(device):
        for states in fold_states:
            models = [_build_model_from_state(backbone, s, device, backbone_dtype) for s in states]
            for m in models:
                if hasattr(m, "set_train"):
                    m.set_train(False)
                m.eval()

            preds_by_model: list[list[torch.Tensor]] = [[] for _ in models]
            for batch in dl:
                if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                    x = batch[0]
                else:
                    x = batch
                x = _ensure_tensor_batch(x, tfms).to(device, non_blocking=True)

                for mi, model in enumerate(models):
                    preds_rots: list[torch.Tensor] = []
                    for k in range(n_rots):
                        x_rot = x if k == 0 else torch.rot90(x, k, dims=(-2, -1))
                        p_log = model(x_rot).float()
                        p = torch.expm1(p_log).clamp_min(0.0)
                        preds_rots.append(p)
                    preds_by_model[mi].append(torch.stack(preds_rots, dim=0).mean(dim=0).detach().cpu())

            preds_models = [torch.cat(p, dim=0) for p in preds_by_model]
            preds = torch.stack(preds_models, dim=0)
            fold_stats.append(_pairwise_stats(preds))
            if return_preds:
                fold_preds.append(preds)

    result = dict(fold_stats=fold_stats)
    if return_preds:
        result["fold_preds"] = fold_preds
    if idxs is not None:
        result["sample_indices"] = idxs
    return result
