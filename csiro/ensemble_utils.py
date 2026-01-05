from __future__ import annotations

import torch


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
