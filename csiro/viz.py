from __future__ import annotations

import random
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import IMAGENET_MEAN, IMAGENET_STD


def _denorm_img(
    x: torch.Tensor,
    *,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(3, 1, 1)
    x = x * std_t + mean_t
    return x.clamp(0, 1)


@torch.no_grad()
def show_nxn_grid(
    *,
    dataset=None,
    dataloader=None,
    n: int = 4,
    indices: Sequence[int] | None = None,
    seed: int = 0,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    show_targets: bool = True,
    targets_are_log1p: bool = True,
    figsize_per_cell: float = 3.0,
) -> None:
    if (dataset is None) == (dataloader is None):
        raise ValueError("Pass exactly one of dataset=... or dataloader=....")

    k = int(n) * int(n)
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []

    if dataset is not None:
        if indices is None:
            rng = random.Random(int(seed))
            indices = [rng.randrange(len(dataset)) for _ in range(k)]
        if len(indices) < k:
            raise ValueError(f"Need at least {k} indices.")

        for i in indices[:k]:
            x, y = dataset[int(i)]
            xs.append(x)
            if show_targets:
                ys.append(y)
    else:
        for xb, yb in dataloader:
            for j in range(int(xb.shape[0])):
                xs.append(xb[j])
                if show_targets:
                    ys.append(yb[j])
                if len(xs) >= k:
                    break
            if len(xs) >= k:
                break

    x_batch = torch.stack(xs, dim=0)
    y_batch = torch.stack(ys, dim=0) if show_targets and ys else None

    fig, axes = plt.subplots(int(n), int(n), figsize=(int(n) * figsize_per_cell, int(n) * figsize_per_cell))
    axes = np.asarray(axes)

    for idx in range(k):
        ax = axes[idx // int(n), idx % int(n)]
        x = _denorm_img(x_batch[idx], mean=mean, std=std)
        img = x.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.axis("off")

        if show_targets and y_batch is not None:
            y = y_batch[idx].detach().cpu()
            if targets_are_log1p:
                y = torch.expm1(y).clamp_min(0.0)
            ax.set_title(" ".join([f"{v:.2f}" for v in y.tolist()]), fontsize=8)

    plt.tight_layout()
    plt.show()
