from __future__ import annotations

import inspect
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

from .transforms import base_train_comp


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
        raise ValueError("preview_augments requires a dataset (no default loader in utils.py).")

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


def filter_kwargs(func, kwargs: dict) -> dict:
    """
    Keep only keys that `func` can accept as keyword args.
    If func has **kwargs, keep everything.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {}

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)

    allowed = {
        name for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.KEYWORD_ONLY)
    }
    return {k: v for k, v in kwargs.items() if k in allowed}