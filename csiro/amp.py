from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch

from . import config


def autocast_context(
    device: str | torch.device,
    dtype: torch.dtype | str | None = None,
) -> ContextManager:
    device_str = str(device)
    if device_str.startswith("cuda"):
        use_dtype = config.parse_dtype(config.DEFAULTS["trainable_dtype"]) if dtype is None else dtype
        if isinstance(use_dtype, str):
            use_dtype = config.parse_dtype(use_dtype)
        return torch.amp.autocast(device_type="cuda", dtype=use_dtype, enabled=True)
    return nullcontext()


def grad_scaler(
    device: str | torch.device,
    dtype: torch.dtype | str | None = None,
) -> torch.cuda.amp.GradScaler:
    device_str = str(device)
    use_dtype = config.parse_dtype(config.DEFAULTS["trainable_dtype"]) if dtype is None else dtype
    if isinstance(use_dtype, str):
        use_dtype = config.parse_dtype(use_dtype)
    enabled = device_str.startswith("cuda") and use_dtype == torch.float16
    return torch.amp.GradScaler(enabled=enabled)
