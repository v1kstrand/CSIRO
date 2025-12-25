from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch

from . import config


def set_dtype(dtype: torch.dtype) -> None:
    config.DTYPE = dtype


def autocast_context(device: str | torch.device, dtype: torch.dtype | None = None) -> ContextManager:
    device_str = str(device)
    if device_str.startswith("cuda"):
        use_dtype = config.DTYPE if dtype is None else dtype
        return torch.amp.autocast(device_type="cuda", dtype=use_dtype, enabled=True)
    return nullcontext()


def grad_scaler(device: str | torch.device) -> torch.cuda.amp.GradScaler:
    device_str = str(device)
    enabled = device_str.startswith("cuda") and config.DTYPE == torch.float16
    return torch.amp.GradScaler(enabled=enabled)
