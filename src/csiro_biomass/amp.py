from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch

DTYPE: torch.dtype = torch.float16


def set_dtype(dtype: torch.dtype) -> None:
    global DTYPE
    DTYPE = dtype


def autocast_context(device: str | torch.device) -> ContextManager:
    device_str = str(device)
    if device_str.startswith("cuda"):
        return torch.amp.autocast(device_type="cuda", dtype=DTYPE, enabled=True)
    return nullcontext()


def grad_scaler(device: str | torch.device) -> torch.cuda.amp.GradScaler:
    device_str = str(device)
    enabled = device_str.startswith("cuda") and DTYPE == torch.float16
    return torch.cuda.amp.GradScaler(enabled=enabled)

