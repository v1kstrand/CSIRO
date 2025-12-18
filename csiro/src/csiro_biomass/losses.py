from __future__ import annotations

import torch
import torch.nn as nn

from .config import DEFAULT_LOSS_WEIGHTS


class WeightedMSELoss(nn.Module):
    def __init__(self, weights=DEFAULT_LOSS_WEIGHTS, normalize: bool = True):
        super().__init__()
        w = torch.as_tensor(weights, dtype=torch.float32)
        self.register_buffer("w", w)
        self.normalize = normalize

    def forward(self, pred_log: torch.Tensor, target_log: torch.Tensor) -> torch.Tensor:
        w = self.w.view(1, -1)
        err2 = (pred_log - target_log).pow(2)
        loss = (err2 * w).sum(dim=-1)
        if self.normalize:
            loss = loss / (self.w.sum() + 1e-12)
        return loss.mean()
