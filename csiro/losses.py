from __future__ import annotations

import torch
import torch.nn as nn
from .config import DEFAULT_LOSS_WEIGHTS, TARGETS

try:
    _IDX_GREEN = TARGETS.index("Dry_Green_g")
    _IDX_CLOVER = TARGETS.index("Dry_Clover_g")
    _IDX_DEAD = TARGETS.index("Dry_Dead_g")
except ValueError as exc:
    raise ValueError("TARGETS must include Dry_Green_g, Dry_Clover_g, Dry_Dead_g.") from exc


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


class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, weights=DEFAULT_LOSS_WEIGHTS, *, beta: float = 1.0, normalize: bool = True):
        super().__init__()
        w = torch.as_tensor(weights, dtype=torch.float32)
        self.register_buffer("w", w)
        self.beta = float(beta)
        self.normalize = normalize

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.w.view(1, -1)
        err = torch.nn.functional.smooth_l1_loss(
            pred,
            target,
            reduction="none",
            beta=self.beta,
        )
        loss = (err * w).sum(dim=-1)
        if self.normalize:
            loss = loss / (self.w.sum() + 1e-12)
        return loss.mean()


class NegativityPenaltyLoss(nn.Module):
    def __init__(self, tau_neg: float = 0.0, *, pred_space: str = "log"):
        super().__init__()
        self.tau = float(tau_neg)
        self.pred_space = str(pred_space).strip().lower()
        if self.pred_space not in ("log", "log1p", "gram", "grams", "linear"):
            raise ValueError(f"Unknown pred_space: {pred_space}")

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        if self.tau <= 0.0:
            return pred.sum() * 0.0
        if self.pred_space in ("log", "log1p"):
            vals = torch.expm1(pred.float())
        else:
            vals = pred.float()
        idx = torch.as_tensor([_IDX_GREEN, _IDX_CLOVER, _IDX_DEAD], device=pred.device)
        comps = torch.index_select(vals, dim=-1, index=idx)
        penalty = torch.relu(-comps).mean()
        return penalty * self.tau
