from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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

class WeightedSmoothL1Loss(nn.Module):
    """
    Weighted SmoothL1 on log targets.
    - pred_log, target_log: [B, T]
    - weights: length T
    """
    def __init__(
        self,
        weights=DEFAULT_LOSS_WEIGHTS,
        normalize: bool = True,
        beta: float = 1.0,
    ):
        super().__init__()
        w = torch.as_tensor(weights, dtype=torch.float32)
        self.register_buffer("w", w)
        self.normalize = normalize
        self.beta = float(beta)

    def forward(self, pred_log: torch.Tensor, target_log: torch.Tensor) -> torch.Tensor:
        w = self.w.view(1, -1).to(pred_log.device, dtype=pred_log.dtype)

        # elementwise SmoothL1 (no reduction) -> [B, T]
        err = F.smooth_l1_loss(
            pred_log, target_log, reduction="none", beta=self.beta
        )

        # weighted sum per sample -> [B]
        loss = (err * w).sum(dim=-1)
        if self.normalize:
            loss = loss / (self.w.sum() + 1e-12)

        return loss.mean()
    
    
def std_balanced_weights(
    base_w: torch.Tensor,
    std_t: torch.Tensor,
    *,
    alpha: float = 0.25,
    eps: float = 1e-8,
    ref: str = "mean",  # or "median"
) -> torch.Tensor:
    base_w = base_w.float()
    std_t = std_t.float()

    std_ref = std_t.mean() if ref == "mean" else std_t.median()
    m = (std_ref / (std_t + eps)).pow(alpha)

    w_eff = base_w * m

    # keep overall scale comparable
    w_eff = w_eff * (base_w.sum() / (w_eff.sum() + eps))
    return w_eff