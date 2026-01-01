from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


from .config import DEFAULT_LOSS_WEIGHTS, TARGETS


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


class PhysicsConsistencyLoss(nn.Module):
    def __init__(
        self,
        tau_physics: float = 0.0,
        *,
        from_log: bool = True,
        w_total: float = 0.25,
        w_gdm: float = 1.0,
        w_ineq: float = 0.5,
    ):
        super().__init__()
        self.tau = float(tau_physics)
        self.from_log = bool(from_log)
        self.w_total = float(w_total)
        self.w_gdm = float(w_gdm)
        self.w_ineq = float(w_ineq)
        try:
            self.idx_green = TARGETS.index("Dry_Green_g")
            self.idx_clover = TARGETS.index("Dry_Clover_g")
            self.idx_dead = TARGETS.index("Dry_Dead_g")
            self.idx_total = TARGETS.index("Dry_Total_g")
            self.idx_gdm = TARGETS.index("GDM_g")
        except ValueError as exc:
            raise ValueError(
                "TARGETS must include Dry_Green_g, Dry_Clover_g, Dry_Dead_g, Dry_Total_g, GDM_g."
            ) from exc

    def forward(
        self,
        pred: torch.Tensor,
        *,
        comet_exp: object | None = None,
        step: int | None = None,
    ) -> torch.Tensor:
        if self.tau <= 0.0:
            return pred.sum() * 0.0
        if self.from_log:
            p = torch.expm1(pred.float()).clamp_min(0.0)
        else:
            p = pred.float().clamp_min(0.0)

        total = p[:, self.idx_total]
        gdm = p[:, self.idx_gdm]
        green = p[:, self.idx_green]
        clover = p[:, self.idx_clover]
        dead = p[:, self.idx_dead]

        loss_total = (total - (green + clover + dead)).abs().mean()
        loss_gdm = (gdm - (green + clover)).abs().mean()
        loss_ineq = (gdm - total).clamp_min(0.0).mean()
        if comet_exp is not None:
            try:
                comet_exp.log_metrics(
                    {
                        "phys_loss_total": float(loss_total.detach().item()),
                        "phys_loss_gdm": float(loss_gdm.detach().item()),
                        "phys_loss_ineq": float(loss_ineq.detach().item()),
                    },
                    step=step,
                )
            except Exception:
                pass
        loss = (self.w_total * loss_total) + (self.w_gdm * loss_gdm) + (self.w_ineq * loss_ineq)
        return loss * self.tau

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
