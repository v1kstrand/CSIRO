from __future__ import annotations

import torch

from .amp import DTYPE

@torch.no_grad()
def eval_global_wr2(model, dl_va, w_vec: torch.Tensor, device: str | torch.device = "cuda") -> float:
    if hasattr(model, "set_train"):
        model.set_train(False)
    model.eval()
    w5 = w_vec.to(device).view(1, -1)
    ss_res = torch.zeros((), device=device)
    sum_w = torch.zeros((), device=device)
    sum_wy = torch.zeros((), device=device)
    sum_wy2 = torch.zeros((), device=device)

    device_str = str(device)
    with torch.inference_mode(), torch.amp.autocast(
        device_type="cuda",
        dtype=DTYPE,
        enabled=device_str.startswith("cuda"),
    ):
        for x, y_log in dl_va:
            x = x.to(device, non_blocking=True)
            y_log = y_log.to(device, non_blocking=True)
            p_log = model(x).float()

            y = torch.expm1(y_log)
            p = torch.expm1(p_log).clamp_min(0.0)

            w = w5.expand_as(y)
            diff = y - p

            ss_res += (w * diff * diff).sum()
            sum_w += w.sum()
            sum_wy += (w * y).sum()
            sum_wy2 += (w * y * y).sum()

    mu = sum_wy / (sum_w + 1e-12)
    ss_tot = sum_wy2 - sum_w * mu * mu
    return (1.0 - ss_res / (ss_tot + 1e-12)).item()
