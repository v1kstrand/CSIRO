from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from .amp import autocast_context
from .config import DEFAULTS, default_num_workers, parse_dtype
from .model import DINOv3Regressor
from .transforms import post_tfms


def _flatten_states(states: Any) -> list[dict[str, Any]]:
    if isinstance(states, dict):
        states = [states]
    if states and isinstance(states[0], list):
        states = [s for fold in states for s in fold]
    return list(states)


def _agg_stack(xs: list[torch.Tensor], agg: str) -> torch.Tensor:
    if len(xs) == 1:
        return xs[0]
    agg = str(agg).lower()
    stacked = torch.stack(xs, dim=0)
    if agg == "mean":
        return stacked.mean(dim=0)
    if agg == "median":
        return stacked.median(dim=0).values
    raise ValueError(f"Unknown aggregation: {agg}")


def _ensure_tensor_batch(x, tfms) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        xs = [xi if torch.is_tensor(xi) else tfms(xi) for xi in x]
        return torch.stack(xs, dim=0)
    return tfms(x).unsqueeze(0)


def _build_model_from_state(
    backbone,
    state: dict[str, Any],
    device: str | torch.device,
    backbone_dtype: torch.dtype | None = None,
):
    model = DINOv3Regressor(
        backbone,
        hidden=int(state["head_hidden"]),
        drop=float(state["head_drop"]),
        depth=int(state["head_depth"]),
        num_neck=int(state["num_neck"]),
        backbone_dtype=backbone_dtype,
    ).to(device)
    parts = state.get("parts")
    if isinstance(parts, dict):
        for name in ("neck", "head", "norm"):
            part = getattr(model, name, None)
            if part is not None and name in parts:
                part.load_state_dict(parts[name], strict=True)
    else:
        model.load_state_dict(state, strict=False)
    return model


def load_states_from_pt(pt_path: str) -> list[dict[str, Any]]:
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    states = ckpt["states"] if isinstance(ckpt, dict) and "states" in ckpt else ckpt
    return _flatten_states(states)


def _require(state: dict[str, Any], key: str) -> Any:
    if key not in state:
        raise ValueError(f"Missing '{key}' in checkpoint state.")
    return state[key]


def load_dinov3_regressor_from_pt(
    pt_path: str,
    backbone,
    *,
    device: str | torch.device = "cuda",
    backbone_dtype: str | torch.dtype | None = None,
    state_idx: int = 0,
) -> DINOv3Regressor:
    states = load_states_from_pt(pt_path)
    if not states:
        raise ValueError("No states found in checkpoint.")
    if int(state_idx) >= len(states):
        raise IndexError(f"state_idx {state_idx} out of range for {len(states)} states.")

    state = states[int(state_idx)]
    if backbone_dtype is None:
        backbone_dtype = DEFAULTS["backbone_dtype"]
    if isinstance(backbone_dtype, str):
        backbone_dtype = parse_dtype(backbone_dtype)

    model = DINOv3Regressor(
        backbone,
        hidden=int(_require(state, "head_hidden")),
        drop=float(_require(state, "head_drop")),
        depth=int(_require(state, "head_depth")),
        num_neck=int(_require(state, "num_neck")),
        backbone_dtype=backbone_dtype,
    ).to(device)

    parts = state.get("parts")
    if isinstance(parts, dict):
        for name in ("neck", "head", "norm"):
            part = getattr(model, name, None)
            if part is not None and name in parts:
                part.load_state_dict(parts[name], strict=True)
    else:
        model.load_state_dict(state, strict=False)

    if hasattr(model, "set_train"):
        model.set_train(False)
    model.eval()
    return model


@torch.no_grad()
def forward_trainable_random(
    model: DINOv3Regressor,
    seed: int,
    *,
    batch_size: int = 1,
    tokens_len: int = 197,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
    head_only: bool = False,
) -> list[torch.Tensor]:
    if hasattr(model, "set_train"):
        model.set_train(False)
    model.eval()
    model = model.to(device)

    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            dtype = torch.float32

    feat_dim = int(model.feat_dim)
    g = torch.Generator()
    g.manual_seed(int(seed))
    x = torch.rand(
        int(batch_size),
        int(tokens_len),
        feat_dim,
        generator=g,
        dtype=dtype,
    )
    x = x.to(device)
    
    tokens = x
    if not head_only:
        for block in model.neck:
            try:
                tokens = block(tokens, None)
            except TypeError:
                tokens = block(tokens)
    cls = tokens[:, 0, :]
    cls = model.norm(cls)
    y = model.head(cls)

    return [y]


@torch.no_grad()
def predict_ensemble(
    data,
    states: list[dict[str, Any]],
    backbone,
    *,
    batch_size: int = 128,
    num_workers: int | None = None,
    device: str | torch.device = "cuda",
    backbone_dtype: str | torch.dtype | None = None,
    tta_rot90: bool = True,
    tta_agg: str = "mean",
    ens_agg: str = "mean",
) -> torch.Tensor:
    states = _flatten_states(states)

    if isinstance(data, DataLoader):
        dl = data
    else:
        num_workers = default_num_workers() if num_workers is None else int(num_workers)
        dl = DataLoader(
            data,
            shuffle=False,
            batch_size=int(batch_size),
            pin_memory=str(device).startswith("cuda"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

    if backbone_dtype is None:
        backbone_dtype = DEFAULTS["backbone_dtype"]
    if isinstance(backbone_dtype, str):
        backbone_dtype = parse_dtype(backbone_dtype)
    models = [_build_model_from_state(backbone, s, device, backbone_dtype) for s in states]
    for model in models:
        if hasattr(model, "set_train"):
            model.set_train(False)
        model.eval()

    tfms = post_tfms()
    n_rots = 4 if tta_rot90 else 1
    preds: list[torch.Tensor] = []
    with torch.inference_mode(), autocast_context(device):
        for batch in dl:
            if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                x = batch[0]
            else:
                x = batch

            x = _ensure_tensor_batch(x, tfms).to(device, non_blocking=True)

            preds_models: list[torch.Tensor] = []
            for model in models:
                preds_rots: list[torch.Tensor] = []
                for k in range(n_rots):
                    x_rot = x if k == 0 else torch.rot90(x, k, dims=(-2, -1))
                    p_log = model(x_rot).float()
                    p = torch.expm1(p_log).clamp_min(0.0)
                    preds_rots.append(p)
                preds_models.append(_agg_stack(preds_rots, tta_agg))

            p_ens = _agg_stack(preds_models, ens_agg)
            preds.append(p_ens.detach().cpu())

    return torch.cat(preds, dim=0)


def predict_ensemble_from_pt(
    data,
    pt_path: str,
    backbone,
    *,
    batch_size: int = 128,
    num_workers: int | None = None,
    device: str | torch.device = "cuda",
    backbone_dtype: str | torch.dtype | None = None,
    tta_rot90: bool = True,
    tta_agg: str = "mean",
    ens_agg: str = "mean",
) -> torch.Tensor:
    states = load_states_from_pt(pt_path)
    return predict_ensemble(
        data,
        states=states,
        backbone=backbone,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        backbone_dtype=backbone_dtype,
        tta_rot90=tta_rot90,
        tta_agg=tta_agg,
        ens_agg=ens_agg,
    )
