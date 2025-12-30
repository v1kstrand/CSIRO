from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from .amp import autocast_context
from .config import DEFAULTS, default_num_workers, parse_dtype
from .model import DINOv3Regressor
from .transforms import post_tfms, TTABatch


def _is_state_dict(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    keys = state.keys()
    return any(k in keys for k in ("head_hidden", "head_depth", "head_drop", "num_neck", "parts"))


def _flatten_states(states: Any) -> list[dict[str, Any]]:
    if _is_state_dict(states):
        return [states]
    if isinstance(states, dict) and "states" in states:
        states = states["states"]
    if isinstance(states, dict):
        states = list(states.values())
    if states and isinstance(states[0], list):
        states = [s for fold in states for s in fold]
    return list(states)


def _extract_seed_states(states: Any) -> dict[str, list[dict[str, Any]]]:
    if _is_state_dict(states):
        return {"0": [states]}
    if isinstance(states, dict) and "seed_results" in states:
        states = states["seed_results"]
    if isinstance(states, dict) and "states" in states:
        return {"0": _flatten_states(states["states"])}
    if isinstance(states, dict):
        seed_map: dict[str, list[dict[str, Any]]] = {}
        for k, v in states.items():
            if isinstance(v, dict) and "states" in v:
                v = v["states"]
            seed_map[str(k)] = _flatten_states(v)
        return seed_map
    return {"0": _flatten_states(states)}


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


def _agg_tta(p: torch.Tensor, agg: str) -> torch.Tensor:
    agg = str(agg).lower()
    if agg == "mean":
        return p.mean(dim=1)
    if agg == "median":
        return p.median(dim=1).values
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


def load_states_from_pt(pt_path: str) -> Any:
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    return ckpt["states"] if isinstance(ckpt, dict) and "states" in ckpt else ckpt


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
    seed: str | int | None = None,
) -> DINOv3Regressor:
    states_raw = load_states_from_pt(pt_path)
    seed_states = _extract_seed_states(states_raw)
    if not seed_states:
        raise ValueError("No states found in checkpoint.")
    if seed is None:
        seed = sorted(seed_states.keys())[0]
    seed_key = str(seed)
    if seed_key not in seed_states:
        raise KeyError(f"Seed '{seed}' not found in checkpoint.")
    states = seed_states[seed_key]
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
def predict_ensemble(
    data,
    states: Any,
    backbone,
    *,
    batch_size: int = 128,
    num_workers: int | None = None,
    device: str | torch.device = "cuda",
    backbone_dtype: str | torch.dtype | None = None,
    tta_rot90: bool = True,
    tta_agg: str = "mean",
    ens_agg: str = "mean",
    seed_agg: str = "flatten",
    tta_n: int | None = None,
    tta_bcs_val: float = 0.0,
    tta_hue_val: float = 0.0,
) -> torch.Tensor:
    seed_states = _extract_seed_states(states)

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

    tfms = post_tfms()
    n_rots = 4 if tta_rot90 else 1
    if tta_n is None:
        tta_n = int(n_rots)
    tta_batch = TTABatch(tta_n=int(tta_n), bcs_val=float(tta_bcs_val), hue_val=float(tta_hue_val))
    seed_agg = str(seed_agg).lower()

    def _predict_with_models(models: list[DINOv3Regressor]) -> torch.Tensor:
        for model in models:
            if hasattr(model, "set_train"):
                model.set_train(False)
            model.eval()

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
                    x_tta = tta_batch(x, flatten=True)
                    p_log = model(x_tta).float()
                    p = torch.expm1(p_log).clamp_min(0.0)
                    p = p.view(x.size(0), int(tta_n), -1)
                    preds_models.append(_agg_tta(p, tta_agg))

                p_ens = _agg_stack(preds_models, ens_agg)
                preds.append(p_ens.detach().cpu())

        return torch.cat(preds, dim=0)

    if seed_agg == "flatten":
        flat_states = [s for ss in seed_states.values() for s in ss]
        models = [_build_model_from_state(backbone, s, device, backbone_dtype) for s in flat_states]
        return _predict_with_models(models)
    if seed_agg in ("mean", "median"):
        preds_seeds: list[torch.Tensor] = []
        for seed_key in sorted(seed_states.keys()):
            models = [
                _build_model_from_state(backbone, s, device, backbone_dtype) for s in seed_states[seed_key]
            ]
            preds_seeds.append(_predict_with_models(models))
        return _agg_stack(preds_seeds, seed_agg)
    raise ValueError(f"Unknown seed_agg: {seed_agg}")


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
    seed_agg: str = "flatten",
    tta_n: int | None = None,
    tta_bcs_val: float = 0.0,
    tta_hue_val: float = 0.0,
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
        seed_agg=seed_agg,
        tta_n=tta_n,
        tta_bcs_val=tta_bcs_val,
        tta_hue_val=tta_hue_val,
    )
