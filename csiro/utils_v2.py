from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Subset

from .config import DEFAULTS
from .eval import _normalize_runs, load_ensemble_states, predict_ensemble, predict_ensemble_tiled


def make_groups_state_quarter(df: pd.DataFrame, date_col: str = "Sampling_Date", state_col: str = "State") -> np.ndarray:
    d = df[date_col]
    if not pd.api.types.is_datetime64_any_dtype(d):
        d = pd.to_datetime(d, errors="raise")
    quarter = d.dt.to_period("Q").astype(str)
    return (df[state_col].astype(str) + "_" + quarter).to_numpy()


def fold_id_from_pairs(groups: np.ndarray, pairs) -> np.ndarray:
    groups = np.asarray(groups)
    uniq_groups = np.asarray(pd.unique(groups))
    fold_id = np.full(groups.shape[0], -1, dtype=np.int64)
    for f, (i, j) in enumerate(pairs):
        gi = uniq_groups[int(i)]
        gj = uniq_groups[int(j)]
        mask = (groups == gi) | (groups == gj)
        fold_id[mask] = f
    if (fold_id < 0).any():
        missing = np.unique(groups[fold_id < 0])
        raise ValueError(f"Unassigned samples. Missing groups: {missing[:10]}")
    return fold_id


def cv_iter_from_pairs(groups_pairs: np.ndarray, pairs, n_splits: int):
    fold_id = fold_id_from_pairs(groups_pairs, pairs)
    for f in range(int(n_splits)):
        va_idx = np.where(fold_id == f)[0]
        tr_idx = np.where(fold_id != f)[0]
        yield tr_idx, va_idx


def build_cv_splits(wide_df: pd.DataFrame, cv_params: dict[str, Any] | None = None):
    if cv_params is None:
        cv_params = DEFAULTS.get("cv_params")
    if not isinstance(cv_params, dict):
        raise ValueError("cv_params must be a dict.")
    if "mode" not in cv_params:
        raise ValueError("cv_params must include 'mode'.")

    mode = str(cv_params["mode"]).lower()
    n_splits = int(cv_params.get("n_splits", DEFAULTS["cv_params"]["n_splits"]))
    if mode == "gkf":
        if "cv_seed" not in cv_params:
            raise ValueError("cv_params must include 'cv_seed' for mode='gkf'.")
        cv_seed = int(cv_params["cv_seed"])
        gkf = GroupKFold(n_splits=int(n_splits), shuffle=True, random_state=int(cv_seed))
        groups = wide_df["Sampling_Date"].values
        return list(gkf.split(wide_df, groups=groups))
    if mode == "pairs":
        if "pairs" not in cv_params:
            raise ValueError("cv_params must include 'pairs' for mode='pairs'.")
        pairs_sel = cv_params["pairs"]
        groups_sq = make_groups_state_quarter(wide_df, "Sampling_Date", "State")
        return list(cv_iter_from_pairs(groups_pairs=groups_sq, pairs=pairs_sel, n_splits=n_splits))
    raise ValueError(f"Unknown cv mode: {cv_params['mode']}")


def build_fold_id(wide_df: pd.DataFrame, cv_params: dict[str, Any] | None = None) -> np.ndarray:
    splits = build_cv_splits(wide_df, cv_params=cv_params)
    fold_id = np.full(len(wide_df), -1, dtype=np.int64)
    for f, (_, va_idx) in enumerate(splits):
        fold_id[np.asarray(va_idx, dtype=np.int64)] = int(f)
    return fold_id


def _infer_tiled_flag(runs: list[list[dict[str, Any]]]) -> bool:
    flags = {bool(s.get("tiled_inp", False)) for run in runs for s in run}
    if len(flags) != 1:
        raise ValueError("Mixed tiled/non-tiled states detected. Use a single mode per OOF run.")
    return flags.pop()


def _filter_runs_for_fold(runs: list[list[dict[str, Any]]], fold_idx: int) -> list[list[dict[str, Any]]]:
    filtered: list[list[dict[str, Any]]] = []
    for run in runs:
        keep = [s for s in run if int(s.get("fold_idx", -1)) != int(fold_idx)]
        if keep:
            filtered.append(keep)
    if not filtered:
        raise ValueError(f"No states left after filtering fold {fold_idx}.")
    return filtered


@torch.no_grad()
def make_oof_predictions(
    *,
    dataset,
    wide_df: pd.DataFrame,
    backbone,
    states: Any | None = None,
    pt_paths: list[str] | str | None = None,
    cv_params: dict[str, Any] | None = None,
    batch_size: int = 128,
    num_workers: int | None = None,
    device: str | torch.device = "cuda",
    tta_agg: str = "mean",
    inner_agg: str = "mean",
    outer_agg: str = "mean",
    tiled_inp: bool | None = None,
) -> dict[str, Any]:
    if pt_paths is None and states is None:
        raise ValueError("Provide pt_paths or states.")

    runs = load_ensemble_states(pt_paths) if pt_paths is not None else _normalize_runs(states)
    if not runs:
        raise ValueError("No states found.")

    if tiled_inp is None:
        tiled_inp = _infer_tiled_flag(runs)
    else:
        tiled_inp = bool(tiled_inp)
        inferred = _infer_tiled_flag(runs)
        if inferred != tiled_inp:
            raise ValueError("tiled_inp does not match checkpoint states.")

    splits = build_cv_splits(wide_df, cv_params=cv_params)
    fold_id = np.full(len(wide_df), -1, dtype=np.int64)
    preds_full: torch.Tensor | None = None

    for f, (_, va_idx) in enumerate(splits):
        fold_id[np.asarray(va_idx, dtype=np.int64)] = int(f)
        runs_f = _filter_runs_for_fold(runs, int(f))
        subset = Subset(dataset, va_idx)

        if tiled_inp:
            preds_f = predict_ensemble_tiled(
                subset,
                states=runs_f,
                backbone=backbone,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                tta_agg=tta_agg,
                inner_agg=inner_agg,
                outer_agg=outer_agg,
            )
        else:
            preds_f = predict_ensemble(
                subset,
                states=runs_f,
                backbone=backbone,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                tta_agg=tta_agg,
                inner_agg=inner_agg,
                outer_agg=outer_agg,
            )

        preds_f = preds_f.detach().cpu()
        if preds_full is None:
            preds_full = torch.zeros((len(wide_df), preds_f.shape[1]), dtype=preds_f.dtype)

        if preds_f.shape[0] != len(va_idx):
            raise ValueError(f"Preds length {preds_f.shape[0]} != va_idx length {len(va_idx)} for fold {f}.")

        preds_full[np.asarray(va_idx, dtype=np.int64)] = preds_f

    if (fold_id < 0).any():
        raise ValueError("Fold assignment incomplete. Check cv_params.")

    return dict(
        preds=preds_full,
        fold_id=fold_id,
        splits=splits,
        tiled_inp=bool(tiled_inp),
    )
