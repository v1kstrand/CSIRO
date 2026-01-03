from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupKFold
from torch.utils.data import Subset

from .config import DEFAULTS, TARGETS
from .data import TransformView, TiledTransformView
from .eval import _normalize_runs, load_ensemble_states, predict_ensemble, predict_ensemble_tiled
from .transforms import post_tfms


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


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def _kl(a, b):
        a = np.clip(a, 1e-12, None)
        b = np.clip(b, 1e-12, None)
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _season_from_month(m: float | int | None) -> str | None:
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return None
    m = int(m)
    if m in (12, 1, 2):
        return "summer"
    if m in (3, 4, 5):
        return "autumn"
    if m in (6, 7, 8):
        return "winter"
    return "spring"


def _score_split(
    *,
    wide_df: pd.DataFrame,
    splits,
    targets: list[str],
    date_col: str,
    state_col: str,
    min_fold_n: int,
    min_target_var: float,
    min_states_per_fold: int | None,
    min_seasons_per_fold: int | None,
    n_bins: int,
    min_bin_n: int,
) -> tuple[bool, dict[str, Any]]:
    n = len(wide_df)
    fold_sizes = [len(va) for _, va in splits]
    if min(fold_sizes) < int(min_fold_n):
        return False, {"reject": "min_fold_n"}

    y = wide_df[targets].to_numpy(dtype=np.float32)
    overall_mean = y.mean(axis=0)
    overall_var = y.var(axis=0)

    min_var = float("inf")
    mean_dev = []
    var_dev = []
    for _, va_idx in splits:
        y_f = y[np.asarray(va_idx, dtype=np.int64)]
        v = y_f.var(axis=0)
        min_var = min(min_var, float(v.min()))
        mean_dev.append(np.abs(y_f.mean(axis=0) - overall_mean) / (overall_mean + 1e-6))
        var_dev.append(np.abs(v - overall_var) / (overall_var + 1e-6))

    if min_var < float(min_target_var):
        return False, {"reject": "min_target_var"}

    # Quantile coverage
    bins = []
    for t_idx in range(len(targets)):
        q = np.quantile(y[:, t_idx], np.linspace(0, 1, n_bins + 1))
        bins.append(q)
    for _, va_idx in splits:
        y_f = y[np.asarray(va_idx, dtype=np.int64)]
        for t_idx in range(len(targets)):
            q = bins[t_idx]
            counts, _ = np.histogram(y_f[:, t_idx], bins=q)
            if counts.min() < int(min_bin_n):
                return False, {"reject": "min_bin_n"}

    # State/season balance scoring
    state_score = None
    season_score = None
    if state_col in wide_df.columns:
        states = wide_df[state_col].astype(str)
        uniq = states.unique()
        overall = states.value_counts(normalize=True).reindex(uniq).fillna(0).to_numpy()
        js = []
        for _, va_idx in splits:
            vals = states.iloc[va_idx]
            if min_states_per_fold is not None and vals.nunique() < int(min_states_per_fold):
                return False, {"reject": "min_states_per_fold"}
            dist = vals.value_counts(normalize=True).reindex(uniq).fillna(0).to_numpy()
            js.append(_js_divergence(dist, overall))
        state_score = 1.0 - float(np.mean(js))

    if date_col in wide_df.columns:
        d = pd.to_datetime(wide_df[date_col], errors="coerce")
        seasons = d.dt.month.apply(_season_from_month)
        uniq = seasons.dropna().unique()
        overall = seasons.value_counts(normalize=True).reindex(uniq).fillna(0).to_numpy()
        js = []
        for _, va_idx in splits:
            vals = seasons.iloc[va_idx]
            if min_seasons_per_fold is not None and vals.nunique() < int(min_seasons_per_fold):
                return False, {"reject": "min_seasons_per_fold"}
            dist = vals.value_counts(normalize=True).reindex(uniq).fillna(0).to_numpy()
            js.append(_js_divergence(dist, overall))
        season_score = 1.0 - float(np.mean(js))

    size_balance = 1.0 - float(np.std(fold_sizes) / (np.mean(fold_sizes) + 1e-6))
    mean_balance = 1.0 - float(np.mean(np.vstack(mean_dev)))
    var_balance = 1.0 - float(np.mean(np.vstack(var_dev)))

    parts = {"size": size_balance, "mean": mean_balance, "var": var_balance}
    if state_score is not None:
        parts["state"] = state_score
    if season_score is not None:
        parts["season"] = season_score

    weights = {
        "size": 0.2,
        "state": 0.25,
        "season": 0.15,
        "mean": 0.2,
        "var": 0.2,
    }
    w_sum = sum(weights[k] for k in parts.keys())
    score = sum(parts[k] * weights[k] for k in parts.keys()) / max(w_sum, 1e-6)

    return True, {
        "score": float(score),
        "fold_sizes": fold_sizes,
        "min_target_var": float(min_var),
        "size_balance": float(size_balance),
        "mean_balance": float(mean_balance),
        "var_balance": float(var_balance),
        "state_balance": None if state_score is None else float(state_score),
        "season_balance": None if season_score is None else float(season_score),
    }


def search_cv_splits(
    wide_df: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_trials: int = 200,
    seed_start: int = 0,
    top_k: int = 5,
    group_mode: str = "state_quarter",
    date_col: str = "Sampling_Date",
    state_col: str = "State",
    targets: list[str] | None = None,
    min_fold_n: int | None = None,
    min_target_var: float = 1e-3,
    min_states_per_fold: int | None = None,
    min_seasons_per_fold: int | None = 2,
    n_bins: int = 4,
    min_bin_n: int = 5,
) -> list[dict[str, Any]]:
    if targets is None:
        targets = list(DEFAULTS.get("targets", [])) or []
    if not targets:
        targets = list(wide_df.columns.intersection(TARGETS))

    if min_fold_n is None:
        min_fold_n = max(20, int(len(wide_df) / max(int(n_splits), 1) * 0.6))

    if group_mode == "state_quarter":
        groups = make_groups_state_quarter(wide_df, date_col=date_col, state_col=state_col)
    elif group_mode == "date":
        groups = wide_df[date_col].astype(str).to_numpy()
    else:
        raise ValueError(f"Unknown group_mode: {group_mode}")

    results: list[dict[str, Any]] = []
    for seed in range(int(seed_start), int(seed_start) + int(n_trials)):
        gkf = GroupKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
        splits = list(gkf.split(wide_df, groups=groups))
        ok, info = _score_split(
            wide_df=wide_df,
            splits=splits,
            targets=targets,
            date_col=date_col,
            state_col=state_col,
            min_fold_n=int(min_fold_n),
            min_target_var=float(min_target_var),
            min_states_per_fold=min_states_per_fold,
            min_seasons_per_fold=min_seasons_per_fold,
            n_bins=int(n_bins),
            min_bin_n=int(min_bin_n),
        )
        if not ok:
            continue
        info["cv_params"] = dict(mode="gkf", cv_seed=int(seed), n_splits=int(n_splits))
        info["group_mode"] = str(group_mode)
        results.append(info)

    results.sort(key=lambda d: float(d.get("score", -1)), reverse=True)
    return results[: int(top_k)]

def _ensure_tensor_dataset(dataset, *, tiled_inp: bool):
    sample = dataset[0]
    if isinstance(sample, (tuple, list)) and len(sample) >= 1:
        x0 = sample[0]
    else:
        x0 = sample

    if torch.is_tensor(x0):
        return dataset

    if tiled_inp:
        if isinstance(sample, (tuple, list)) and len(sample) >= 3:
            return TiledTransformView(dataset, post_tfms(), tile_swap=False)
        raise ValueError("tiled_inp expects dataset to return (left, right, y) or tensors.")

    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return TransformView(dataset, post_tfms())
    if isinstance(x0, Image.Image):
        raise ValueError("dataset returns PIL without labels; provide a labeled dataset for OOF.")
    raise ValueError("Unsupported dataset sample format for OOF.")


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
    dataset = _ensure_tensor_dataset(dataset, tiled_inp=bool(tiled_inp))
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
