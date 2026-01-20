from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch

from schedule_utils import deep_merge, dump_yaml, load_yaml, resolve_path


DEFAULT_SCHEDULE = Path(__file__).with_name("schedule.yaml")


def _load_schedule(path: Path) -> dict:
    schedule = load_yaml(path)
    base_dir = path.parent
    output_dir = Path(schedule.get("output_dir", "/notebooks/kaggle/csiro/output"))
    state_file = schedule.get("state_file", "scheduler_state.yaml")
    state_path = output_dir / state_file
    experiments_dir = resolve_path(base_dir, schedule.get("experiments_dir", "configs/experiments"))
    base_config = resolve_path(base_dir, schedule.get("base_config", "configs/base.yaml"))
    completed_dir = resolve_path(base_dir, schedule.get("completed_dir", "configs/completed"))
    repo_root = resolve_path(base_dir, schedule.get("repo_root", "/notebooks/CSIRO"))
    return dict(
        output_dir=output_dir,
        state_path=state_path,
        experiments_dir=experiments_dir,
        base_config=base_config,
        completed_dir=completed_dir,
        repo_root=repo_root,
    )


def _load_state(state_path: Path) -> dict:
    state = load_yaml(state_path)
    state.setdefault("queue_index", 0)
    state.setdefault("ongoing", [])
    state.setdefault("completed", [])
    state.setdefault("skipped", [])
    return state


def _save_state(state_path: Path, state: dict) -> None:
    dump_yaml(state_path, state)


def _resolve_config_path(experiments_dir: Path, config_id: str) -> Path:
    path = Path(config_id)
    if path.is_absolute():
        return path
    return experiments_dir / path


def _load_config(schedule: dict, config_id: str) -> dict:
    base_cfg = load_yaml(schedule["base_config"]) if schedule["base_config"] else {}
    override_path = _resolve_config_path(schedule["experiments_dir"], config_id)
    override_cfg = load_yaml(override_path)
    if "sweeps" in base_cfg or "sweeps" in override_cfg:
        raise ValueError("sweeps are no longer supported; create separate experiment entries.")
    merged = deep_merge(base_cfg, override_cfg)
    return merged


def _resolve_run_name(config: dict, config_id: str) -> str:
    run_name = str(config.get("run_name", "")).strip()
    model_name = str(config.get("model_name", "")).strip()
    if run_name and model_name and run_name != model_name:
        raise ValueError(
            f"run_name and model_name must match for {config_id}; "
            "use run_name only to avoid mismatched artifacts."
        )
    if not run_name:
        run_name = model_name
    if not run_name:
        raise ValueError(f"run_name is required for {config_id}.")
    return run_name


def _model_paths(output_dir: Path, run_name: str) -> dict:
    states_dir = output_dir / "states"
    return dict(
        checkpoint=states_dir / f"{run_name}_checkpoint.pt",
        cv_state=states_dir / f"{run_name}_cv_state.pt",
        final=(output_dir / "complete" / f"{run_name}.pt"),
    )


def _move_config_to_completed(schedule: dict, config_id: str) -> None:
    config_path = _resolve_config_path(schedule["experiments_dir"], config_id)
    if not config_path.exists():
        return
    completed_dir = schedule["completed_dir"]
    completed_dir.mkdir(parents=True, exist_ok=True)
    dest = completed_dir / config_path.name
    if dest.exists():
        stem = config_path.stem
        suffix = config_path.suffix
        idx = 1
        while True:
            candidate = completed_dir / f"{stem}_{idx}{suffix}"
            if not candidate.exists():
                dest = candidate
                break
            idx += 1
    config_path.replace(dest)

def _ensure_checkpoint_link(checkpoint: Path, cv_state: Path) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() or checkpoint.is_symlink():
        return
    try:
        checkpoint.symlink_to(cv_state)
    except OSError:
        if cv_state.exists():
            shutil.copy2(cv_state, checkpoint)


def _ensure_cv_state(checkpoint: Path, cv_state: Path) -> None:
    cv_state.parent.mkdir(parents=True, exist_ok=True)
    if cv_state.exists():
        return
    if checkpoint.exists() and checkpoint.is_file():
        shutil.copy2(checkpoint, cv_state)


def _finalize_output(output_dir: Path, run_name: str) -> None:
    complete_dir = output_dir / "complete"
    complete_dir.mkdir(parents=True, exist_ok=True)
    final_path = complete_dir / f"{run_name}.pt"
    if not final_path.exists():
        return


def _mark_completed(state_path: Path, config_id: str) -> None:
    state = _load_state(state_path)
    if config_id in state.get("ongoing", []):
        state["ongoing"] = [c for c in state["ongoing"] if c != config_id]
    if config_id not in state.get("completed", []):
        state["completed"].append(config_id)
    _save_state(state_path, state)


def _run_training(config: dict, *, run_name: str, output_dir: Path, repo_root: Path) -> None:
    if repo_root and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from csiro.config import (
        DEFAULT_DATA_ROOT,
        DEFAULT_DINO_ROOT,
        DEFAULTS,
        dino_hub_name,
        dino_weights_path_from_size,
        neck_num_heads_for,
    )
    from csiro.data import BiomassBaseCached, BiomassFullCached, BiomassTiledCached, load_train_wide
    from csiro.train import run_groupkfold_cv

    cfg = dict(DEFAULTS)
    cfg.update(config)
    cfg["save_output_dir"] = str(output_dir)
    device = str(cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    csv = cfg.pop("csv", None)
    root = cfg.pop("root", DEFAULT_DATA_ROOT)
    dino_repo = cfg.pop("dino_repo", DEFAULT_DINO_ROOT)
    dino_weights = cfg.pop("dino_weights", None)
    model_size = cfg.pop("model_size", None)
    plus = cfg.pop("plus", "")

    if csv is None:
        csv = os.path.join(str(root), "train.csv")

    sys.path.insert(0, str(dino_repo))
    wide_df = load_train_wide(str(csv), root=str(root))
    dataset_cache: dict[str, object] = {}
    backbone_cache: dict[tuple[str, str, str], object] = {}

    kwargs = dict(cfg)
    if (
        "backbone_size" in kwargs
        and "neck_num_heads" not in kwargs
        and int(kwargs.get("num_neck", 0)) > 0
    ):
        kwargs["neck_num_heads"] = neck_num_heads_for(kwargs["backbone_size"])

    sweep_model_size = str(
        kwargs.get("backbone_size", model_size or cfg.get("backbone_size", "b"))
    )
    sweep_dino_weights = dino_weights
    if sweep_dino_weights is None:
        sweep_dino_weights = dino_weights_path_from_size(
            str(kwargs.get("backbone_size", "b"))
        )
    if sweep_dino_weights is None:
        raise ValueError("Set DINO_B_WEIGHTS_PATH or DINO_L_WEIGHTS_PATH for the chosen backbone_size.")

    cache_key = (str(sweep_model_size), str(sweep_dino_weights), str(plus))
    if cache_key not in backbone_cache:
        backbone_cache[cache_key] = torch.hub.load(
            str(dino_repo),
            dino_hub_name(model_size=str(sweep_model_size), plus=str(plus)),
            source="local",
            weights=str(sweep_dino_weights),
        )

    kwargs["backbone"] = backbone_cache[cache_key]
    tiled_inp = bool(kwargs.get("tiled_inp", cfg.get("tiled_inp", False)))
    model_arch = str(kwargs.get("model_name", cfg.get("model_name", ""))).strip().lower()
    full_rect = model_arch in ("rect_full", "full_rect", "rect")
    if full_rect and tiled_inp:
        raise ValueError("rect_full model requires tiled_inp=False.")
    if full_rect:
        tiled_inp = False
    tile_geom_mode = str(kwargs.get("tile_geom_mode", cfg.get("tile_geom_mode", "shared"))).strip().lower()
    if tile_geom_mode not in ("shared", "independent"):
        raise ValueError(f"tile_geom_mode must be 'shared' or 'independent' (got {tile_geom_mode})")
    use_shared_geom = tiled_inp and tile_geom_mode == "shared"
    img_preprocess = bool(kwargs.get("img_preprocess", cfg.get("img_preprocess", False)))
    cache_key = "full_rect" if full_rect else ("shared" if use_shared_geom else ("tiled" if tiled_inp else "base"))
    if cache_key not in dataset_cache:
        if full_rect:
            dataset_cache[cache_key] = BiomassFullCached(
                wide_df,
                img_preprocess=img_preprocess,
            )
        elif use_shared_geom:
            dataset_cache[cache_key] = BiomassFullCached(
                wide_df,
                img_preprocess=img_preprocess,
            )
        elif tiled_inp:
            dataset_cache[cache_key] = BiomassTiledCached(
                wide_df,
                img_size=int(cfg["img_size"]),
                img_preprocess=img_preprocess,
            )
        else:
            dataset_cache[cache_key] = BiomassBaseCached(
                wide_df,
                img_size=int(cfg["img_size"]),
                img_preprocess=img_preprocess,
            )
    kwargs["dataset"] = dataset_cache[cache_key]
    kwargs["wide_df"] = wide_df
    kwargs["config_name"] = run_name
    run_groupkfold_cv(return_details=True, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_id")
    parser.add_argument("--schedule", default=str(DEFAULT_SCHEDULE))
    args = parser.parse_args()

    schedule = _load_schedule(Path(args.schedule))
    config = _load_config(schedule, args.config_id)
    run_name = _resolve_run_name(config, args.config_id)
    config = dict(config)
    config.pop("run_name", None)
    config["model_name"] = run_name

    output_dir = schedule["output_dir"]
    paths = _model_paths(output_dir, run_name)
    paths["final"].parent.mkdir(parents=True, exist_ok=True)

    if paths["final"].exists():
        paths["checkpoint"].unlink(missing_ok=True)
        paths["cv_state"].unlink(missing_ok=True)
        _move_config_to_completed(schedule, args.config_id)
        _mark_completed(schedule["state_path"], args.config_id)
        return 0

    _ensure_cv_state(paths["checkpoint"], paths["cv_state"])
    _ensure_checkpoint_link(paths["checkpoint"], paths["cv_state"])

    _run_training(
        config,
        run_name=run_name,
        output_dir=output_dir,
        repo_root=schedule["repo_root"],
    )
    _finalize_output(output_dir, run_name)
    if paths["final"].exists():
        paths["checkpoint"].unlink(missing_ok=True)
        paths["cv_state"].unlink(missing_ok=True)
        _move_config_to_completed(schedule, args.config_id)
        _mark_completed(schedule["state_path"], args.config_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
