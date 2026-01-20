from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    queue = list(schedule.get("config_queue", []) or [])
    max_runs = int(schedule.get("max_runs", 1))
    return dict(
        schedule_path=path,
        output_dir=output_dir,
        state_path=state_path,
        experiments_dir=experiments_dir,
        base_config=base_config,
        queue=queue,
        max_runs=max_runs,
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
    return deep_merge(base_cfg, override_cfg)


def _model_paths(output_dir: Path, model_name: str) -> dict:
    states_dir = output_dir / "states"
    return dict(
        checkpoint=states_dir / f"{model_name}_checkpoint.pt",
        cv_state=states_dir / f"{model_name}_cv_state.pt",
        final=(output_dir / "complete" / f"{model_name}.pt"),
    )


def _cleanup_ongoing(schedule: dict, state: dict) -> None:
    output_dir = schedule["output_dir"]
    ongoing = []
    for config_id in list(state.get("ongoing", [])):
        try:
            cfg = _load_config(schedule, config_id)
        except Exception as exc:
            print(f"[scheduler] skip {config_id}: {exc}", file=sys.stderr)
            state.setdefault("skipped", []).append(config_id)
            continue
        model_name = str(cfg.get("model_name", "")).strip()
        if not model_name:
            print(f"[scheduler] skip {config_id}: missing model_name", file=sys.stderr)
            state.setdefault("skipped", []).append(config_id)
            continue
        paths = _model_paths(output_dir, model_name)
        if paths["final"].exists():
            if paths["checkpoint"].exists():
                paths["checkpoint"].unlink(missing_ok=True)
            if paths["cv_state"].exists():
                paths["cv_state"].unlink(missing_ok=True)
            if config_id not in state["completed"]:
                state["completed"].append(config_id)
            continue
        if not paths["checkpoint"].exists() and not paths["cv_state"].exists():
            state.setdefault("skipped", []).append(config_id)
            continue
        ongoing.append(config_id)
    state["ongoing"] = ongoing


def _select_next(schedule: dict, state: dict) -> list[str]:
    queue = schedule["queue"]
    max_runs = schedule["max_runs"]
    output_dir = schedule["output_dir"]

    if state["ongoing"]:
        return list(state["ongoing"])[:max_runs]

    selected: list[str] = []
    idx = int(state.get("queue_index", 0))
    while len(selected) < max_runs and idx < len(queue):
        config_id = str(queue[idx])
        idx += 1
        try:
            cfg = _load_config(schedule, config_id)
        except Exception as exc:
            print(f"[scheduler] skip {config_id}: {exc}", file=sys.stderr)
            state.setdefault("skipped", []).append(config_id)
            continue
        model_name = str(cfg.get("model_name", "")).strip()
        if not model_name:
            print(f"[scheduler] skip {config_id}: missing model_name", file=sys.stderr)
            state.setdefault("skipped", []).append(config_id)
            continue
        paths = _model_paths(output_dir, model_name)
        if paths["final"].exists():
            if config_id not in state["completed"]:
                state["completed"].append(config_id)
            continue
        selected.append(config_id)
    state["queue_index"] = idx
    state["ongoing"] = selected
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["next"])
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--schedule", default=str(DEFAULT_SCHEDULE))
    args = parser.parse_args()

    schedule_path = Path(args.schedule)
    schedule = _load_schedule(schedule_path)
    if args.max_runs is not None:
        schedule["max_runs"] = int(args.max_runs)
    state = _load_state(schedule["state_path"])
    _cleanup_ongoing(schedule, state)
    selected = _select_next(schedule, state)
    _save_state(schedule["state_path"], state)
    print(json.dumps(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
