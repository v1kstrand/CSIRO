from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from schedule_utils import deep_merge, dump_yaml, load_yaml, resolve_path


DEFAULT_SCHEDULE = Path(__file__).with_name("schedule.yaml")

def _log(message: str) -> None:
    print(f"[scheduler] {message}", file=sys.stderr)


def _load_schedule(path: Path) -> dict:
    schedule = load_yaml(path)
    base_dir = path.parent
    output_dir = Path(schedule.get("output_dir", "/notebooks/kaggle/csiro/output"))
    state_file = schedule.get("state_file", "scheduler_state.yaml")
    state_path = output_dir / state_file
    experiments_dir = resolve_path(base_dir, schedule.get("experiments_dir", "configs/experiments"))
    base_config = resolve_path(base_dir, schedule.get("base_config", "configs/base.yaml"))
    completed_dir = resolve_path(base_dir, schedule.get("completed_dir", "configs/completed"))
    raw_queue = schedule.get("config_queue", None)
    queue = None if raw_queue is None else list(raw_queue or [])
    max_runs = int(schedule.get("max_runs", 1))
    return dict(
        schedule_path=path,
        output_dir=output_dir,
        state_path=state_path,
        experiments_dir=experiments_dir,
        base_config=base_config,
        completed_dir=completed_dir,
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


def _resolve_run_name(config: dict, config_id: str) -> str:
    run_name = str(config.get("run_name", "")).strip()
    model_name = str(config.get("model_name", "")).strip()
    if not run_name:
        run_name = model_name
    if not run_name:
        raise ValueError(f"run_name is required for {config_id} (or set model_name for legacy configs).")
    return run_name


def _model_paths(output_dir: Path, run_name: str) -> dict:
    states_dir = output_dir / "states"
    return dict(
        checkpoint=states_dir / f"{run_name}_checkpoint.pt",
        cv_state=states_dir / f"{run_name}_cv_state.pt",
        final=(output_dir / "complete" / f"{run_name}.pt"),
    )


def _scan_experiments(experiments_dir: Path) -> list[str]:
    if experiments_dir is None or not experiments_dir.exists():
        return []
    paths = sorted(experiments_dir.glob("*.yaml"))
    paths.extend(sorted(experiments_dir.glob("*.yml")))
    return sorted({p.name for p in paths})


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


def _cleanup_ongoing(schedule: dict, state: dict) -> None:
    output_dir = schedule["output_dir"]
    ongoing = []
    dropped = 0
    for config_id in list(state.get("ongoing", [])):
        try:
            cfg = _load_config(schedule, config_id)
            run_name = _resolve_run_name(cfg, config_id)
        except Exception as exc:
            _log(f"skip {config_id}: {exc}")
            state.setdefault("skipped", []).append(config_id)
            dropped += 1
            continue
        paths = _model_paths(output_dir, run_name)
        if paths["final"].exists():
            if paths["checkpoint"].exists():
                paths["checkpoint"].unlink(missing_ok=True)
            if paths["cv_state"].exists():
                paths["cv_state"].unlink(missing_ok=True)
            if config_id not in state["completed"]:
                state["completed"].append(config_id)
            _move_config_to_completed(schedule, config_id)
            dropped += 1
            continue
        if not paths["checkpoint"].exists() and not paths["cv_state"].exists():
            state.setdefault("skipped", []).append(config_id)
            dropped += 1
            continue
        ongoing.append(config_id)
    state["ongoing"] = ongoing
    if dropped:
        _log(f"pruned {dropped} ongoing entries")


def _select_next(schedule: dict, state: dict) -> list[str]:
    queue = schedule["queue"]
    max_runs = schedule["max_runs"]
    output_dir = schedule["output_dir"]

    if queue is None:
        schedule["queue"] = _scan_experiments(schedule["experiments_dir"])
        state["queue_index"] = 0
        queue = schedule["queue"]
        _log(f"auto-queued {len(queue)} configs from {schedule['experiments_dir']}")

    if state["ongoing"]:
        return list(state["ongoing"])[:max_runs]

    selected: list[str] = []
    idx = int(state.get("queue_index", 0))
    while len(selected) < max_runs and idx < len(queue):
        config_id = str(queue[idx])
        idx += 1
        try:
            cfg = _load_config(schedule, config_id)
            run_name = _resolve_run_name(cfg, config_id)
        except Exception as exc:
            _log(f"skip {config_id}: {exc}")
            state.setdefault("skipped", []).append(config_id)
            continue
        paths = _model_paths(output_dir, run_name)
        if paths["final"].exists():
            if config_id not in state["completed"]:
                state["completed"].append(config_id)
            _move_config_to_completed(schedule, config_id)
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
    queue_len = len(schedule["queue"]) if schedule["queue"] is not None else None
    _log(
        "loaded schedule: "
        f"queue_len={queue_len} "
        f"max_runs={schedule['max_runs']} "
        f"state={schedule['state_path']}"
    )
    _log(
        "state: "
        f"ongoing={len(state['ongoing'])} "
        f"completed={len(state['completed'])} "
        f"skipped={len(state['skipped'])} "
        f"queue_index={state.get('queue_index', 0)}"
    )
    _cleanup_ongoing(schedule, state)
    selected = _select_next(schedule, state)
    _log(f"selected={selected}")
    _save_state(schedule["state_path"], state)
    print(json.dumps(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
