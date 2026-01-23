from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from schedule_utils import deep_merge, load_yaml, resolve_path


DEFAULT_SCHEDULE = Path(__file__).with_name("schedule.yaml")
EXP_NAME_RE = re.compile(r"^exp(\d+)")


def _log(message: str) -> None:
    print(f"[scheduler] {message}", file=sys.stderr)


def _load_schedule(path: Path) -> dict:
    schedule = load_yaml(path)
    base_dir = path.parent
    output_dir = Path(schedule.get("output_dir", "/notebooks/kaggle/csiro/output"))
    experiments_dir = resolve_path(base_dir, schedule.get("experiments_dir", "configs/experiments"))
    ongoing_dir = resolve_path(base_dir, schedule.get("ongoing_dir", "configs/ongoing"))
    completed_dir = resolve_path(base_dir, schedule.get("completed_dir", "configs/completed"))
    skipped_dir = resolve_path(base_dir, schedule.get("skipped_dir", "configs/skipped"))
    base_config = resolve_path(base_dir, schedule.get("base_config", "configs/base.yaml"))
    max_runs = int(schedule.get("max_runs", 1))
    return dict(
        schedule_path=path,
        output_dir=output_dir,
        experiments_dir=experiments_dir,
        ongoing_dir=ongoing_dir,
        completed_dir=completed_dir,
        skipped_dir=skipped_dir,
        base_config=base_config,
        max_runs=max_runs,
    )


def _parse_exp_index(name: str) -> tuple[int | None, str | None]:
    stem = Path(name).stem
    if not stem.startswith("exp"):
        return None, "name must start with 'exp'"
    match = EXP_NAME_RE.match(stem)
    if not match:
        return None, "missing experiment index"
    idx = int(match.group(1))
    suffix = stem[len(match.group(0)) :]
    if suffix and not suffix.startswith("_"):
        return None, "suffix must start with '_'"
    return idx, None


def _scan_configs(config_dir: Path | None) -> list[str]:
    if config_dir is None or not config_dir.exists():
        return []
    paths = sorted(config_dir.glob("*.yaml"))
    paths.extend(sorted(config_dir.glob("*.yml")))
    items: list[tuple[int, str]] = []
    for path in paths:
        idx, err = _parse_exp_index(path.name)
        if idx is None:
            _log(f"skip {path.name}: {err}")
            continue
        items.append((idx, path.name))
    items.sort(key=lambda item: (item[0], item[1]))
    return [name for _, name in items]


def _load_config(base_config: Path | None, override_path: Path) -> dict:
    base_cfg = load_yaml(base_config) if base_config else {}
    override_cfg = load_yaml(override_path)
    return deep_merge(base_cfg, override_cfg)


def _resolve_run_name_with_comet(
    base_config: Path | None,
    override_path: Path,
    *,
    config_id: str,
) -> str:
    override_cfg = load_yaml(override_path)
    base_cfg = load_yaml(base_config) if base_config else {}
    run_name = str(override_cfg.get("run_name", "")).strip() or str(base_cfg.get("run_name", "")).strip()
    exp_name = str(override_cfg.get("comet_exp_name", "")).strip() or str(base_cfg.get("comet_exp_name", "")).strip()
    if not run_name or not exp_name:
        raise ValueError(f"run_name and comet_exp_name required for {config_id}")
    return f"{exp_name}_{run_name}"


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


def _move_to_completed(config_path: Path, completed_dir: Path) -> None:
    if not config_path.exists():
        return
    completed_dir.mkdir(parents=True, exist_ok=True)
    dest = completed_dir / config_path.name
    if dest.exists():
        idx = 1
        while True:
            candidate = completed_dir / f"dup{idx}_{config_path.name}"
            if not candidate.exists():
                dest = candidate
                break
            idx += 1
    config_path.replace(dest)


def _move_to_skipped(config_path: Path, skipped_dir: Path) -> None:
    if not config_path.exists():
        return
    skipped_dir.mkdir(parents=True, exist_ok=True)
    dest = skipped_dir / config_path.name
    if dest.exists():
        idx = 1
        while True:
            candidate = skipped_dir / f"dup{idx}_{config_path.name}"
            if not candidate.exists():
                dest = candidate
                break
            idx += 1
    config_path.replace(dest)


def _cleanup_ongoing(schedule: dict) -> list[str]:
    output_dir = schedule["output_dir"]
    ongoing_dir = schedule["ongoing_dir"]
    completed_dir = schedule["completed_dir"]
    skipped_dir = schedule["skipped_dir"]
    ongoing = []
    for config_name in _scan_configs(ongoing_dir):
        config_path = ongoing_dir / config_name
        try:
            run_name = _resolve_run_name_with_comet(
                schedule["base_config"],
                config_path,
                config_id=config_name,
            )
        except Exception as exc:
            _log(f"skip {config_name}: {exc}")
            _move_to_skipped(config_path, skipped_dir)
            continue
        paths = _model_paths(output_dir, run_name)
        if paths["final"].exists():
            if paths["checkpoint"].exists():
                paths["checkpoint"].unlink(missing_ok=True)
            if paths["cv_state"].exists():
                paths["cv_state"].unlink(missing_ok=True)
            _move_to_completed(config_path, completed_dir)
            continue
        ongoing.append(config_name)
    return ongoing


def _select_next(schedule: dict) -> list[str]:
    output_dir = schedule["output_dir"]
    experiments_dir = schedule["experiments_dir"]
    ongoing_dir = schedule["ongoing_dir"]
    completed_dir = schedule["completed_dir"]
    skipped_dir = schedule["skipped_dir"]
    max_runs = schedule["max_runs"]

    if ongoing_dir is not None:
        ongoing_dir.mkdir(parents=True, exist_ok=True)

    ongoing = _cleanup_ongoing(schedule)
    selected: list[str] = list(ongoing)[:max_runs]

    if len(selected) >= max_runs:
        return selected

    for config_name in _scan_configs(experiments_dir):
        if config_name in ongoing:
            continue
        config_path = experiments_dir / config_name
        try:
            run_name = _resolve_run_name_with_comet(
                schedule["base_config"],
                config_path,
                config_id=config_name,
            )
        except Exception as exc:
            _log(f"skip {config_name}: {exc}")
            _move_to_skipped(config_path, skipped_dir)
            continue
        paths = _model_paths(output_dir, run_name)
        if paths["final"].exists():
            _move_to_completed(config_path, completed_dir)
            continue
        dest = ongoing_dir / config_name
        if dest.exists():
            continue
        config_path.replace(dest)
        selected.append(config_name)
        if len(selected) >= max_runs:
            break

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

    _log(
        "loaded schedule: "
        f"max_runs={schedule['max_runs']} "
        f"experiments={schedule['experiments_dir']} "
        f"ongoing={schedule['ongoing_dir']} "
        f"completed={schedule['completed_dir']}"
    )
    selected = _select_next(schedule)
    _log(f"selected={selected}")
    print(json.dumps(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
