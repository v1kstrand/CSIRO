from __future__ import annotations

import argparse
import subprocess
import sys
import time
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
    return dict(
        schedule_path=path,
        output_dir=output_dir,
        state_path=state_path,
        experiments_dir=experiments_dir,
        base_config=base_config,
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
    return deep_merge(base_cfg, override_cfg)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_ids", nargs="+")
    parser.add_argument("--schedule", default=str(DEFAULT_SCHEDULE))
    parser.add_argument("--spawn-delay", type=float, default=5.0)
    args = parser.parse_args()

    schedule = _load_schedule(Path(args.schedule))
    state = _load_state(schedule["state_path"])
    output_dir = schedule["output_dir"]

    to_launch: list[str] = []
    for config_id in args.config_ids:
        config = _load_config(schedule, config_id)
        run_name = _resolve_run_name(config, config_id)
        paths = _model_paths(output_dir, run_name)
        if paths["final"].exists():
            print(f"[quick_launch] {config_id} already completed; skipping.")
            continue
        if config_id not in state["ongoing"]:
            state["ongoing"].append(config_id)
        if config_id not in to_launch:
            to_launch.append(config_id)

    _save_state(schedule["state_path"], state)

    script_dir = Path(__file__).resolve().parent
    main_init = script_dir / "main_init.py"
    for config_id in to_launch:
        subprocess.Popen([sys.executable, str(main_init), "run", config_id])
        time.sleep(float(args.spawn_delay))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
