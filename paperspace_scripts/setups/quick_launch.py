from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from schedule_utils import deep_merge, load_yaml, resolve_path


DEFAULT_SCHEDULE = Path(__file__).with_name("schedule.yaml")


def _load_schedule(path: Path) -> dict:
    schedule = load_yaml(path)
    base_dir = path.parent
    output_dir = Path(schedule.get("output_dir", "/notebooks/kaggle/csiro/output"))
    experiments_dir = resolve_path(base_dir, schedule.get("experiments_dir", "configs/experiments"))
    ongoing_dir = resolve_path(base_dir, schedule.get("ongoing_dir", "configs/ongoing"))
    base_config = resolve_path(base_dir, schedule.get("base_config", "configs/base.yaml"))
    return dict(
        schedule_path=path,
        output_dir=output_dir,
        experiments_dir=experiments_dir,
        ongoing_dir=ongoing_dir,
        base_config=base_config,
    )


def _resolve_config_path(schedule: dict, config_id: str) -> Path:
    path = Path(config_id)
    if path.is_absolute():
        return path
    for config_dir in (schedule["experiments_dir"], schedule["ongoing_dir"]):
        if config_dir is None:
            continue
        candidate = config_dir / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Config not found: {config_id}")


def _load_config(schedule: dict, config_path: Path) -> dict:
    base_cfg = load_yaml(schedule["base_config"]) if schedule["base_config"] else {}
    override_cfg = load_yaml(config_path)
    if "sweeps" in base_cfg or "sweeps" in override_cfg:
        raise ValueError("sweeps are no longer supported; create separate experiment entries.")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_id")
    parser.add_argument("--schedule", default=str(DEFAULT_SCHEDULE))
    parser.add_argument("--spawn-delay", type=float, default=5.0)
    args = parser.parse_args()

    schedule = _load_schedule(Path(args.schedule))
    output_dir = schedule["output_dir"]

    config_id = args.config_id
    config_path = _resolve_config_path(schedule, config_id)
    config = _load_config(schedule, config_path)
    run_name = _resolve_run_name(config, config_id)
    paths = _model_paths(output_dir, run_name)
    if paths["final"].exists():
        print(f"[quick_launch] {config_id} already completed; skipping.")
        return 0
    ongoing_dir = schedule["ongoing_dir"]
    if ongoing_dir is not None:
        ongoing_dir.mkdir(parents=True, exist_ok=True)
        if config_path.parent != ongoing_dir:
            dest = ongoing_dir / config_path.name
            if dest.exists():
                config_path = dest
            else:
                config_path.replace(dest)
                config_path = dest

    script_dir = Path(__file__).resolve().parent
    terminal_launch = script_dir / "terminal_launch.py"
    subprocess.Popen([sys.executable, str(terminal_launch), "run", str(config_path)])
    time.sleep(float(args.spawn_delay))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
