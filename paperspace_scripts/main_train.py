print("Waiting 5s for Jupyter init...")
import time
time.sleep(5)

import json
import os
import subprocess
import sys

SCHEDULER = "scheduler.py"
TRAIN_VIT = False
SPAWN_DELAY_S = 10


def fetch_next_configs() -> list[str]:
    result = subprocess.run(
        [sys.executable, SCHEDULER, "next"],
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = result.stdout.strip()
    if not stdout:
        return []
    configs = json.loads(stdout)
    if not isinstance(configs, list):
        raise ValueError("Scheduler output must be a JSON list.")
    return [str(cfg) for cfg in configs]


def run_many(script: str = "main_init.py", configs: list[str] | None = None, spawn_delay_s: float = 10.0) -> None:
    if not configs:
        print("No configs selected.")
        return
    procs: list[subprocess.Popen] = []
    for config_id in configs:
        p = subprocess.Popen([sys.executable, script, "run", config_id])
        procs.append(p)
        time.sleep(spawn_delay_s)
    _ = [p.wait() for p in procs]


if __name__ == "__main__":
    if TRAIN_VIT:
        print("TRAIN_VIT=True: starting train_vit.py")
        run_many(configs=["vit"], spawn_delay_s=SPAWN_DELAY_S)
    else:
        configs = fetch_next_configs()
        print(f"Starting configs: {configs}")
        run_many(configs=configs, spawn_delay_s=SPAWN_DELAY_S)
