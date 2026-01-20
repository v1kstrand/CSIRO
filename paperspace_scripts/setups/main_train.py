print("Waiting 5s for Jupyter init...")
import time
time.sleep(5)

import json
import os
import subprocess
import sys

SETUPS_DIR = "/notebooks/setups"
SCHEDULER = os.path.join(SETUPS_DIR, "scheduler.py")
TRAIN_VIT = False
SPAWN_DELAY_S = 10
POLL_INTERVAL_S = 30
SPECIAL_MODES = {"vit", "backup"}
NO_WAIT_MODES = {"backup"}


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


def spawn_init(config_id: str, script: str = "main_init.py") -> subprocess.Popen:
    init_path = script
    if not os.path.isabs(script):
        init_path = os.path.join(SETUPS_DIR, script)
    if config_id in SPECIAL_MODES:
        return subprocess.Popen([sys.executable, init_path, config_id])
    return subprocess.Popen([sys.executable, init_path, "run", config_id])


def stop_process(proc: subprocess.Popen, label: str, timeout_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    print(f"Stopping {label}...")
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()


if __name__ == "__main__":
    backup_proc = spawn_init("backup")
    active: dict[str, subprocess.Popen] = {}
    vit_proc: subprocess.Popen | None = None

    if TRAIN_VIT:
        print("TRAIN_VIT=True: starting train_vit.py")
        vit_proc = spawn_init("vit")
        time.sleep(SPAWN_DELAY_S)

    while True:
        for config_id, proc in list(active.items()):
            if proc.poll() is not None:
                print(f"Run finished: {config_id} (exit={proc.returncode})")
                active.pop(config_id, None)

        configs = fetch_next_configs()
        if configs:
            if vit_proc is not None:
                stop_process(vit_proc, "train_vit.py")
                vit_proc = None
            for config_id in configs:
                if config_id in active:
                    continue
                print(f"Starting config: {config_id}")
                active[config_id] = spawn_init(config_id)
                time.sleep(SPAWN_DELAY_S)
        else:
            if vit_proc is None or vit_proc.poll() is not None:
                print("No configs available; starting train_vit.py to keep GPU warm.")
                vit_proc = spawn_init("vit")
                time.sleep(SPAWN_DELAY_S)

        if backup_proc.poll() is not None:
            print("gpu_backup.sh exited; restarting.")
            backup_proc = spawn_init("backup")

        time.sleep(POLL_INTERVAL_S)
