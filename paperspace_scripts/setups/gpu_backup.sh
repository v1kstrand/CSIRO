#!/usr/bin/env bash
set -euo pipefail

SETUPS_DIR="/notebooks/setups"
VENV_DIR="/notebooks/venvs/pt27cu118"

BACKUP_TIME_MIN="${BACKUP_TIME_MIN:-15}"
CHECK_INTERVAL_S="${BACKUP_CHECK_INTERVAL_S:-60}"
GPU_IDLE_MB="${GPU_IDLE_MB:-200}"

export PYTHONPATH="/notebooks/CSIRO:${PYTHONPATH:-}"

if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

idle_secs=0
idle_target=$((BACKUP_TIME_MIN * 60))

echo "[gpu_backup] Watching GPU memory <= ${GPU_IDLE_MB} MB for ${BACKUP_TIME_MIN} min."

while true; do
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu_backup] nvidia-smi not found; sleeping."
    sleep "$CHECK_INTERVAL_S"
    continue
  fi

  mem_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [ -z "$mem_used" ]; then
    echo "[gpu_backup] Unable to read GPU memory; sleeping."
    sleep "$CHECK_INTERVAL_S"
    continue
  fi

  if [ "$mem_used" -le "$GPU_IDLE_MB" ]; then
    idle_secs=$((idle_secs + CHECK_INTERVAL_S))
  else
    idle_secs=0
  fi

  if [ "$idle_secs" -ge "$idle_target" ]; then
    echo "[gpu_backup] GPU idle detected; launching train_vit.py."
    python "$SETUPS_DIR/train_vit.py" || echo "[gpu_backup] train_vit.py failed; continuing monitor."
    idle_secs=0
  fi

  sleep "$CHECK_INTERVAL_S"
done
