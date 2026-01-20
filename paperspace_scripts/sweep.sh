#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="/notebooks/venvs/pt27cu118"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 path/to/script.py [script args...]"
  exit 1
fi

SCRIPT="$1"
shift || true

# Create venv if missing
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Run the python script with any extra args
python3 "$SCRIPT" "$@"
