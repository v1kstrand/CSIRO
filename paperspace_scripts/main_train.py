from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SETUPS_DIR = Path("/notebooks/setups")
TERMINAL_LAUNCH = SETUPS_DIR / "terminal_launch.py"

os.chdir(SETUPS_DIR)
subprocess.run([sys.executable, str(TERMINAL_LAUNCH), "init"], check=True)
