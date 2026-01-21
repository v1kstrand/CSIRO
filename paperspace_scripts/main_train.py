import time

print("Waiting 5s for Jupyter init...")
time.sleep(5)

import os
import subprocess
import sys
from pathlib import Path

SETUPS_DIR = Path("/notebooks/setups")
TERMINAL_LAUNCH = SETUPS_DIR / "terminal_launch.py"

os.chdir(SETUPS_DIR)
subprocess.run([sys.executable, str(TERMINAL_LAUNCH), "init"], check=True)
