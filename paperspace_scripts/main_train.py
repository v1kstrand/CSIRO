from __future__ import annotations

import os
import runpy
from pathlib import Path

SETUPS_DIR = Path("/notebooks/setups")
MAIN_TRAIN = SETUPS_DIR / "main_train.py"

os.chdir(SETUPS_DIR)
runpy.run_path(str(MAIN_TRAIN), run_name="__main__")
