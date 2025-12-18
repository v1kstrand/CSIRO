from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_IMPL_SRC = _REPO_ROOT / "csiro" / "src"
sys.path.insert(0, str(_IMPL_SRC))

# Convenience re-exports so that after `import csiro`, you can do:
#   csiro.data, csiro.losses, ...
from . import amp, config, data, losses, metrics, model, train, transforms, viz  # noqa: E402,F401

try:
    import csiro_biomass as csiro_biomass  # noqa: E402,F401
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import the underlying implementation package `csiro_biomass`.\n"
        "If you cloned the repo to `/notebooks/CSIRO`, ensure you have:\n"
        "  - `/notebooks/CSIRO/csiro/src/csiro_biomass/`\n"
        "or install the package with:\n"
        "  pip install -e /notebooks/CSIRO"
    ) from e

__all__ = [
    "amp",
    "config",
    "data",
    "losses",
    "metrics",
    "model",
    "train",
    "transforms",
    "viz",
    "csiro_biomass",
]

