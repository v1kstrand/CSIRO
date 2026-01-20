from __future__ import annotations

import copy
from pathlib import Path

try:
    from ruamel.yaml import YAML

    _YAML = YAML(typ="safe")
except Exception:  # pragma: no cover - fallback for environments without ruamel.yaml
    _YAML = None
    import yaml  # type: ignore


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    if _YAML is not None:
        data = _YAML.load(path.read_text(encoding="utf-8")) or {}
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _YAML is not None:
        with path.open("w", encoding="utf-8") as handle:
            _YAML.dump(data, handle)
        return
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path
