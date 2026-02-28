"""YAML configuration loader with minimal validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping/dict. Got: {type(data)}")
    return data


def cfg_get(cfg: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Fetch a nested key like 'eeg.epoch.tmin_s' with a default."""
    cur: Any = cfg
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
