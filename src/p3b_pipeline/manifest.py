"""Run manifest emission for strict reproducibility.

Each top-level script writes a `run_manifest.json` capturing:
- CLI args
- software versions
- CPU/GPU info
- key environment variables
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .env import env_snapshot, cuda_device_summary


def _safe_git_commit() -> Optional[str]:
    """Best-effort git commit hash (None if not in a git repo)."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def write_manifest(
    *,
    out_dir: Path,
    run_id: str,
    entrypoint: str,
    args: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"run_manifest_{entrypoint}_{run_id}.json"

    payload: Dict[str, Any] = {
        "run_id": run_id,
        "entrypoint": entrypoint,
        "args": args,
        "env": env_snapshot(),
        "git_commit": _safe_git_commit(),
        "python": {
            "version": platform.python_version(),
            "executable": os.environ.get("PYTHON_EXECUTABLE") or os.sys.executable,
        },
        "cuda": cuda_device_summary(),
    }
    if extra:
        payload["extra"] = extra

    # Atomic write
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(manifest_path)
    return manifest_path
