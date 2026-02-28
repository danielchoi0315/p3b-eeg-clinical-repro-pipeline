"""Logging utilities.

We use Python's standard logging module with:
- console handler (INFO)
- file handler (DEBUG) with timestamps
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


def configure_logging(*, log_dir: Path, run_id: str, name: str = "p3b_pipeline") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}_{run_id}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # avoid duplicate logs if root logger configured

    # Idempotent: clear existing handlers if reconfigured in same process.
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (DEBUG)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.debug("Logging configured. log_path=%s", log_path)
    return logger
