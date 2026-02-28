"""PyTorch utilities: mixed precision, compile fallback, seeding."""

from __future__ import annotations

import contextlib
import logging
from typing import Optional

import torch

from common.hardware import FP8ProbeResult, resolve_autocast_dtype


def select_amp_dtype(amp: str, *, enable_fp8: bool = False) -> tuple[torch.dtype, FP8ProbeResult]:
    """Resolve autocast dtype with BF16 default and optional FP8 probing."""
    dtype, fp8 = resolve_autocast_dtype(amp=amp, enable_fp8=enable_fp8)
    return dtype, fp8


@contextlib.contextmanager
def autocast_ctx(*, device_type: str, dtype: torch.dtype, enabled: bool = True):
    if not enabled:
        yield
        return
    with torch.autocast(device_type=device_type, dtype=dtype):
        yield


def safe_compile(
    model: torch.nn.Module,
    *,
    enabled: bool = True,
    backend: str = "inductor",
    logger: Optional[logging.Logger] = None,
    context: str = "",
) -> torch.nn.Module:
    """`torch.compile` with eager fallback and explicit logging."""
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        if logger is not None:
            logger.warning("torch.compile unavailable in this torch build; using eager. context=%s", context)
        return model
    try:
        return torch.compile(model, backend=backend)
    except Exception as exc:
        msg = f"torch.compile failed; falling back to eager. context={context} reason={exc}"
        if logger is not None:
            logger.warning(msg)
        else:
            print(f"[WARN] {msg}")
        return model


def seed_all(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
