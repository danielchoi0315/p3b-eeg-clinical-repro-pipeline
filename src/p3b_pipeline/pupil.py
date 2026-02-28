"""Pupillometry preprocessing and event-locked PDR extraction.

We treat pupil dilation response (PDR) as an LC–NE proxy signal and compute
per-trial baseline-corrected response:

PDR_t = mean(pupil[t + response_window]) - mean(pupil[t + baseline_window])

The extraction is designed to be robust to modest sampling irregularities and
missing data; if a window lacks sufficient samples, we return NaN for that trial.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import cfg_get
from .bids_utils import build_layout


def find_eyetrack_file(
    *,
    bids_root: Path,
    subject: str,
    task: Optional[str],
    run: Optional[str],
    session: Optional[str],
) -> Optional[Path]:
    """Locate a pupil/eyetrack TSV for the given recording entities.

    OpenNeuro datasets are not fully consistent here:
    - some use suffix `eyetrack` with a TSV payload
    - ds003838 uses `datatype=pupil` and `*_pupil.tsv`
    We support both patterns and fall back to direct filename matching.
    """
    layout = build_layout(bids_root)

    base_kwargs = {
        "subject": subject,
        "task": task,
        "run": run,
        "session": session,
        "return_type": "filename",
    }

    # Query most-specific patterns first.
    query_specs = [
        {"datatype": "pupil", "suffix": "pupil"},
        {"datatype": "pupil", "suffix": "eyetrack"},
        {"suffix": "pupil"},
        {"suffix": "eyetrack"},
    ]
    candidates: List[Path] = []
    for spec in query_specs:
        for ext in (".tsv.gz", ".tsv"):
            hits = layout.get(extension=ext, **base_kwargs, **spec)
            for h in hits:
                p = Path(h)
                if p.exists():
                    candidates.append(p)
        if candidates:
            break

    # Fallback for datasets where entities are partially unindexed but filenames
    # are still BIDS-like (e.g., sub-*_task-*_pupil.tsv in a `pupil/` folder).
    if not candidates:
        sub_dir = bids_root / f"sub-{subject}"
        if sub_dir.exists():
            raw_hits: List[Path] = []
            raw_hits.extend(sub_dir.rglob("*_pupil.tsv"))
            raw_hits.extend(sub_dir.rglob("*_pupil.tsv.gz"))
            raw_hits.extend(sub_dir.rglob("*_eyetrack.tsv"))
            raw_hits.extend(sub_dir.rglob("*_eyetrack.tsv.gz"))
            for p in raw_hits:
                n = p.name
                if f"sub-{subject}" not in n:
                    continue
                if task and f"_task-{task}" not in n:
                    continue
                if run and f"_run-{run}" not in n:
                    continue
                if session and f"_ses-{session}" not in n:
                    continue
                if p.exists():
                    candidates.append(p)

    if not candidates:
        return None
    candidates = sorted(set(candidates))
    gz = [c for c in candidates if str(c).endswith(".tsv.gz")]
    return gz[0] if gz else candidates[0]


def load_pupil_timeseries(eyetrack_tsv: Path, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Load pupil timeseries (time_s, pupil) from BIDS eyetrack TSV."""
    df = pd.read_csv(eyetrack_tsv, sep="\t")

    time_candidates = cfg_get(
        cfg,
        "pupil.column_candidates.time_s",
        ["time", "timestamp", "t", "pupil_timestamp", "gaze_timestamp"],
    )
    pupil_candidates = cfg_get(
        cfg,
        "pupil.column_candidates.pupil",
        ["pupil", "pupil_size", "pupil_diameter", "diameter", "diameter_3d", "pupil_area"],
    )

    time_col = None
    for c in time_candidates:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError(f"Could not find time column in {eyetrack_tsv}. Tried: {time_candidates}")

    pupil_col = None
    for c in pupil_candidates:
        if c in df.columns:
            pupil_col = c
            break
    if pupil_col is None:
        raise ValueError(f"Could not find pupil column in {eyetrack_tsv}. Tried: {pupil_candidates}")

    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(df[pupil_col], errors="coerce").to_numpy(dtype=float)

    # Drop rows with non-finite time
    m = np.isfinite(t)
    t = t[m]
    p = p[m]

    # Sort by time (some files are not strictly ordered)
    order = np.argsort(t)
    t = t[order]
    p = p[order]

    # Absolute Unix timestamps are common in ds003838 pupil files. Convert to
    # run-relative seconds while preserving local spacing.
    if t.size and np.isfinite(t[0]) and float(t[0]) > 1e6:
        t = t - float(t[0])

    return t, p


def _window_mean(t: np.ndarray, x: np.ndarray, t0: float, w: Tuple[float, float]) -> float:
    """Mean of x within [t0+w0, t0+w1], ignoring NaNs."""
    lo, hi = t0 + float(w[0]), t0 + float(w[1])
    m = (t >= lo) & (t <= hi)
    if m.sum() < 5:  # too few samples (heuristic)
        return float("nan")
    v = x[m]
    v = v[np.isfinite(v)]
    if v.size < 5:
        return float("nan")
    return float(np.mean(v))


def extract_pdr(
    *,
    onset_s: np.ndarray,
    time_s: np.ndarray,
    pupil: np.ndarray,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """Compute per-trial baseline-corrected pupil dilation response (PDR)."""
    baseline = cfg_get(cfg, "pupil.baseline_s", [-0.2, 0.0])
    response = cfg_get(cfg, "pupil.response_s", [0.5, 2.5])
    baseline_w = (float(baseline[0]), float(baseline[1]))
    response_w = (float(response[0]), float(response[1]))

    out = np.full_like(onset_s, np.nan, dtype=np.float32)
    for i, t0 in enumerate(onset_s):
        if not np.isfinite(t0):
            continue
        b = _window_mean(time_s, pupil, float(t0), baseline_w)
        r = _window_mean(time_s, pupil, float(t0), response_w)
        out[i] = np.float32(r - b) if (np.isfinite(b) and np.isfinite(r)) else np.nan
    return out
