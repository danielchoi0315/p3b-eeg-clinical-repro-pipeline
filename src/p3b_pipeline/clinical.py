"""Clinical translation utilities: deviation scoring + robust regression.

Important:
- This is **research code**, not a validated clinical tool.
- The goal is to compute individualized deviation scores and relate them to
  clinical severity metrics in a robust way.

We use statsmodels Robust Linear Model (RLM) with HuberT by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def trial_zscore(y: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    var = np.asarray(var, dtype=float)
    sd = np.sqrt(np.maximum(var, 1e-6))
    return (y - mu) / sd


def subject_load_deviation_score(df_trials: pd.DataFrame) -> pd.DataFrame:
    """Compute per-subject deviation scores.

    We return:
      - z_mean: mean z-score across all trials
      - z_hi_minus_lo: mean z(highest load) - mean z(lowest load)  ("load deviation")
    """
    required = {"subject", "memory_load", "z"}
    missing = required - set(df_trials.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out_rows = []
    for sid, g in df_trials.groupby("subject"):
        g = g.dropna(subset=["z", "memory_load"])
        if g.empty:
            continue
        z_mean = float(g["z"].mean())

        lo = float(g["memory_load"].min())
        hi = float(g["memory_load"].max())
        z_lo = float(g.loc[g["memory_load"] == lo, "z"].mean())
        z_hi = float(g.loc[g["memory_load"] == hi, "z"].mean())
        z_hi_minus_lo = float(z_hi - z_lo)

        row = {
            "subject": sid,
            "z_mean": z_mean,
            "z_hi_minus_lo": z_hi_minus_lo,
            "n_trials": int(len(g)),
            "load_lo": lo,
            "load_hi": hi,
        }
        # keep optional covariates if present
        for col in ["age", "sex", "site", "dataset", "severity"]:
            if col in g.columns:
                # subject-level: take first non-null
                vv = g[col].dropna()
                row[col] = vv.iloc[0] if len(vv) else np.nan
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def robust_regress_severity(
    df_subject: pd.DataFrame,
    *,
    score_col: str,
    severity_col: str,
    covariates: Tuple[str, ...] = ("age",),
) -> Optional[Dict[str, Any]]:
    """Robust regression: severity ~ score + covariates.

    Returns a dict with coefficients + SEs, or None if severity missing.
    """
    if severity_col not in df_subject.columns:
        return None
    if score_col not in df_subject.columns:
        raise ValueError(f"score_col not found: {score_col}")

    df = df_subject.copy()
    cols = [severity_col, score_col] + [c for c in covariates if c in df.columns]
    df = df[cols].dropna()
    if len(df) < 10:
        return None

    import statsmodels.api as sm

    y = df[severity_col].astype(float)
    X = df[[score_col] + [c for c in covariates if c in df.columns]].astype(float)
    X = sm.add_constant(X, has_constant="add")

    model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    res = model.fit()

    summary = {
        "n": int(len(df)),
        "params": res.params.to_dict(),
        "bse": res.bse.to_dict(),
        "pvalues": res.pvalues.to_dict() if hasattr(res, "pvalues") else None,
        "scale": float(res.scale),
    }
    return summary
