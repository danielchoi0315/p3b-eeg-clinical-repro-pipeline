"""Functional output linkage: P3b margin -> subsequent RT.

We implement the "neural margin" idea from the report:

  m_{i,t} = P3_{i,t} - \hat{P3}_i(load_{i,t})

where \hat{P3}_i(load) is a within-subject reference curve.
Then test whether m_{i,t} predicts RT_{i,t+1} (subsequent RT).

This module is deliberately conservative:
- within-subject modeling avoids cross-subject scaling artifacts
- robust regression protects against RT outliers
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _fit_within_subject_expected_p3(g: pd.DataFrame) -> np.ndarray:
    """Fit \hat{P3} = b0 + b1*load + b2*trial_order (OLS) and return residuals."""
    import statsmodels.api as sm

    # Work on a dense positional index; upstream groupby frames often keep global indices.
    g = g.reset_index(drop=True)
    cols = ["p3b_amp", "memory_load", "trial_order"]
    gg = g[cols].dropna()
    if len(gg) < 20:
        # Too few trials; return NaNs
        return np.full(len(g), np.nan, dtype=float)

    y = gg["p3b_amp"].astype(float)
    X = gg[["memory_load", "trial_order"]].astype(float)
    X = sm.add_constant(X, has_constant="add")
    res = sm.OLS(y, X).fit()
    yhat = res.predict(X)

    # Map back to original indices
    resid = np.full(len(g), np.nan, dtype=float)
    resid[gg.index.to_numpy()] = (y - yhat).to_numpy(dtype=float)
    return resid


def margin_to_next_rt(df_trials: pd.DataFrame) -> Dict[str, Any]:
    """Compute per-subject robust slope of RT_next ~ margin + load, summarize across subjects."""
    required = {"subject", "p3b_amp", "memory_load", "trial_order", "rt"}
    missing = required - set(df_trials.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for sid, g in df_trials.groupby("subject"):
        g = g.sort_values("trial_order").copy()
        g["margin"] = _fit_within_subject_expected_p3(g)
        g["rt_next"] = g["rt"].shift(-1)

        gg = g[["margin", "rt_next", "memory_load"]].dropna()
        if len(gg) < 20:
            continue

        import statsmodels.api as sm

        y = gg["rt_next"].astype(float)
        X = gg[["margin", "memory_load"]].astype(float)
        X = sm.add_constant(X, has_constant="add")
        model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
        res = model.fit()

        rows.append(
            {
                "subject": sid,
                "n": int(len(gg)),
                "beta_margin": float(res.params.get("margin", np.nan)),
                "beta_load": float(res.params.get("memory_load", np.nan)),
                "scale": float(res.scale),
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "n_subjects": int(df["subject"].nunique()) if not df.empty else 0,
        "median_beta_margin": float(df["beta_margin"].median()) if not df.empty else float("nan"),
        "mean_beta_margin": float(df["beta_margin"].mean()) if not df.empty else float("nan"),
        "df_subject": df,
    }
    return summary
