#!/usr/bin/env python3
"""Law-C audit for ds004796 Sternberg features (locked endpoint)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from common.lawc_audit import (
    collect_lawc_trials_from_features,
    permutation_test_median_rho,
    subject_level_rhos,
)


def _shuffle_within_subject(df: pd.DataFrame, *, col: str, seed: int) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(int(seed))
    for _, idx in out.groupby("subject_key").groups.items():
        idx_list = list(idx)
        vals = out.loc[idx_list, col].to_numpy(copy=True)
        out.loc[idx_list, col] = vals[rng.permutation(len(vals))]
    return out


def _control_shuffle_summary(
    df: pd.DataFrame,
    *,
    col: str,
    n_shuffles: int,
    seed: int,
    min_trials: int,
) -> Dict[str, Any]:
    medians: List[float] = []
    for rep in range(max(1, int(n_shuffles))):
        s = _shuffle_within_subject(df, col=col, seed=int(seed) + rep * 7919)
        subj = subject_level_rhos(s, min_trials=min_trials)
        medians.append(float(subj.get("median_rho", np.nan)))
    arr = np.asarray(medians, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median_rho": float("nan"), "p_nonnegative": float("nan"), "n_shuffles": 0}
    med = float(np.median(arr))
    p_nonneg = float((1.0 + np.sum(arr >= 0.0)) / (1.0 + arr.size))
    return {"median_rho": med, "p_nonnegative": p_nonneg, "n_shuffles": int(arr.size)}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--dataset_id", type=str, default="ds004796")
    ap.add_argument("--n_perm", type=int, default=50000)
    ap.add_argument("--min_trials", type=int, default=20)
    ap.add_argument("--n_control_shuffles", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_lawc_trials_from_features(args.features_root, args.dataset_id)
    if df.empty:
        raise RuntimeError(f"no extracted trials found for dataset {args.dataset_id}")
    df = df[np.isfinite(df["p3b_amp"]) & np.isfinite(df["memory_load"])].copy()
    if df.empty:
        raise RuntimeError(f"no finite p3b/load rows for dataset {args.dataset_id}")

    subj = subject_level_rhos(df, min_trials=int(args.min_trials))
    perm = permutation_test_median_rho(
        df,
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        mode="y_shuffle",
        min_trials=int(args.min_trials),
    )

    x_ctrl = _control_shuffle_summary(
        df,
        col="p3b_amp",
        n_shuffles=int(args.n_control_shuffles),
        seed=int(args.seed) + 17,
        min_trials=int(args.min_trials),
    )
    y_ctrl = _control_shuffle_summary(
        df,
        col="memory_load",
        n_shuffles=int(args.n_control_shuffles),
        seed=int(args.seed) + 29,
        min_trials=int(args.min_trials),
    )

    result = {
        "dataset_id": args.dataset_id,
        "n_trials": int(len(df)),
        "n_subjects": int(df["subject_key"].nunique()),
        "median_rho": float(subj.get("median_rho", np.nan)),
        "n_subjects_used": int(subj.get("n_subjects_used", 0)),
        "p_value_perm": float(perm.get("p_value", np.nan)),
        "n_perm": int(args.n_perm),
        "min_trials": int(args.min_trials),
        "fallback_non_pz_trials": int((df["p3b_channel"].astype(str).str.upper() != "PZ").sum()),
        "rt_nonmissing_rate": float(np.isfinite(df["rt"]).mean()) if "rt" in df.columns else float("nan"),
        "control_x_shuffle": x_ctrl,
        "control_y_shuffle": y_ctrl,
        "x_degrade_pass": bool((x_ctrl["median_rho"] <= 0.0) or (x_ctrl["p_nonnegative"] > 0.2)),
        "y_degrade_pass": bool((y_ctrl["median_rho"] <= 0.0) or (y_ctrl["p_nonnegative"] > 0.2)),
    }

    locked_json = out_dir / "locked_test_results.json"
    locked_csv = out_dir / "locked_test_results.csv"
    neg_csv = out_dir / "negative_controls.csv"

    locked_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame([result]).to_csv(locked_csv, index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": args.dataset_id,
                "control": "shuffle_p3_within_subject",
                "median_rho": x_ctrl["median_rho"],
                "p_nonnegative": x_ctrl["p_nonnegative"],
                "n_shuffles": x_ctrl["n_shuffles"],
                "degrade_pass": result["x_degrade_pass"],
            },
            {
                "dataset_id": args.dataset_id,
                "control": "shuffle_load_within_subject",
                "median_rho": y_ctrl["median_rho"],
                "p_nonnegative": y_ctrl["p_nonnegative"],
                "n_shuffles": y_ctrl["n_shuffles"],
                "degrade_pass": result["y_degrade_pass"],
            },
        ]
    ).to_csv(neg_csv, index=False)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
