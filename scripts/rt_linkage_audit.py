#!/usr/bin/env python3
"""Reviewer-facing RT-linkage inference battery."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from common.lawc_audit import bh_fdr  # noqa: E402
from p3b_pipeline.h5io import iter_subject_feature_files, read_subject_h5  # noqa: E402
from p3b_pipeline.rt_linkage import _fit_within_subject_expected_p3  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", type=Path, required=True)
    ap.add_argument("--datasets", type=str, default="ds005095,ds003655,ds004117")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_perm", type=int, default=20000)
    ap.add_argument("--min_trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=0, help="0=auto (<= #datasets)")
    ap.add_argument("--expect_negative", action="store_true", default=True)
    return ap.parse_args()


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _rank(a: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(a, dtype=float)).rank(method="average").to_numpy(dtype=float)


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(x), dtype=float), x.astype(float)])
    try:
        beta = np.linalg.lstsq(X, y.astype(float), rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.full(len(y), np.nan, dtype=float)
    return y - X @ beta


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 1e-12 or sb <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _collect_dataset_trials(features_root: Path, dataset_id: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for fp in iter_subject_feature_files(features_root):
        arrays, attrs = read_subject_h5(fp)
        ds = _safe_str(attrs.get("dataset_id")).strip()
        if ds != dataset_id:
            continue
        if not {"p3b_amp", "memory_load", "trial_order", "rt"}.issubset(arrays.keys()):
            continue
        p3 = np.asarray(arrays["p3b_amp"], dtype=float)
        load = np.asarray(arrays["memory_load"], dtype=float)
        order = np.asarray(arrays["trial_order"], dtype=float)
        rt = np.asarray(arrays["rt"], dtype=float)
        if len(p3) == 0:
            continue

        subject_key = _safe_str(attrs.get("subject_key")).strip()
        if not subject_key:
            sub = _safe_str(attrs.get("bids_subject") or attrs.get("subject")).strip()
            ses = _safe_str(attrs.get("bids_session") or attrs.get("session")).strip()
            if ses and ses.lower() not in {"na", "none"}:
                subject_key = f"{dataset_id}:sub-{sub}:ses-{ses}"
            else:
                subject_key = f"{dataset_id}:sub-{sub}"

        rows.append(
            pd.DataFrame(
                {
                    "dataset_id": dataset_id,
                    "subject_key": subject_key,
                    "p3b_amp": p3,
                    "memory_load": load,
                    "trial_order": order,
                    "rt": rt,
                }
            )
        )

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, axis=0, ignore_index=True)


def _subject_structs(df: pd.DataFrame, min_trials: int) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    structs: List[Dict[str, Any]] = []
    subj_rows: List[Dict[str, Any]] = []

    for sid, g in df.groupby("subject_key"):
        g = g.sort_values("trial_order").copy()
        g["margin"] = _fit_within_subject_expected_p3(g[["p3b_amp", "memory_load", "trial_order"]])
        g["rt_next"] = g["rt"].shift(-1)
        gg = g[["margin", "rt_next", "memory_load"]].dropna()
        if len(gg) < int(min_trials):
            continue

        margin = gg["margin"].to_numpy(dtype=float)
        rt_next = gg["rt_next"].to_numpy(dtype=float)
        load = gg["memory_load"].to_numpy(dtype=float)
        if np.std(margin) <= 1e-12 or np.std(rt_next) <= 1e-12 or np.std(load) <= 1e-12:
            continue

        m_rank = _rank(margin)
        r_rank = _rank(rt_next)
        l_rank = _rank(load)

        m_res = _residualize(m_rank, l_rank)
        r_res = _residualize(r_rank, l_rank)
        effect = _safe_corr(m_res, r_res)
        if not np.isfinite(effect):
            continue

        subj_rows.append({"subject_key": str(sid), "n_trials": int(len(gg)), "effect_partial_spearman": float(effect)})
        structs.append(
            {
                "subject_key": str(sid),
                "m_rank": m_rank,
                "r_rank": r_rank,
                "l_rank": l_rank,
                "m_res": m_res,
                "r_res": r_res,
                "obs_effect": float(effect),
            }
        )

    return structs, pd.DataFrame(subj_rows)


def _perm_distribution(
    structs: List[Dict[str, Any]],
    *,
    n_perm: int,
    seed: int,
    mode: str,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    null = np.full(int(n_perm), np.nan, dtype=float)
    if not structs:
        return null

    for i in range(int(n_perm)):
        vals: List[float] = []
        for s in structs:
            if mode == "shuffle_margin":
                m_perm = s["m_rank"][rng.permutation(len(s["m_rank"]))]
                m_res = _residualize(m_perm, s["l_rank"])
                val = _safe_corr(m_res, s["r_res"])
            elif mode == "shuffle_rt":
                r_perm = s["r_rank"][rng.permutation(len(s["r_rank"]))]
                r_res = _residualize(r_perm, s["l_rank"])
                val = _safe_corr(s["m_res"], r_res)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            if np.isfinite(val):
                vals.append(float(val))
        if vals:
            null[i] = float(np.median(np.asarray(vals, dtype=float)))
    return null


def _dataset_analysis(
    *,
    dataset_id: str,
    features_root: Path,
    n_perm: int,
    seed: int,
    min_trials: int,
    expect_negative: bool,
) -> Dict[str, Any]:
    df = _collect_dataset_trials(features_root, dataset_id)
    if df.empty:
        return {
            "dataset_id": dataset_id,
            "status": "SKIP",
            "reason": "no trials found in features cache",
            "n_trials": 0,
            "n_subjects": 0,
            "rt_nonmissing_rate": 0.0,
            "subject_effects": [],
            "subject_table": [],
        }

    rt_nonmissing = float(np.isfinite(df["rt"]).mean())
    if rt_nonmissing < 0.5:
        return {
            "dataset_id": dataset_id,
            "status": "SKIP",
            "reason": f"rt_nonmissing_rate={rt_nonmissing:.3f} < 0.5",
            "n_trials": int(len(df)),
            "n_subjects": int(df["subject_key"].nunique()),
            "rt_nonmissing_rate": rt_nonmissing,
            "subject_effects": [],
            "subject_table": [],
        }

    structs, subj_df = _subject_structs(df, min_trials=min_trials)
    if not structs:
        return {
            "dataset_id": dataset_id,
            "status": "SKIP",
            "reason": f"no subjects with >= {min_trials} valid probe trials",
            "n_trials": int(len(df)),
            "n_subjects": int(df["subject_key"].nunique()),
            "rt_nonmissing_rate": rt_nonmissing,
            "subject_effects": [],
            "subject_table": [],
        }

    observed = float(np.median(np.asarray([s["obs_effect"] for s in structs], dtype=float)))

    null_primary = _perm_distribution(structs, n_perm=n_perm, seed=seed + abs(hash(dataset_id)) % 100_000, mode="shuffle_margin")
    finite_primary = null_primary[np.isfinite(null_primary)]
    if finite_primary.size > 0:
        if expect_negative:
            p_primary = float((1.0 + np.sum(finite_primary <= observed)) / (1.0 + finite_primary.size))
        else:
            p_primary = float((1.0 + np.sum(finite_primary >= observed)) / (1.0 + finite_primary.size))
    else:
        p_primary = float("nan")

    null_control = _perm_distribution(structs, n_perm=n_perm, seed=seed + 13 + abs(hash(dataset_id)) % 100_000, mode="shuffle_rt")
    finite_control = null_control[np.isfinite(null_control)]
    if finite_control.size > 0:
        p_control_vs_zero = float((1.0 + np.sum(finite_control <= 0.0)) / (1.0 + finite_control.size))
        control_median = float(np.median(finite_control))
    else:
        p_control_vs_zero = float("nan")
        control_median = float("nan")

    return {
        "dataset_id": dataset_id,
        "status": "PASS",
        "reason": "",
        "n_trials": int(len(df)),
        "n_subjects": int(df["subject_key"].nunique()),
        "n_subjects_used": int(len(structs)),
        "rt_nonmissing_rate": rt_nonmissing,
        "effect_metric": "partial_spearman(rt_next, margin | load)",
        "observed_median_effect": observed,
        "p_value": p_primary,
        "control_shuffle_rt_median_effect": control_median,
        "control_shuffle_rt_p_vs_zero": p_control_vs_zero,
        "subject_effects": [float(s["obs_effect"]) for s in structs],
        "subject_table": subj_df.to_dict(orient="records"),
        "null_primary": finite_primary.tolist(),
        "null_control_rt": finite_control.tolist(),
    }


def _null_summary(dataset_id: str, name: str, vals: np.ndarray) -> Dict[str, Any]:
    vv = np.asarray(vals, dtype=float)
    vv = vv[np.isfinite(vv)]
    if vv.size == 0:
        return {
            "dataset_id": dataset_id,
            "null_type": name,
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "q025": float("nan"),
            "q975": float("nan"),
        }
    return {
        "dataset_id": dataset_id,
        "null_type": name,
        "n": int(vv.size),
        "mean": float(np.mean(vv)),
        "median": float(np.median(vv)),
        "std": float(np.std(vv)),
        "q025": float(np.quantile(vv, 0.025)),
        "q975": float(np.quantile(vv, 0.975)),
    }


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samp = arr[rng.integers(0, arr.size, size=arr.size)]
        stats[i] = float(np.median(samp))
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def _plot_subject_distributions(results_df: pd.DataFrame, subj_effects: Dict[str, np.ndarray], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    order = [x for x in results_df["dataset_id"].tolist() if x != "pooled"]
    data = [subj_effects.get(ds, np.asarray([], dtype=float)) for ds in order]
    if any(len(x) > 0 for x in data):
        ax.boxplot(data, labels=order, showmeans=True)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_ylabel("Subject-level partial Spearman effect")
        ax.set_title("RT Linkage Subject Effect Distributions")
    else:
        ax.text(0.5, 0.5, "No dataset PASS rows", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_null_overlay(results: List[Dict[str, Any]], out_png: Path) -> None:
    pass_rows = [r for r in results if r.get("status") == "PASS" and r.get("dataset_id") != "pooled"]
    n = len(pass_rows)
    if n == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No PASS datasets", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        return

    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, row in zip(axes, pass_rows):
        primary = np.asarray(row.get("null_primary", []), dtype=float)
        control = np.asarray(row.get("null_control_rt", []), dtype=float)
        obs = float(row.get("observed_median_effect", np.nan))

        if primary.size > 0:
            ax.hist(primary, bins=60, alpha=0.5, label="Null: shuffle margin", density=True)
        if control.size > 0:
            ax.hist(control, bins=60, alpha=0.5, label="Control null: shuffle RT", density=True)
        if np.isfinite(obs):
            ax.axvline(obs, color="black", linestyle="--", linewidth=1.0, label="Observed median")
        ax.set_title(str(row["dataset_id"]))
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if int(args.n_perm) < 1000:
        raise ValueError("n_perm must be >= 1000 for reviewer-proof inference")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets = _split_csv(args.datasets)
    workers = int(args.workers) if int(args.workers) > 0 else min(len(datasets), max(1, os.cpu_count() or 1))
    workers = max(1, min(workers, len(datasets)))

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                _dataset_analysis,
                dataset_id=ds,
                features_root=args.features_root,
                n_perm=int(args.n_perm),
                seed=int(args.seed),
                min_trials=int(args.min_trials),
                expect_negative=bool(args.expect_negative),
            ): ds
            for ds in datasets
        }
        for fut in as_completed(futs):
            row = fut.result()
            print(
                f"[RT-linkage] {row['dataset_id']} status={row['status']} "
                f"median={row.get('observed_median_effect', float('nan')):.6g} p={row.get('p_value', float('nan')):.6g}",
                flush=True,
            )
            results.append(row)

    results = sorted(results, key=lambda r: str(r.get("dataset_id", "")))
    pass_rows = [r for r in results if r.get("status") == "PASS"]
    pvals = [float(r.get("p_value", np.nan)) for r in pass_rows]
    qvals = bh_fdr(pvals) if pvals else []
    for r, q in zip(pass_rows, qvals):
        r["q_value"] = float(q)
    for r in results:
        if "q_value" not in r:
            r["q_value"] = float("nan")

    all_subject_effects: List[float] = []
    subj_effects_by_dataset: Dict[str, np.ndarray] = {}
    for r in pass_rows:
        vals = np.asarray(r.get("subject_effects", []), dtype=float)
        vals = vals[np.isfinite(vals)]
        subj_effects_by_dataset[str(r["dataset_id"])] = vals
        if vals.size > 0:
            all_subject_effects.extend(vals.tolist())
        real = float(r.get("observed_median_effect", np.nan))
        ctrl = float(r.get("control_shuffle_rt_median_effect", np.nan))
        pctrl = float(r.get("control_shuffle_rt_p_vs_zero", np.nan))
        r["control_weaker_than_primary"] = bool((not np.isfinite(real)) or (abs(ctrl) <= abs(real) and (not np.isfinite(pctrl) or pctrl > 0.2)))

    pooled_vals = np.asarray(all_subject_effects, dtype=float)
    pooled_vals = pooled_vals[np.isfinite(pooled_vals)]
    pooled_q025, pooled_q975 = _bootstrap_ci(pooled_vals, n_boot=2000, seed=int(args.seed) + 1000)
    pooled = {
        "dataset_id": "pooled",
        "status": "PASS" if pooled_vals.size > 0 else "SKIP",
        "reason": "" if pooled_vals.size > 0 else "no PASS datasets",
        "n_trials": int(sum(int(r.get("n_trials", 0)) for r in pass_rows)),
        "n_subjects": int(pooled_vals.size),
        "n_subjects_used": int(pooled_vals.size),
        "rt_nonmissing_rate": float(np.nanmean([float(r.get("rt_nonmissing_rate", np.nan)) for r in pass_rows])) if pass_rows else float("nan"),
        "effect_metric": "partial_spearman(rt_next, margin | load)",
        "observed_median_effect": float(np.median(pooled_vals)) if pooled_vals.size > 0 else float("nan"),
        "pooled_mean_effect": float(np.mean(pooled_vals)) if pooled_vals.size > 0 else float("nan"),
        "pooled_bootstrap_ci95_low": pooled_q025,
        "pooled_bootstrap_ci95_high": pooled_q975,
        "p_value": float("nan"),
        "q_value": float("nan"),
        "control_shuffle_rt_median_effect": float("nan"),
        "control_shuffle_rt_p_vs_zero": float("nan"),
        "control_weaker_than_primary": True,
        "subject_effects": pooled_vals.tolist(),
        "subject_table": [],
        "null_primary": [],
        "null_control_rt": [],
    }
    results.append(pooled)

    results_rows = []
    null_rows = []
    for r in results:
        results_rows.append({k: v for k, v in r.items() if k not in {"subject_effects", "subject_table", "null_primary", "null_control_rt"}})
        if r.get("dataset_id") == "pooled":
            continue
        null_rows.append(_null_summary(str(r["dataset_id"]), "primary_shuffle_margin", np.asarray(r.get("null_primary", []), dtype=float)))
        null_rows.append(_null_summary(str(r["dataset_id"]), "control_shuffle_rt", np.asarray(r.get("null_control_rt", []), dtype=float)))

    results_df = pd.DataFrame(results_rows)
    nulls_df = pd.DataFrame(null_rows)

    out_results = args.out_dir / "rt_linkage_results.csv"
    out_nulls = args.out_dir / "rt_linkage_nulls.csv"
    out_json = args.out_dir / "rt_linkage_results.json"
    fig_dist = args.out_dir / "FIG_rt_linkage_distributions.png"
    fig_null = args.out_dir / "FIG_rt_linkage_null_overlay.png"
    subj_json = args.out_dir / "rt_linkage_subject_tables.json"

    results_df.to_csv(out_results, index=False)
    nulls_df.to_csv(out_nulls, index=False)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    subj_json.write_text(
        json.dumps(
            {str(r["dataset_id"]): r.get("subject_table", []) for r in results if r.get("dataset_id") != "pooled"},
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_subject_distributions(results_df, subj_effects_by_dataset, fig_dist)
    _plot_null_overlay(results, fig_null)

    print(f"Wrote {out_results}")
    print(f"Wrote {out_nulls}")
    print(f"Wrote {out_json}")
    print(f"Wrote {subj_json}")
    print(f"Wrote {fig_dist}")
    print(f"Wrote {fig_null}")


if __name__ == "__main__":
    main()
