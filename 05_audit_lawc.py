#!/usr/bin/env python3
"""Fail-closed Law-C audit gate run after feature extraction and before GPU modules."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd

from common.lawc_audit import (
    bh_fdr,
    collect_lawc_trials_from_features,
    load_lawc_event_map,
    permutation_test_median_rho,
    subject_level_rhos,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--event_map", type=Path, default=Path("configs/lawc_event_map.yaml"))
    ap.add_argument("--datasets", type=str, default="ds005095,ds003655,ds004117")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--n_control_shuffles", type=int, default=0, help="0=use config/default")
    ap.add_argument("--control_p_threshold", type=float, default=None)
    ap.add_argument("--control_rho_threshold", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=0, help="Parallel dataset workers. 0=auto")
    return ap.parse_args()


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _shuffle_within_subject(df: pd.DataFrame, *, col: str, seed: int) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(int(seed))
    for _, idx in out.groupby("subject_key").groups.items():
        idx_list = list(idx)
        vals = out.loc[idx_list, col].to_numpy(copy=True)
        out.loc[idx_list, col] = vals[rng.permutation(len(vals))]
    return out


def _gather_feature_attr_stats(features_root: Path, dataset_id: str) -> Dict[str, Any]:
    event_filters = {}
    load_columns = {}
    load_signs = {}
    rt_sources = {}
    channels = {}

    for fp in sorted(features_root.rglob("*.h5")):
        with h5py.File(fp, "r") as h:
            ds = str(h.attrs.get("dataset_id", ""))
            if ds != dataset_id:
                continue
            ef = str(h.attrs.get("lawc_event_filter", ""))
            lc = str(h.attrs.get("lawc_load_column", ""))
            ls = h.attrs.get("lawc_load_sign", None)
            rs = str(h.attrs.get("lawc_rt_source", ""))
            if ef:
                event_filters[ef] = event_filters.get(ef, 0) + 1
            if lc:
                load_columns[lc] = load_columns.get(lc, 0) + 1
            if ls is not None:
                try:
                    ls_key = f"{float(ls):g}"
                except Exception:
                    ls_key = str(ls)
                load_signs[ls_key] = load_signs.get(ls_key, 0) + 1
            if rs:
                rt_sources[rs] = rt_sources.get(rs, 0) + 1
            if "p3b_channel" in h:
                ch = np.asarray(h["p3b_channel"]).astype(str)
                for c in ch:
                    channels[c] = channels.get(c, 0) + 1

    return {
        "event_filter_counts": event_filters,
        "load_column_counts": load_columns,
        "load_sign_counts": load_signs,
        "rt_source_counts": rt_sources,
        "channel_counts": channels,
    }


def _control_shuffle_summary(
    df: pd.DataFrame,
    *,
    col: str,
    n_shuffles: int,
    seed: int,
    min_trials: int,
) -> Dict[str, Any]:
    medians: List[float] = []
    reps = max(1, int(n_shuffles))
    for rep in range(reps):
        s = _shuffle_within_subject(df, col=col, seed=int(seed) + rep * 7919)
        subj = subject_level_rhos(s, min_trials=min_trials)
        medians.append(float(subj.get("median_rho", np.nan)))

    arr = np.asarray(medians, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median_rho": float("nan"), "p_value": float("nan"), "n_shuffles": 0}

    med = float(np.median(arr))
    # Probability of non-negative median correlation under repeated control shuffles.
    p_nonnegative = float((1.0 + np.sum(arr >= 0.0)) / (1.0 + arr.size))
    return {
        "median_rho": med,
        "p_value": p_nonnegative,
        "n_shuffles": int(arr.size),
    }


def _analyze_dataset(
    *,
    dataset_id: str,
    features_root: Path,
    n_perm: int,
    seed: int,
    min_trials: int,
    n_control_shuffles: int,
    control_p_threshold: float,
    control_rho_threshold: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    df = collect_lawc_trials_from_features(features_root, dataset_id)
    if df.empty:
        raise RuntimeError(f"Law-C audit failure: no extracted feature trials found for dataset={dataset_id}")

    df = df[np.isfinite(df["p3b_amp"]) & np.isfinite(df["memory_load"])].copy()
    if df.empty:
        raise RuntimeError(f"Law-C audit failure: no finite p3b/load rows for dataset={dataset_id}")

    subj = subject_level_rhos(df, min_trials=min_trials)
    perm = permutation_test_median_rho(
        df,
        n_perm=n_perm,
        seed=seed + (abs(hash(dataset_id)) % 10000),
        mode="y_shuffle",
        min_trials=min_trials,
    )

    x_ctrl = _control_shuffle_summary(
        df,
        col="p3b_amp",
        n_shuffles=n_control_shuffles,
        seed=seed + 17,
        min_trials=min_trials,
    )
    y_ctrl = _control_shuffle_summary(
        df,
        col="memory_load",
        n_shuffles=n_control_shuffles,
        seed=seed + 29,
        min_trials=min_trials,
    )

    x_degrade = (
        float(x_ctrl.get("median_rho", np.nan)) <= float(control_rho_threshold)
        or float(x_ctrl.get("p_value", np.nan)) > control_p_threshold
    )
    y_degrade = (
        float(y_ctrl.get("median_rho", np.nan)) <= float(control_rho_threshold)
        or float(y_ctrl.get("p_value", np.nan)) > control_p_threshold
    )

    rt_nonmissing = float(np.isfinite(df["rt"]).mean()) if "rt" in df.columns else float("nan")
    attr_stats = _gather_feature_attr_stats(features_root, dataset_id)

    row = {
        "dataset_id": dataset_id,
        "n_trials": int(len(df)),
        "n_subjects": int(df["subject_key"].nunique()),
        "median_rho": float(subj.get("median_rho", np.nan)),
        "p_value": float(perm.get("p_value", np.nan)),
        "n_subjects_used": int(subj.get("n_subjects_used", 0)),
        "rt_nonmissing_rate": rt_nonmissing,
        "fallback_non_pz_trials": int((df["p3b_channel"].astype(str).str.upper() != "PZ").sum()),
        "event_filter_counts": attr_stats.get("event_filter_counts", {}),
        "load_column_counts": attr_stats.get("load_column_counts", {}),
        "load_sign_counts": attr_stats.get("load_sign_counts", {}),
        "rt_source_counts": attr_stats.get("rt_source_counts", {}),
        "channel_counts": attr_stats.get("channel_counts", {}),
        "x_control_median_rho": float(x_ctrl.get("median_rho", np.nan)),
        "x_control_p_value": float(x_ctrl.get("p_value", np.nan)),
        "y_control_median_rho": float(y_ctrl.get("median_rho", np.nan)),
        "y_control_p_value": float(y_ctrl.get("p_value", np.nan)),
        "control_n_shuffles": int(max(x_ctrl.get("n_shuffles", 0), y_ctrl.get("n_shuffles", 0))),
        "x_control_degrade_pass": bool(x_degrade),
        "y_control_degrade_pass": bool(y_degrade),
    }

    controls = [
        {
            "dataset_id": dataset_id,
            "control": "x_shuffle",
            "median_rho": float(x_ctrl.get("median_rho", np.nan)),
            "p_value": float(x_ctrl.get("p_value", np.nan)),
            "n_shuffles": int(x_ctrl.get("n_shuffles", 0)),
            "degrade_pass": bool(x_degrade),
        },
        {
            "dataset_id": dataset_id,
            "control": "y_shuffle",
            "median_rho": float(y_ctrl.get("median_rho", np.nan)),
            "p_value": float(y_ctrl.get("p_value", np.nan)),
            "n_shuffles": int(y_ctrl.get("n_shuffles", 0)),
            "degrade_pass": bool(y_degrade),
        },
    ]
    return row, controls


def main() -> None:
    args = parse_args()
    if args.n_perm < 2000:
        raise ValueError("n_perm must be >= 2000 for locked Law-C inference")

    event_map = load_lawc_event_map(args.event_map)
    defaults = event_map.get("defaults", {})

    control_p_threshold = (
        float(args.control_p_threshold)
        if args.control_p_threshold is not None
        else float(defaults.get("control_p_threshold", 0.2))
    )
    min_trials = int(defaults.get("min_trials_per_subject", 20))
    n_control_shuffles = int(args.n_control_shuffles) if int(args.n_control_shuffles) > 0 else int(defaults.get("control_n_shuffles", 128))

    datasets = _split_csv(args.datasets)
    audit_dir = args.out_root / "lawc_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_audit_md = args.out_root / "AUDIT" / "LawcAudit.md"
    out_audit_md.parent.mkdir(parents=True, exist_ok=True)

    primary_rows: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []

    workers = int(args.workers) if int(args.workers) > 0 else min(len(datasets), max(1, os.cpu_count() or 1))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                _analyze_dataset,
                dataset_id=ds,
                features_root=args.features_root,
                n_perm=args.n_perm,
                seed=args.seed,
                min_trials=min_trials,
                n_control_shuffles=n_control_shuffles,
                control_p_threshold=control_p_threshold,
                control_rho_threshold=float(args.control_rho_threshold),
            ): ds
            for ds in datasets
        }
        for fut in as_completed(futs):
            ds = futs[fut]
            row, controls = fut.result()
            primary_rows.append(row)
            control_rows.extend(controls)
            print(f"[LawC] dataset complete: {ds} median_rho={row['median_rho']:.6f} p={row['p_value']:.6g}", flush=True)

    primary_rows = sorted(primary_rows, key=lambda r: str(r.get("dataset_id", "")))
    control_rows = sorted(control_rows, key=lambda r: (str(r.get("dataset_id", "")), str(r.get("control", ""))))

    pvals = [float(r["p_value"]) for r in primary_rows]
    qvals = bh_fdr(pvals)

    failures: List[str] = []
    for row, q in zip(primary_rows, qvals):
        row["q_value"] = float(q)
        pass_primary = bool(float(row["median_rho"]) > 0.0 and float(q) < 0.05)
        pass_controls = bool(row["x_control_degrade_pass"] and row["y_control_degrade_pass"])
        row["pass_primary"] = pass_primary
        row["pass_controls"] = pass_controls
        row["pass_all"] = bool(pass_primary and pass_controls)

        if not row["pass_all"]:
            reasons = []
            if not pass_primary:
                reasons.append(f"primary failed (median_rho={row['median_rho']:.4f}, q={row['q_value']:.4g})")
            if not pass_controls:
                reasons.append(
                    f"controls failed (x_degrade={row['x_control_degrade_pass']}, y_degrade={row['y_control_degrade_pass']})"
                )
            failures.append(f"{row['dataset_id']}: " + "; ".join(reasons))

    payload = {
        "datasets": primary_rows,
        "control_thresholds": {
            "control_p_threshold": control_p_threshold,
            "control_rho_threshold": float(args.control_rho_threshold),
            "n_control_shuffles": int(n_control_shuffles),
        },
        "pass": len(failures) == 0,
        "failures": failures,
    }

    out_json = audit_dir / "locked_test_results.json"
    out_csv = audit_dir / "locked_test_results.csv"
    out_controls = audit_dir / "negative_controls.csv"

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(primary_rows).to_csv(out_csv, index=False)
    pd.DataFrame(control_rows).to_csv(out_controls, index=False)

    lines: List[str] = []
    lines.append("# Law-C Audit")
    lines.append("")
    lines.append(f"- Features root: `{args.features_root}`")
    lines.append(f"- Datasets: `{','.join(datasets)}`")
    lines.append(f"- Permutations: `{args.n_perm}`")
    lines.append(f"- Control shuffles: `{n_control_shuffles}`")
    lines.append(f"- Parallel workers: `{workers}`")
    lines.append(f"- Control p-threshold: `{control_p_threshold}`")
    lines.append("")
    lines.append("## Locked Results")
    lines.append("| Dataset | Median rho | p | q | Controls degrade | PASS |")
    lines.append("|---|---:|---:|---:|---|---|")
    for row in primary_rows:
        ctrl = f"x={row['x_control_degrade_pass']} y={row['y_control_degrade_pass']}"
        lines.append(
            f"| {row['dataset_id']} | {row['median_rho']:.6f} | {row['p_value']:.6g} | {row['q_value']:.6g} | {ctrl} | {row['pass_all']} |"
        )
    lines.append("")

    lines.append("## Event Selection + Load Mapping")
    for row in primary_rows:
        lines.append(f"### {row['dataset_id']}")
        lines.append(f"- event_filter counts: `{json.dumps(row['event_filter_counts'])}`")
        lines.append(f"- load_column counts: `{json.dumps(row['load_column_counts'])}`")
        lines.append(f"- load_sign counts: `{json.dumps(row['load_sign_counts'])}`")
        lines.append(f"- rt_source counts: `{json.dumps(row['rt_source_counts'])}`")
        lines.append(f"- channel counts: `{json.dumps(row['channel_counts'])}`")
        lines.append(f"- fallback (non-Pz) trials: `{row['fallback_non_pz_trials']}`")
        lines.append(f"- RT non-missing rate: `{row['rt_nonmissing_rate']:.3f}`")
        lines.append("")

    if failures:
        lines.append("## FAILURES")
        for f in failures:
            lines.append(f"- {f}")
    else:
        lines.append("## PASS")
        lines.append("- All datasets passed primary Law-C + negative controls.")

    out_audit_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_controls}")
    print(f"Wrote {out_audit_md}")

    if failures:
        raise RuntimeError("Law-C audit failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()
