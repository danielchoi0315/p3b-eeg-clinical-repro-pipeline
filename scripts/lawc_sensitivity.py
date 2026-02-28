#!/usr/bin/env python3
"""Law-C sensitivity checks for load-decoding transforms."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
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

from common.lawc_audit import (  # noqa: E402
    align_probe_events_to_epochs,
    load_lawc_event_map,
    prepare_probe_event_table,
    subject_level_rhos,
    subject_key_from_entities,
)
from p3b_pipeline.h5io import iter_subject_feature_files, read_subject_h5  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--event_map", type=Path, default=Path("configs/lawc_event_map.yaml"))
    ap.add_argument("--datasets", type=str, default="ds005095,ds003655,ds004117")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--min_trials", type=int, default=20)
    ap.add_argument("--onset_tolerance", type=float, default=0.01)
    return ap.parse_args()


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _events_path_for_attrs(dataset_root: Path, attrs: Dict[str, Any]) -> Optional[Path]:
    subject = _safe_str(attrs.get("bids_subject") or attrs.get("subject")).strip()
    if not subject:
        return None
    session = _safe_str(attrs.get("bids_session") or attrs.get("session")).strip()
    task = _safe_str(attrs.get("task")).strip()
    run = _safe_str(attrs.get("bids_run") or attrs.get("run")).strip()

    sub_dir = dataset_root / f"sub-{subject}"
    if session and session.lower() not in {"na", "none"}:
        eeg_dir = sub_dir / f"ses-{session}" / "eeg"
    else:
        eeg_dir = sub_dir / "eeg"
    if not eeg_dir.exists():
        return None

    patterns: List[str] = []
    if task and run:
        patterns.extend(
            [
                f"*task-{task}*run-{run}*_events.tsv",
                f"*task-{task}*run-{run}*_events.tsv.gz",
            ]
        )
    if task:
        patterns.extend([f"*task-{task}*_events.tsv", f"*task-{task}*_events.tsv.gz"])
    patterns.extend([f"sub-{subject}*_events.tsv", f"sub-{subject}*_events.tsv.gz"])

    for pat in patterns:
        hits = sorted(eeg_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _collect_dataset_trials(
    *,
    dataset_id: str,
    features_root: Path,
    data_root: Path,
    event_map: Dict[str, Any],
    onset_tolerance: float,
) -> Tuple[pd.DataFrame, List[str]]:
    dataset_root = data_root / dataset_id
    if not dataset_root.exists():
        return pd.DataFrame(), [f"dataset root missing: {dataset_root}"]

    errors: List[str] = []
    rows: List[pd.DataFrame] = []
    files = iter_subject_feature_files(features_root)
    for fp in files:
        arrays, attrs = read_subject_h5(fp)
        ds_attr = _safe_str(attrs.get("dataset_id")).strip()
        if ds_attr != dataset_id:
            continue
        if "p3b_amp" not in arrays or "memory_load" not in arrays or "onset_s" not in arrays:
            continue

        events_path = _events_path_for_attrs(dataset_root, attrs)
        if events_path is None:
            errors.append(f"{fp}: events file not found from BIDS entities")
            continue

        subject = _safe_str(attrs.get("bids_subject") or attrs.get("subject")).strip()
        session = _safe_str(attrs.get("bids_session") or attrs.get("session")).strip()
        run = _safe_str(attrs.get("bids_run") or attrs.get("run")).strip()
        task = _safe_str(attrs.get("task")).strip()
        if not subject:
            errors.append(f"{fp}: missing bids_subject/subject")
            continue

        try:
            probe_df, diag = prepare_probe_event_table(
                events_path=events_path,
                dataset_id=dataset_id,
                event_map=event_map,
                dataset_root=dataset_root,
                bids_subject=subject,
                bids_task=task or None,
                bids_run=run or None,
                bids_session=session or None,
            )
        except Exception as exc:
            errors.append(f"{fp}: prepare_probe_event_table failed: {exc}")
            continue

        aligned = align_probe_events_to_epochs(
            epoch_onsets_s=np.asarray(arrays["onset_s"], dtype=float),
            probe_df=probe_df,
            tolerance_s=float(onset_tolerance),
        )
        matched = aligned[aligned["matched"]].copy()
        if matched.empty:
            errors.append(f"{fp}: no matched epochs for probe events")
            continue

        idx = matched["epoch_idx"].to_numpy(dtype=int)
        p3b = np.asarray(arrays["p3b_amp"], dtype=float)[idx]
        as_is_load = np.asarray(arrays["memory_load"], dtype=float)[idx]

        subject_key = _safe_str(attrs.get("subject_key")).strip()
        if not subject_key:
            subject_key = subject_key_from_entities(dataset_id=dataset_id, bids_subject=subject, bids_session=session or None)

        frame = pd.DataFrame(
            {
                "dataset_id": dataset_id,
                "subject_key": subject_key,
                "p3b_amp": p3b,
                "as_is_load": as_is_load,
                "trial_type": matched["trial_type"] if "trial_type" in matched.columns else np.nan,
                "value": matched["value"] if "value" in matched.columns else np.nan,
                "memory_cond": matched["memory_cond"] if "memory_cond" in matched.columns else np.nan,
                "load_column_used": _safe_str(diag.get("load_column_used")),
            }
        )
        rows.append(frame)

    if not rows:
        return pd.DataFrame(), errors
    out = pd.concat(rows, axis=0, ignore_index=True)
    return out, errors


def _collapsed_load(df: pd.DataFrame, dataset_id: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if dataset_id == "ds005095":
        if "value" not in df.columns:
            return None, "source column 'value' missing; cannot apply floor(value/10)"
        raw = pd.to_numeric(df["value"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(raw).mean() <= 0.0:
            return None, "source column 'value' has no finite values"
        return np.floor(raw / 10.0), None

    if dataset_id == "ds003655":
        if "trial_type" not in df.columns:
            return None, "source column 'trial_type' missing; cannot parse load"
        parsed = df["trial_type"].astype(str).str.extract(r"(\d+)", expand=False)
        load = pd.to_numeric(parsed, errors="coerce").to_numpy(dtype=float)
        if np.isfinite(load).mean() <= 0.0:
            return None, "failed to parse integer load from trial_type"
        return load, None

    if dataset_id == "ds004117":
        return pd.to_numeric(df["as_is_load"], errors="coerce").to_numpy(dtype=float), None

    return None, "no collapse rule configured"


def _summarize_transform(
    *,
    dataset_id: str,
    transform: str,
    df: pd.DataFrame,
    load_values: np.ndarray,
    min_trials: int,
    reason_if_skip: str = "",
) -> Dict[str, Any]:
    n_trials_total = int(len(df))
    tdf = pd.DataFrame(
        {
            "subject_key": df["subject_key"].astype(str),
            "p3b_amp": pd.to_numeric(df["p3b_amp"], errors="coerce"),
            "memory_load": pd.to_numeric(pd.Series(load_values), errors="coerce"),
        }
    )
    valid = tdf[np.isfinite(tdf["p3b_amp"]) & np.isfinite(tdf["memory_load"])].copy()
    n_trials_kept = int(len(valid))
    frac_kept = float(n_trials_kept / max(n_trials_total, 1))
    n_subj_total = int(df["subject_key"].nunique())
    n_levels = int(pd.Series(valid["memory_load"]).nunique()) if n_trials_kept > 0 else 0

    if n_trials_kept == 0:
        return {
            "dataset_id": dataset_id,
            "transform": transform,
            "status": "SKIP",
            "reason": reason_if_skip or "no finite trials for transform",
            "n_trials_total": n_trials_total,
            "n_trials_kept": n_trials_kept,
            "frac_trials_kept": frac_kept,
            "n_subjects_total": n_subj_total,
            "n_subjects_used": 0,
            "n_unique_load_levels": n_levels,
            "median_rho": float("nan"),
            "sign_consistency_posfrac": float("nan"),
        }

    subj = subject_level_rhos(valid, min_trials=min_trials, x_col="p3b_amp", y_col="memory_load")
    rhos = np.asarray(subj.get("subject_rhos", []), dtype=float)
    sign_consistency = float(np.mean(rhos > 0.0)) if rhos.size > 0 else float("nan")
    n_used = int(subj.get("n_subjects_used", 0))
    status = "PASS" if n_used > 0 else "SKIP"
    reason = "" if status == "PASS" else (reason_if_skip or f"no subjects with >= {min_trials} valid trials")

    return {
        "dataset_id": dataset_id,
        "transform": transform,
        "status": status,
        "reason": reason,
        "n_trials_total": n_trials_total,
        "n_trials_kept": n_trials_kept,
        "frac_trials_kept": frac_kept,
        "n_subjects_total": n_subj_total,
        "n_subjects_used": n_used,
        "n_unique_load_levels": n_levels,
        "median_rho": float(subj.get("median_rho", np.nan)),
        "sign_consistency_posfrac": sign_consistency,
    }


def _plot_sensitivity(df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    pass_df = df[df["status"] == "PASS"].copy()
    if pass_df.empty:
        ax.text(0.5, 0.5, "No PASS rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = [f"{d}\n{t}" for d, t in zip(pass_df["dataset_id"], pass_df["transform"])]
        vals = pass_df["median_rho"].to_numpy(dtype=float)
        colors = ["#1f77b4" if t == "as_is" else "#ff7f0e" for t in pass_df["transform"]]
        ax.bar(np.arange(len(vals)), vals, color=colors)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Median within-subject Spearman rho")
        ax.set_title("Law-C Sensitivity: As-Is vs Collapsed Load")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_level_counts(df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{d}\n{t}" for d, t in zip(df["dataset_id"], df["transform"])]
    vals = df["n_unique_load_levels"].to_numpy(dtype=float)
    colors = ["#2ca02c" if s == "PASS" else "#d62728" for s in df["status"]]
    ax.bar(np.arange(len(vals)), vals, color=colors)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Unique load levels (kept trials)")
    ax.set_title("Law-C Load-Level Counts by Transform")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _summary_markdown(df: pd.DataFrame, errors: Dict[str, List[str]]) -> str:
    lines: List[str] = []
    lines.append("# Law-C Sensitivity Summary")
    lines.append("")
    lines.append("| Dataset | Transform | Status | Median rho | Sign(rho>0) | Load levels | Trials kept | Reason |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for _, r in df.sort_values(["dataset_id", "transform"]).iterrows():
        med = float(r["median_rho"]) if np.isfinite(r["median_rho"]) else float("nan")
        pos = float(r["sign_consistency_posfrac"]) if np.isfinite(r["sign_consistency_posfrac"]) else float("nan")
        lines.append(
            f"| {r['dataset_id']} | {r['transform']} | {r['status']} | {med:.6g} | {pos:.3f} | "
            f"{int(r['n_unique_load_levels'])} | {float(r['frac_trials_kept']):.3f} | {r['reason']} |"
        )
    lines.append("")
    lines.append("## Collection Notes")
    for ds, errs in sorted(errors.items()):
        if not errs:
            continue
        lines.append(f"### {ds}")
        for msg in errs[:20]:
            lines.append(f"- {msg}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    event_map = load_lawc_event_map(args.event_map)
    min_trials = int(args.min_trials or (event_map.get("defaults", {}) or {}).get("min_trials_per_subject", 20))

    rows: List[Dict[str, Any]] = []
    all_errors: Dict[str, List[str]] = {}
    for dataset_id in _split_csv(args.datasets):
        df, errs = _collect_dataset_trials(
            dataset_id=dataset_id,
            features_root=args.features_root,
            data_root=args.data_root,
            event_map=event_map,
            onset_tolerance=float(args.onset_tolerance),
        )
        all_errors[dataset_id] = errs

        if df.empty:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "transform": "as_is",
                    "status": "SKIP",
                    "reason": "no matched trials in features/events alignment",
                    "n_trials_total": 0,
                    "n_trials_kept": 0,
                    "frac_trials_kept": 0.0,
                    "n_subjects_total": 0,
                    "n_subjects_used": 0,
                    "n_unique_load_levels": 0,
                    "median_rho": float("nan"),
                    "sign_consistency_posfrac": float("nan"),
                }
            )
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "transform": "collapsed",
                    "status": "SKIP",
                    "reason": "no matched trials in features/events alignment",
                    "n_trials_total": 0,
                    "n_trials_kept": 0,
                    "frac_trials_kept": 0.0,
                    "n_subjects_total": 0,
                    "n_subjects_used": 0,
                    "n_unique_load_levels": 0,
                    "median_rho": float("nan"),
                    "sign_consistency_posfrac": float("nan"),
                }
            )
            continue

        as_is = _summarize_transform(
            dataset_id=dataset_id,
            transform="as_is",
            df=df,
            load_values=pd.to_numeric(df["as_is_load"], errors="coerce").to_numpy(dtype=float),
            min_trials=min_trials,
        )
        rows.append(as_is)

        collapsed_values, collapse_reason = _collapsed_load(df, dataset_id)
        if collapsed_values is None:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "transform": "collapsed",
                    "status": "SKIP",
                    "reason": collapse_reason or "collapsed transform unavailable",
                    "n_trials_total": int(len(df)),
                    "n_trials_kept": 0,
                    "frac_trials_kept": 0.0,
                    "n_subjects_total": int(df["subject_key"].nunique()),
                    "n_subjects_used": 0,
                    "n_unique_load_levels": 0,
                    "median_rho": float("nan"),
                    "sign_consistency_posfrac": float("nan"),
                }
            )
        else:
            rows.append(
                _summarize_transform(
                    dataset_id=dataset_id,
                    transform="collapsed",
                    df=df,
                    load_values=collapsed_values,
                    min_trials=min_trials,
                )
            )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["dataset_id", "transform"]).reset_index(drop=True)

    table_csv = args.out_dir / "sensitivity_table.csv"
    summary_md = args.out_dir / "sensitivity_summary.md"
    fig_sensitivity = args.out_dir / "FIG_lawc_sensitivity.png"
    fig_levels = args.out_dir / "FIG_lawc_level_counts.png"
    err_json = args.out_dir / "collection_errors.json"

    out_df.to_csv(table_csv, index=False)
    summary_md.write_text(_summary_markdown(out_df, all_errors), encoding="utf-8")
    _plot_sensitivity(out_df, fig_sensitivity)
    _plot_level_counts(out_df, fig_levels)
    err_json.write_text(json.dumps(all_errors, indent=2), encoding="utf-8")

    print(f"Wrote {table_csv}")
    print(f"Wrote {summary_md}")
    print(f"Wrote {fig_sensitivity}")
    print(f"Wrote {fig_levels}")
    print(f"Wrote {err_json}")


if __name__ == "__main__":
    main()
