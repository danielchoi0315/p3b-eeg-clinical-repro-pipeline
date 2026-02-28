#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd

from common import ensure_out_tree, ensure_stage_status, stop_reason


CORE_DATASETS = ["ds003655", "ds004117", "ds005095"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--worker_index", type=int, default=-1)
    ap.add_argument("--max_workers", type=int, default=8)
    return ap.parse_args()


def _safe_subject(attrs: Dict[str, Any], fp: Path) -> str:
    for k in ["subject_key", "subject", "bids_subject", "subject_id"]:
        if k in attrs and str(attrs[k]).strip():
            return str(attrs[k]).strip()
    return fp.stem


def _read_subject_file(fp: Path) -> Tuple[str, str, np.ndarray]:
    with h5py.File(fp, "r") as h:
        attrs = {k: h.attrs[k] for k in h.attrs.keys()}
        ds = str(attrs.get("dataset_id", "")).strip()
        if not ds:
            ds = fp.parts[-3] if len(fp.parts) >= 3 else ""
        subject = _safe_subject(attrs, fp)
        if "p3b_amp" not in h:
            return ds, subject, np.asarray([], dtype=float)
        arr = np.asarray(h["p3b_amp"], dtype=float)
        return ds, subject, arr


def _collect_feature_files(features_root: Path) -> List[Path]:
    return [p for p in features_root.rglob("*.h5") if p.is_file()]


def _repeat_worker(inputs: Tuple[Path, int]) -> List[Dict[str, Any]]:
    features_root, seed = inputs
    rng = np.random.default_rng(seed)
    files = _collect_feature_files(features_root)
    rows: List[Dict[str, Any]] = []

    by_ds: Dict[str, Dict[str, Tuple[float, float]]] = {ds: {} for ds in CORE_DATASETS}
    for fp in files:
        ds, subj, arr = _read_subject_file(fp)
        if ds not in by_ds:
            continue
        x = np.asarray(arr, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 4:
            continue
        perm = rng.permutation(x.size)
        odd = x[perm[::2]]
        even = x[perm[1::2]]
        if odd.size == 0 or even.size == 0:
            continue
        by_ds[ds][subj] = (float(np.mean(odd)), float(np.mean(even)))

    for ds, subj_map in by_ds.items():
        if not subj_map:
            rows.append({"dataset_id": ds, "repeat": seed, "n_subjects": 0, "split_half_rho": float("nan")})
            continue
        a = [v[0] for v in subj_map.values()]
        b = [v[1] for v in subj_map.values()]
        df = pd.DataFrame({"a": a, "b": b})
        rho = float(df["a"].corr(df["b"], method="spearman")) if len(df) >= 3 else float("nan")
        rows.append({"dataset_id": ds, "repeat": seed, "n_subjects": int(len(df)), "split_half_rho": rho})
    return rows


def main() -> int:
    args = parse_args()
    out_root = args.out_root
    paths = ensure_out_tree(out_root)
    audit = paths["AUDIT"]
    rel_dir = paths["RELIABILITY"]

    stage3 = audit / "stage3_match_check_summary.json"
    if not stage3.exists() or json.loads(stage3.read_text(encoding="utf-8")).get("status") != "PASS":
        stop_reason(audit / "STOP_REASON_stage5_reliability.md", "stage5_reliability", "Blocked because stage3 is not PASS")
        ensure_stage_status(audit, "stage5_reliability", "SKIP", {"reason": "blocked_by_stage3"})
        return 0

    features_root = paths["REPRO_FROM_SCRATCH"] / "_features_cache"
    if not features_root.exists():
        stop_reason(audit / "STOP_REASON_stage5_reliability.md", "stage5_reliability", "Features root missing", {"features_root": str(features_root)})
        ensure_stage_status(audit, "stage5_reliability", "FAIL", {"reason": "features_missing"})
        return 1

    if args.worker_index >= 0:
        rows = _repeat_worker((features_root, int(args.worker_index)))
        (rel_dir / f"repeat_{args.worker_index:03d}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return 0

    repeats = int(max(1, args.repeats))
    workers = max(1, min(int(args.max_workers), mp.cpu_count(), repeats))
    with mp.Pool(processes=workers) as pool:
        nested = pool.map(_repeat_worker, [(features_root, i) for i in range(repeats)])

    rows = [r for rr in nested for r in rr]
    out_csv = rel_dir / "split_half_repeats.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset_id", "repeat", "n_subjects", "split_half_rho"])
        w.writeheader()
        w.writerows(rows)

    # Attenuation-corrected effect (exploratory): observed lawc median / sqrt(reliability).
    lawc_csv = paths["REPRO_FROM_SCRATCH"] / "PACK_CORE_LAWC" / "lawc_ultradeep" / "lawc_audit" / "locked_test_results.csv"
    lawc_rows: Dict[str, float] = {}
    if lawc_csv.exists():
        with lawc_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                ds = str(row.get("dataset_id", "")).strip()
                if not ds:
                    continue
                try:
                    lawc_rows[ds] = float(row.get("observed_median", row.get("median_subject_rho", "nan")))
                except Exception:
                    pass

    df = pd.DataFrame(rows)
    out_att: List[Dict[str, Any]] = []
    for ds in CORE_DATASETS:
        sub = df[df["dataset_id"] == ds].copy()
        rel = float(np.nanmedian(pd.to_numeric(sub["split_half_rho"], errors="coerce").to_numpy(dtype=float))) if not sub.empty else float("nan")
        obs = lawc_rows.get(ds, float("nan"))
        corr = float(obs / math.sqrt(max(rel, 1e-6))) if math.isfinite(obs) and math.isfinite(rel) and rel > 0 else float("nan")
        out_att.append({"dataset_id": ds, "observed_median_rho": obs, "split_half_reliability_median": rel, "attenuation_corrected_slope_exploratory": corr})

    att_csv = rel_dir / "attenuation_report.csv"
    with att_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["dataset_id", "observed_median_rho", "split_half_reliability_median", "attenuation_corrected_slope_exploratory"],
        )
        w.writeheader()
        w.writerows(out_att)

    lines = [
        "# Reliability Report",
        "",
        f"- repeats: `{repeats}`",
        f"- split_half_csv: `{out_csv}`",
        f"- attenuation_csv: `{att_csv}`",
    ]
    (rel_dir / "reliability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ensure_stage_status(
        audit,
        "stage5_reliability",
        "PASS",
        {"repeats": repeats, "split_half_csv": str(out_csv), "attenuation_csv": str(att_csv)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
