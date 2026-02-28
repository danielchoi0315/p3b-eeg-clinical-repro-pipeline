#!/usr/bin/env python3
"""Extract ds004796 Sternberg features using existing pipeline modules."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cmd(cmd: List[str], *, cwd: Path, log_file: Path, env: Dict[str, str]) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, text=True, check=False)
    return int(p.returncode)


def _extract_subject(path: Path) -> str:
    m = re.search(r"sub-([A-Za-z0-9]+)", str(path))
    return m.group(1) if m else ""


def _collect_sternberg_subjects(dataset_root: Path) -> List[str]:
    out = set()
    for p in dataset_root.rglob("*task-sternberg*_events.tsv"):
        sid = _extract_subject(p)
        if sid:
            out.add(sid)
    return sorted(out)


def _decode_obj_array(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind not in {"O", "S", "U"}:
        return arr
    out = []
    for x in arr.tolist():
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="ignore"))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=object)


def _build_trial_table(features_root: Path, dataset_id: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    ds_root = features_root / dataset_id
    for fp in sorted(ds_root.rglob("*.h5")):
        with h5py.File(fp, "r") as h:
            keys = set(h.keys())
            if "p3b_amp" not in keys or "memory_load" not in keys:
                continue
            n = int(np.asarray(h["p3b_amp"]).shape[0])
            if n <= 0:
                continue

            trial = pd.DataFrame(
                {
                    "p3_amp_uV": pd.to_numeric(np.asarray(h["p3b_amp"], dtype=float), errors="coerce"),
                    "p3_lat_s": pd.to_numeric(np.asarray(h["p3b_lat"], dtype=float), errors="coerce")
                    if "p3b_lat" in keys
                    else np.full(n, np.nan),
                    "memory_load": pd.to_numeric(np.asarray(h["memory_load"], dtype=float), errors="coerce"),
                    "trial_order": pd.to_numeric(np.asarray(h["trial_order"], dtype=float), errors="coerce")
                    if "trial_order" in keys
                    else np.arange(1, n + 1, dtype=float),
                    "onset_s": pd.to_numeric(np.asarray(h["onset_s"], dtype=float), errors="coerce")
                    if "onset_s" in keys
                    else np.full(n, np.nan),
                    "rt": pd.to_numeric(np.asarray(h["rt"], dtype=float), errors="coerce")
                    if "rt" in keys
                    else np.full(n, np.nan),
                    "accuracy": pd.to_numeric(np.asarray(h["accuracy"], dtype=float), errors="coerce")
                    if "accuracy" in keys
                    else np.full(n, np.nan),
                    "age": pd.to_numeric(np.asarray(h["age"], dtype=float), errors="coerce")
                    if "age" in keys
                    else np.full(n, np.nan),
                }
            )

            trial["p3_channel"] = (
                _decode_obj_array(np.asarray(h["p3b_channel"])).astype(str)
                if "p3b_channel" in keys
                else np.asarray([""] * n, dtype=object)
            )
            trial["subject_key"] = (
                _decode_obj_array(np.asarray(h["subject_key"])).astype(str)
                if "subject_key" in keys
                else np.asarray([str(h.attrs.get("subject_key", ""))] * n, dtype=object)
            )
            trial["dataset_id"] = (
                _decode_obj_array(np.asarray(h["dataset_id"])).astype(str)
                if "dataset_id" in keys
                else np.asarray([str(h.attrs.get("dataset_id", dataset_id))] * n, dtype=object)
            )

            # BIDS entities for endpoint joins and diagnostics.
            for c in ["bids_subject", "bids_session", "bids_run"]:
                if c in keys:
                    trial[c] = _decode_obj_array(np.asarray(h[c])).astype(str)
                else:
                    trial[c] = str(h.attrs.get(c, ""))
            trial["features_h5"] = str(fp)
            rows.append(trial)

    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, axis=0, ignore_index=True)
    df["subject_id"] = (
        df["subject_key"]
        .astype(str)
        .str.extract(r"sub-([^:]+)", expand=False)
        .fillna(df["bids_subject"].astype(str).str.replace(r"^sub-", "", regex=True))
    )
    return df


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--deriv_root", type=Path, required=True)
    ap.add_argument("--features_root", type=Path, required=True)
    ap.add_argument("--event_map", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--dataset_id", type=str, default="ds004796")
    ap.add_argument("--cpu_workers", type=int, default=32)
    ap.add_argument("--out_summary", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_summary = args.out_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    log_file = out_summary.parent / "extract_features.log"
    trial_csv = out_summary.parent / "trial_table.csv"

    subjects = _collect_sternberg_subjects(args.dataset_root)
    if not subjects:
        raise RuntimeError(f"no sternberg events found under {args.dataset_root}")

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["NUMEXPR_NUM_THREADS"] = "2"
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(REPO_ROOT / "src")

    args.deriv_root.mkdir(parents=True, exist_ok=True)
    args.features_root.mkdir(parents=True, exist_ok=True)

    pre_cmd = [
        sys.executable,
        "01_preprocess_CPU.py",
        "--bids_root",
        str(args.dataset_root),
        "--deriv_root",
        str(args.deriv_root),
        "--config",
        str(args.config),
        "--workers",
        str(max(1, min(12, args.cpu_workers // 3))),
        "--mne_n_jobs",
        "1",
        "--subjects",
        *subjects,
    ]
    rc = _run_cmd(pre_cmd, cwd=REPO_ROOT, log_file=log_file, env=env)
    if rc != 0:
        raise RuntimeError(f"preprocess failed rc={rc}; see {log_file}")

    ext_cmd = [
        sys.executable,
        "02_extract_features_CPU.py",
        "--bids_root",
        str(args.dataset_root),
        "--deriv_root",
        str(args.deriv_root),
        "--features_root",
        str(args.features_root),
        "--config",
        str(args.config),
        "--cohort",
        "healthy",
        "--dataset_id",
        str(args.dataset_id),
        "--lawc_event_map",
        str(args.event_map),
        "--workers",
        str(max(1, min(12, args.cpu_workers // 3))),
        "--subjects",
        *subjects,
    ]
    rc = _run_cmd(ext_cmd, cwd=REPO_ROOT, log_file=log_file, env=env)
    if rc != 0:
        raise RuntimeError(f"feature extraction failed rc={rc}; see {log_file}")

    trial_df = _build_trial_table(args.features_root, str(args.dataset_id))
    if trial_df.empty:
        raise RuntimeError("no extracted feature trials found after extraction")
    trial_df.to_csv(trial_csv, index=False)

    pz = trial_df["p3_channel"].astype(str).str.upper().eq("PZ")
    fallback = (~pz).sum()
    summary = {
        "status": "PASS",
        "dataset_id": str(args.dataset_id),
        "dataset_root": str(args.dataset_root),
        "deriv_root": str(args.deriv_root),
        "features_root": str(args.features_root),
        "trial_table_csv": str(trial_csv),
        "n_subjects": int(trial_df["subject_id"].nunique()),
        "n_trials": int(len(trial_df)),
        "n_trials_with_load": int(np.isfinite(pd.to_numeric(trial_df["memory_load"], errors="coerce")).sum()),
        "n_trials_with_rt": int(np.isfinite(pd.to_numeric(trial_df["rt"], errors="coerce")).sum()),
        "n_trials_with_accuracy": int(np.isfinite(pd.to_numeric(trial_df["accuracy"], errors="coerce")).sum()),
        "p3_channel_counts": {str(k): int(v) for k, v in trial_df["p3_channel"].astype(str).value_counts().to_dict().items()},
        "fallback_non_pz_trials": int(fallback),
        "log_file": str(log_file),
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
