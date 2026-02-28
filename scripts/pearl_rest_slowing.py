#!/usr/bin/env python3
"""Exploratory resting-state EEG slowing metrics for ds004796."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


def _extract_subject(path: Path) -> str:
    m = re.search(r"sub-([A-Za-z0-9]+)", str(path))
    return m.group(1) if m else ""


def _rest_runs(dataset_root: Path) -> List[Path]:
    out = sorted(dataset_root.rglob("*task-rest*_eeg.vhdr"))
    if out:
        return out
    # fallback for alternative EEG formats
    for ext in ["edf", "bdf", "set", "fif"]:
        out.extend(sorted(dataset_root.rglob(f"*task-rest*_eeg.{ext}")))
    return sorted(set(out))


def _read_raw(path: Path):
    suf = path.suffix.lower()
    if suf == ".vhdr":
        return mne.io.read_raw_brainvision(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".edf":
        return mne.io.read_raw_edf(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".bdf":
        return mne.io.read_raw_bdf(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".set":
        return mne.io.read_raw_eeglab(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".fif":
        return mne.io.read_raw_fif(path.as_posix(), preload=True, verbose="ERROR")
    raise RuntimeError(f"unsupported rest file suffix: {path.suffix}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--participants_tsv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_subjects", type=int, default=32)
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _rest_runs(args.dataset_root)
    if not runs:
        payload = {"status": "SKIP", "reason": "no rest EEG runs found", "out_dir": str(out_dir)}
        (out_dir / "rest_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    # one run per subject to keep the exploratory pass fast
    by_sub: Dict[str, Path] = {}
    for p in runs:
        sid = _extract_subject(p)
        if sid and sid not in by_sub:
            by_sub[sid] = p
    sub_ids = sorted(by_sub.keys())[: max(1, int(args.max_subjects))]

    rows: List[Dict[str, Any]] = []
    for sid in sub_ids:
        p = by_sub[sid]
        try:
            raw = _read_raw(p)
            raw.pick_types(eeg=True, eog=False, misc=False, stim=False)
            if len(raw.ch_names) == 0:
                continue
            sf = float(raw.info["sfreq"])
            if sf > 256:
                raw.resample(256, verbose="ERROR")
            data = raw.get_data()
            if data.size == 0:
                continue
            psd, freqs = mne.time_frequency.psd_array_welch(
                data,
                sfreq=float(raw.info["sfreq"]),
                fmin=1.0,
                fmax=30.0,
                n_fft=1024,
                n_overlap=512,
                average="mean",
                verbose="ERROR",
            )
            pxx = np.nanmean(psd, axis=0)
            f = np.asarray(freqs, dtype=float)
            theta = float(np.trapz(pxx[(f >= 4) & (f < 8)], f[(f >= 4) & (f < 8)]))
            alpha = float(np.trapz(pxx[(f >= 8) & (f < 12)], f[(f >= 8) & (f < 12)]))
            beta = float(np.trapz(pxx[(f >= 13) & (f < 30)], f[(f >= 13) & (f < 30)]))
            total = float(np.trapz(pxx[(f >= 1) & (f <= 30)], f[(f >= 1) & (f <= 30)]))
            rows.append(
                {
                    "subject_id": sid,
                    "rest_file": str(p),
                    "theta_rel": theta / max(total, 1e-12),
                    "alpha_rel": alpha / max(total, 1e-12),
                    "beta_rel": beta / max(total, 1e-12),
                    "theta_alpha_ratio": theta / max(alpha, 1e-12),
                }
            )
        except Exception:
            continue

    if not rows:
        payload = {"status": "SKIP", "reason": "no usable rest runs after read/QC", "out_dir": str(out_dir)}
        (out_dir / "rest_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    df = pd.DataFrame(rows)
    parts = pd.read_csv(args.participants_tsv, sep="\t")
    parts["subject_id"] = parts["participant_id"].astype(str).str.replace(r"^sub-", "", regex=True)
    if "APOE_haplotype" in parts.columns:
        parts["APOE_e4_carrier"] = parts["APOE_haplotype"].astype(str).str.lower().str.contains("e4", na=False).astype(float)
    if "PICALM_rs3851179" in parts.columns:
        parts["PICALM_A_carrier"] = parts["PICALM_rs3851179"].astype(str).str.upper().str.contains("A", na=False).astype(float)
    df = df.merge(parts[["subject_id", "APOE_e4_carrier", "PICALM_A_carrier", "age", "sex"]], on="subject_id", how="left")

    metrics_csv = out_dir / "rest_slowing_metrics.csv"
    df.to_csv(metrics_csv, index=False)

    # Minimal exploratory group comparisons.
    out_rows: List[Dict[str, Any]] = []
    for ep in ["APOE_e4_carrier", "PICALM_A_carrier"]:
        if ep not in df.columns:
            continue
        work = df.copy()
        work["grp"] = pd.to_numeric(work[ep], errors="coerce")
        work = work.dropna(subset=["grp", "theta_alpha_ratio"])
        if work["grp"].nunique() < 2 or min((work["grp"] == 0).sum(), (work["grp"] == 1).sum()) < 6:
            continue
        v0 = work.loc[work["grp"] == 0, "theta_alpha_ratio"].to_numpy(dtype=float)
        v1 = work.loc[work["grp"] == 1, "theta_alpha_ratio"].to_numpy(dtype=float)
        obs = float(np.median(v1) - np.median(v0))
        rng = np.random.default_rng(int(args.seed) + len(out_rows) * 101)
        null = np.full(int(args.n_perm), np.nan, dtype=float)
        vals = work["theta_alpha_ratio"].to_numpy(dtype=float)
        grp = work["grp"].to_numpy(dtype=float)
        for i in range(int(args.n_perm)):
            g = grp.copy()
            rng.shuffle(g)
            a = vals[g == 0]
            b = vals[g == 1]
            if len(a) == 0 or len(b) == 0:
                continue
            null[i] = float(np.median(b) - np.median(a))
        finite = null[np.isfinite(null)]
        p = float((1.0 + np.sum(np.abs(finite) >= abs(obs))) / (1.0 + finite.size)) if finite.size else float("nan")
        out_rows.append(
            {
                "endpoint": ep,
                "n": int(len(work)),
                "delta_median_theta_alpha_ratio": obs,
                "perm_p": p,
            }
        )

    assoc_csv = out_dir / "rest_slowing_associations.csv"
    pd.DataFrame(out_rows).to_csv(assoc_csv, index=False)

    fig = out_dir / "FIG_rest_theta_alpha_ratio.png"
    fig_obj, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(pd.to_numeric(df["theta_alpha_ratio"], errors="coerce").dropna().to_numpy(dtype=float), bins=24, color="#2a607f", alpha=0.85)
    ax.set_title("Resting-State Theta/Alpha Ratio (Exploratory)")
    ax.set_xlabel("theta/alpha")
    ax.set_ylabel("count")
    ax.grid(alpha=0.2)
    fig_obj.tight_layout()
    fig_obj.savefig(fig, dpi=160)
    plt.close(fig_obj)

    payload = {
        "status": "PASS",
        "n_subjects": int(df["subject_id"].nunique()),
        "metrics_csv": str(metrics_csv),
        "associations_csv": str(assoc_csv),
        "figure": str(fig),
    }
    (out_dir / "rest_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
