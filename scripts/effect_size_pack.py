#!/usr/bin/env python3
"""Compute reviewer-facing effect sizes from healthy feature cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", type=Path, required=True)
    ap.add_argument("--datasets", type=str, default="ds005095,ds003655,ds004117")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min_trials", type=int, default=20)
    return ap.parse_args()


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _bootstrap_ci(values: Sequence[float], *, n_boot: int, seed: int) -> List[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(int(seed))
    boots = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        smp = arr[rng.integers(0, arr.size, size=arr.size)]
        boots[i] = float(np.median(smp))
    return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def _iter_dataset_files(features_root: Path, dataset_id: str) -> List[Path]:
    root = features_root / dataset_id
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*.h5") if p.is_file()])


def _subject_key_from_path(path: Path, dataset_id: str) -> str:
    sub = next((x for x in path.parts if x.startswith("sub-")), "sub-unknown")
    return f"{dataset_id}:{sub}"


def _load_h5_trial_frame(path: Path, dataset_id: str) -> pd.DataFrame:
    import h5py

    with h5py.File(path, "r") as h:
        if "p3b_amp" not in h or "memory_load" not in h:
            return pd.DataFrame()
        p3 = np.asarray(h["p3b_amp"], dtype=float) * 1e6  # V -> uV
        load = np.asarray(h["memory_load"], dtype=float)
        ch = np.asarray(h["p3b_channel"]).astype(str) if "p3b_channel" in h else np.asarray(["unknown"] * len(p3))
        if "subject_key" in h:
            sk = np.asarray(h["subject_key"]).astype(str)
            subject_key = sk[0] if sk.size else _subject_key_from_path(path, dataset_id)
        else:
            subject_key = str(h.attrs.get("subject_key", "")) or _subject_key_from_path(path, dataset_id)

    return pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "subject_key": subject_key,
            "p3_uv": p3,
            "memory_load": load,
            "p3_channel": ch,
        }
    )


def _robust_slope(load: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(load, dtype=float)
    yy = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(yy)
    x = x[m]
    yy = yy[m]
    if x.size < 3 or np.unique(x).size < 2:
        return float("nan")
    try:
        slope, _, _, _ = theilslopes(yy, x)
        return float(slope)
    except Exception:
        try:
            return float(np.polyfit(x, yy, deg=1)[0])
        except Exception:
            return float("nan")


def _plot_slopes(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    dsets = sorted(df["dataset_id"].unique().tolist())
    for i, ds in enumerate(dsets):
        vals = pd.to_numeric(df.loc[df["dataset_id"] == ds, "slope_uv_per_load"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        x = np.full(vals.shape, i + 1, dtype=float)
        jitter = np.linspace(-0.18, 0.18, num=len(vals), dtype=float) if len(vals) > 1 else np.asarray([0.0])
        ax.scatter(x + jitter, vals, alpha=0.45, s=15)
        if vals.size:
            ax.hlines(float(np.median(vals)), i + 0.8, i + 1.2, color="#b22222", linewidth=2.0)
    ax.set_xticks(range(1, len(dsets) + 1), dsets)
    ax.set_ylabel("Slope (uV per load unit)")
    ax.set_title("Per-subject P3-load slopes")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_delta(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    dsets = sorted(df["dataset_id"].unique().tolist())
    for i, ds in enumerate(dsets):
        vals = pd.to_numeric(df.loc[df["dataset_id"] == ds, "delta_uv_high_minus_low"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        x = np.full(vals.shape, i + 1, dtype=float)
        jitter = np.linspace(-0.18, 0.18, num=len(vals), dtype=float) if len(vals) > 1 else np.asarray([0.0])
        ax.scatter(x + jitter, vals, alpha=0.45, s=15, color="#224b8f")
        if vals.size:
            ax.hlines(float(np.median(vals)), i + 0.8, i + 1.2, color="#ff7f11", linewidth=2.0)
    ax.set_xticks(range(1, len(dsets) + 1), dsets)
    ax.set_ylabel("Delta uV (highest load - lowest load)")
    ax.set_title("Load contrast effect sizes")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_load_curve(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    dsets = sorted(df["dataset_id"].unique().tolist())
    colors = ["#004e64", "#9fffcb", "#ff7f11", "#7f4f24"]
    for i, ds in enumerate(dsets):
        g = df[df["dataset_id"] == ds].sort_values("memory_load")
        x = pd.to_numeric(g["memory_load"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g["mean_p3_uv"], errors="coerce").to_numpy(dtype=float)
        e = pd.to_numeric(g["sem_p3_uv"], errors="coerce").to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.0, color=colors[i % len(colors)], label=ds)
        ax.fill_between(x, y - e, y + e, alpha=0.18, color=colors[i % len(colors)])
    ax.set_xlabel("Memory load")
    ax.set_ylabel("Mean P3 amplitude (uV)")
    ax.set_title("Grand-average load curves (window means)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets = _split_csv(args.datasets)
    rng = np.random.default_rng(int(args.seed))

    subj_rows: List[Dict[str, Any]] = []
    load_curve_rows: List[Dict[str, Any]] = []

    for ds in datasets:
        files = _iter_dataset_files(args.features_root, ds)
        if not files:
            continue

        frames = [_load_h5_trial_frame(fp, ds) for fp in files]
        data = pd.concat([f for f in frames if not f.empty], axis=0, ignore_index=True) if frames else pd.DataFrame()
        if data.empty:
            continue

        for sid, g in data.groupby("subject_key"):
            gg = g[["memory_load", "p3_uv"]].dropna()
            if len(gg) < int(args.min_trials):
                continue
            load = pd.to_numeric(gg["memory_load"], errors="coerce").to_numpy(dtype=float)
            p3 = pd.to_numeric(gg["p3_uv"], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(load) & np.isfinite(p3)
            load = load[m]
            p3 = p3[m]
            if load.size < int(args.min_trials) or np.unique(load).size < 2:
                continue

            slope = _robust_slope(load, p3)
            lo = float(np.nanmin(load))
            hi = float(np.nanmax(load))
            low_mean = float(np.nanmean(p3[load == lo])) if np.any(load == lo) else float("nan")
            high_mean = float(np.nanmean(p3[load == hi])) if np.any(load == hi) else float("nan")
            delta = high_mean - low_mean if np.isfinite(low_mean) and np.isfinite(high_mean) else float("nan")

            subj_rows.append(
                {
                    "dataset_id": ds,
                    "subject_key": sid,
                    "n_trials": int(load.size),
                    "slope_uv_per_load": slope,
                    "delta_uv_high_minus_low": delta,
                    "load_low": lo,
                    "load_high": hi,
                }
            )

        for load_level, g in data.groupby("memory_load"):
            vals = pd.to_numeric(g["p3_uv"], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            load_curve_rows.append(
                {
                    "dataset_id": ds,
                    "memory_load": float(load_level),
                    "mean_p3_uv": float(np.mean(vals)),
                    "sem_p3_uv": float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0,
                    "n_trials": int(vals.size),
                    "n_subjects": int(data.loc[data["memory_load"] == load_level, "subject_key"].nunique()),
                }
            )

    subj_df = pd.DataFrame(subj_rows)
    curve_df = pd.DataFrame(load_curve_rows)

    subj_csv = args.out_dir / "per_subject_effect_sizes.csv"
    curve_csv = args.out_dir / "grand_average_by_load.csv"
    subj_df.to_csv(subj_csv, index=False)
    curve_df.to_csv(curve_csv, index=False)

    summary_rows: List[Dict[str, Any]] = []
    if not subj_df.empty:
        for ds, g in subj_df.groupby("dataset_id"):
            slopes = pd.to_numeric(g["slope_uv_per_load"], errors="coerce").to_numpy(dtype=float)
            deltas = pd.to_numeric(g["delta_uv_high_minus_low"], errors="coerce").to_numpy(dtype=float)
            slopes = slopes[np.isfinite(slopes)]
            deltas = deltas[np.isfinite(deltas)]

            summary_rows.append(
                {
                    "dataset_id": ds,
                    "n_subjects": int(g["subject_key"].nunique()),
                    "slope_median_uv_per_load": float(np.median(slopes)) if slopes.size else float("nan"),
                    "slope_ci95_lo": _bootstrap_ci(slopes, n_boot=int(args.n_boot), seed=int(rng.integers(0, 2**31 - 1)))[0]
                    if slopes.size
                    else float("nan"),
                    "slope_ci95_hi": _bootstrap_ci(slopes, n_boot=int(args.n_boot), seed=int(rng.integers(0, 2**31 - 1)))[1]
                    if slopes.size
                    else float("nan"),
                    "delta_median_uv": float(np.median(deltas)) if deltas.size else float("nan"),
                    "delta_ci95_lo": _bootstrap_ci(deltas, n_boot=int(args.n_boot), seed=int(rng.integers(0, 2**31 - 1)))[0]
                    if deltas.size
                    else float("nan"),
                    "delta_ci95_hi": _bootstrap_ci(deltas, n_boot=int(args.n_boot), seed=int(rng.integers(0, 2**31 - 1)))[1]
                    if deltas.size
                    else float("nan"),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = args.out_dir / "effect_size_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    if not subj_df.empty:
        _plot_slopes(subj_df, args.out_dir / "FIG_slopes_uv_per_load.png")
        _plot_delta(subj_df, args.out_dir / "FIG_delta_uv_high_vs_low.png")
    if not curve_df.empty:
        _plot_load_curve(curve_df, args.out_dir / "FIG_waveforms_by_load.png")

    payload = {
        "features_root": str(args.features_root),
        "datasets": datasets,
        "n_subject_rows": int(len(subj_df)),
        "n_curve_rows": int(len(curve_df)),
        "outputs": {
            "per_subject_effect_sizes": str(subj_csv),
            "grand_average_by_load": str(curve_csv),
            "effect_size_summary": str(summary_csv),
            "figures": [
                str(args.out_dir / "FIG_slopes_uv_per_load.png"),
                str(args.out_dir / "FIG_delta_uv_high_vs_low.png"),
                str(args.out_dir / "FIG_waveforms_by_load.png"),
            ],
        },
    }

    (args.out_dir / "effect_size_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"EFFECT_SIZE_SUMMARY={args.out_dir / 'effect_size_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
