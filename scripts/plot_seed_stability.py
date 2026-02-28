#!/usr/bin/env python3
"""Plot many-seed module04 stability metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module04_root", type=Path, required=True, help="Path containing seed_<k>/ outputs")
    ap.add_argument("--out_png", type=Path, default=None)
    ap.add_argument("--out_csv", type=Path, default=None)
    return ap.parse_args()


def _read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect_rows(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed_dir in sorted(root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        seed = seed_dir.name.replace("seed_", "")
        run_id = f"seed_{seed}"
        norm_path = seed_dir / "reports" / "normative" / run_id / "normative_metrics.json"
        rt_path = seed_dir / "reports" / "normative" / run_id / "rt_linkage_summary.json"

        norm = _read_json(norm_path) or {}
        rt = _read_json(rt_path) or {}
        healthy = norm.get("healthy") or {}
        cal = healthy.get("calibration") or {}
        stab = healthy.get("z_stability") or {}
        linkage = ((rt.get("linkage") or {}).get("healthy") or {})

        rows.append(
            {
                "seed": int(seed) if str(seed).isdigit() else seed,
                "healthy_nll": float(healthy.get("nll", np.nan)),
                "healthy_z_std": float(cal.get("z_std", np.nan)),
                "healthy_subject_mean_std": float(stab.get("subject_mean_std", np.nan)),
                "healthy_rt_beta_margin": float(linkage.get("mean_beta_margin", np.nan)),
                "seed_effect_checksum": str(norm.get("seed_effect_checksum", "")),
            }
        )
    return pd.DataFrame(rows)


def _plot(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    metrics = [
        ("healthy_nll", "Healthy NLL"),
        ("healthy_z_std", "Healthy calibration z_std"),
        ("healthy_subject_mean_std", "Healthy subject mean z std"),
        ("healthy_rt_beta_margin", "Healthy RT beta_margin"),
    ]

    for ax, (col, title) in zip(axes, metrics):
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue
        ax.hist(vals, bins=min(30, max(8, int(np.sqrt(vals.size)))), alpha=0.75, color="#1f77b4")
        ax.axvline(float(np.mean(vals)), color="black", linestyle="--", linewidth=1.0, label="mean")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Module04 Many-Seed Stability")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_png = args.out_png or (args.module04_root / "FIG_normative_seed_stability.png")
    out_csv = args.out_csv or (args.module04_root / "seed_stability_metrics.csv")

    df = _collect_rows(args.module04_root)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    _plot(df, out_png)

    summary = {
        "n_seeds": int(len(df)),
        "n_unique_checksums": int(df["seed_effect_checksum"].nunique()) if "seed_effect_checksum" in df.columns else 0,
        "csv": str(out_csv),
        "figure": str(out_png),
    }
    (args.module04_root / "seed_stability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")
    print(f"Wrote {args.module04_root / 'seed_stability_summary.json'}")


if __name__ == "__main__":
    main()
