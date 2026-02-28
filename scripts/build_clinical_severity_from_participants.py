#!/usr/bin/env python3
"""Build clinical_severity.csv from clinical dataset participants tables."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical_bids_root", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--summary_json", type=Path, default=None)
    return ap.parse_args()


def _norm_subject_id(v: Any) -> str:
    s = str(v).strip()
    if not s:
        return ""
    if s.startswith("sub-"):
        return s[len("sub-") :]
    return s


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _map_group_value(v: Any) -> Tuple[float, str]:
    if pd.isna(v):
        return float("nan"), "missing"
    s = str(v).strip()
    if not s:
        return float("nan"), "empty"

    num = pd.to_numeric(pd.Series([s]), errors="coerce").iloc[0]
    if pd.notna(num):
        f = float(num)
        if np.isfinite(f) and f in {0.0, 1.0}:
            return f, "numeric_binary"
        # non-binary numeric label is ambiguous for case/control endpoint
        return float("nan"), f"numeric_nonbinary:{f}"

    sl = s.lower()
    if any(tok in sl for tok in ["control", "hc", "healthy", "sham", "no_tbi", "non_tbi", "non-tbi"]):
        return 0.0, "string_control"
    if any(tok in sl for tok in ["patient", "case", "mtbi", "mild tbi", "tbi", "injury", "injured"]):
        return 1.0, "string_case"

    return float("nan"), f"unmapped:{s}"


def _collect_phenotype(dataset_root: Path) -> pd.DataFrame:
    phen_root = dataset_root / "phenotype"
    if not phen_root.exists():
        return pd.DataFrame()

    frames = []
    for fp in sorted(phen_root.glob("*.tsv")):
        try:
            df = pd.read_csv(fp, sep="\t")
        except Exception:
            continue
        sid_col = _pick_col(df, ["participant_id", "subject_id", "subject", "sub_id", "id"])
        if sid_col is None:
            continue
        df = df.copy()
        df["subject_id"] = df[sid_col].map(_norm_subject_id)
        keep_cols = [c for c in df.columns if c not in {sid_col, "subject_id"}]
        if not keep_cols:
            continue
        sub = df[["subject_id"] + keep_cols].copy()
        sub = sub.drop_duplicates(subset=["subject_id"], keep="first")
        # Prefix phenotypic columns with filename stem to avoid collisions.
        stem = fp.stem.replace(" ", "_")
        ren = {c: f"{stem}__{c}" for c in keep_cols}
        sub = sub.rename(columns=ren)
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="subject_id", how="outer")
    return out


def main() -> int:
    args = parse_args()
    if not args.clinical_bids_root.exists():
        raise RuntimeError(f"clinical_bids_root does not exist: {args.clinical_bids_root}")

    datasets = sorted([p for p in args.clinical_bids_root.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if not datasets:
        raise RuntimeError(f"no dataset directories found under {args.clinical_bids_root}")

    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "clinical_bids_root": str(args.clinical_bids_root),
        "datasets": [],
        "warnings": [],
    }

    for ds_root in datasets:
        dataset_id = ds_root.name
        p_tsv = ds_root / "participants.tsv"
        if not p_tsv.exists():
            summary["warnings"].append(f"{dataset_id}: missing participants.tsv")
            summary["datasets"].append(
                {
                    "dataset_id": dataset_id,
                    "status": "SKIP",
                    "reason": "participants.tsv missing",
                }
            )
            continue

        try:
            pdf = pd.read_csv(p_tsv, sep="\t")
        except Exception as exc:
            summary["warnings"].append(f"{dataset_id}: failed reading participants.tsv: {exc}")
            summary["datasets"].append(
                {
                    "dataset_id": dataset_id,
                    "status": "SKIP",
                    "reason": f"participants read error: {exc}",
                }
            )
            continue

        sid_col = _pick_col(pdf, ["participant_id", "subject_id", "subject", "sub_id", "id"])
        if sid_col is None:
            summary["warnings"].append(f"{dataset_id}: could not find subject id column in participants.tsv")
            summary["datasets"].append(
                {
                    "dataset_id": dataset_id,
                    "status": "SKIP",
                    "reason": "subject id column missing",
                }
            )
            continue

        group_candidates = ["Group", "group", "GROUP", "diagnosis", "dx", "case_control", "condition", "status"]
        group_col = _pick_col(pdf, group_candidates)
        if group_col is None:
            summary["warnings"].append(f"{dataset_id}: no group-like column found in participants.tsv")

        age_col = _pick_col(pdf, ["age", "Age", "AGE"])
        sex_col = _pick_col(pdf, ["sex", "Sex", "SEX", "gender", "Gender"])

        pheno = _collect_phenotype(ds_root)

        df = pdf.copy()
        df["subject_id"] = df[sid_col].map(_norm_subject_id)
        df = df[df["subject_id"].astype(str).str.len() > 0].copy()

        if group_col is not None:
            mapped = df[group_col].map(_map_group_value)
            df["group"] = [float(t[0]) for t in mapped]
            df["group_map_note"] = [str(t[1]) for t in mapped]
        else:
            df["group"] = np.nan
            df["group_map_note"] = "missing_group_column"

        out = pd.DataFrame(
            {
                "dataset_id": dataset_id,
                "subject_id": df["subject_id"].astype(str),
                "group": pd.to_numeric(df["group"], errors="coerce"),
                "age": pd.to_numeric(df[age_col], errors="coerce") if age_col is not None else np.nan,
                "sex": df[sex_col].astype(str) if sex_col is not None else "",
                "session": "",
            }
        )

        if not pheno.empty:
            out = out.merge(pheno, on="subject_id", how="left")

        # dataset-level summary
        n_total = int(len(out))
        n_group_finite = int(np.isfinite(pd.to_numeric(out["group"], errors="coerce")).sum())
        n_controls = int((pd.to_numeric(out["group"], errors="coerce") == 0).sum())
        n_cases = int((pd.to_numeric(out["group"], errors="coerce") == 1).sum())

        summary["datasets"].append(
            {
                "dataset_id": dataset_id,
                "status": "PASS" if n_group_finite > 0 else "SKIP",
                "participants_tsv": str(p_tsv),
                "group_column": group_col,
                "subject_column": sid_col,
                "age_column": age_col,
                "sex_column": sex_col,
                "n_total": n_total,
                "n_group_finite": n_group_finite,
                "n_controls": n_controls,
                "n_cases": n_cases,
                "group_map_notes": sorted(set(df["group_map_note"].astype(str).tolist()))[:20],
            }
        )

        rows.extend(out.to_dict(orient="records"))

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise RuntimeError("No clinical participant rows collected from participants.tsv")

    out_df = out_df.drop_duplicates(subset=["dataset_id", "subject_id"], keep="first").reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    summary["out_csv"] = str(args.out_csv)
    summary["n_rows"] = int(len(out_df))
    summary["n_datasets"] = int(out_df["dataset_id"].nunique()) if "dataset_id" in out_df.columns else 0

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"WROTE={args.out_csv}")
    if args.summary_json is not None:
        print(f"SUMMARY={args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
