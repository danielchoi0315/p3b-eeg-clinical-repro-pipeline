#!/usr/bin/env python3
"""Clinical translation stage for NN_SOLID_1_2.

This script is intentionally fail-closed:
- If clinical root or severity sheet is missing, write STOP_REASON and exit SKIP(0).
- If event mapping is missing/ambiguous, write STOP_REASON and exit SKIP(0).

When full clinical inputs are available, this script can be extended to run feature
extraction, normative application, and clinical association models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical_bids_root", type=Path, required=True)
    ap.add_argument("--clinical_severity_csv", type=Path, required=True)
    ap.add_argument("--healthy_features_root", type=Path, required=True)
    ap.add_argument("--event_map", type=Path, default=Path("configs/clinical_event_map.yaml"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n_perm", type=int, default=20000)
    ap.add_argument("--gpu_parallel_procs", type=int, default=6)
    ap.add_argument("--cpu_workers", type=int, default=32)
    return ap.parse_args()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _event_column_snapshot(clinical_root: Path, max_files: int = 8) -> Dict[str, Any]:
    files = sorted(list(clinical_root.rglob("*_events.tsv")) + list(clinical_root.rglob("*_events.tsv.gz")))
    rows: List[Dict[str, Any]] = []
    for fp in files[:max_files]:
        try:
            df = pd.read_csv(fp, sep="\t", nrows=200)
        except Exception as exc:
            rows.append({"file": str(fp), "status": "read_error", "error": str(exc)})
            continue
        rows.append(
            {
                "file": str(fp),
                "status": "ok",
                "columns": list(df.columns),
                "n_rows_sampled": int(len(df)),
            }
        )
    return {"n_event_files": int(len(files)), "snapshots": rows}


def _write_stop_reason(out_dir: Path, reason: str, inspection: Dict[str, Any]) -> Path:
    path = out_dir / "STOP_REASON.md"
    lines = [
        "# Clinical Stage STOP_REASON",
        "",
        "## Why this stage was skipped",
        reason,
        "",
        "## Expected folder structure",
        "- `<CLINICAL_BIDS_ROOT>/dataset_or_site_id/sub-*/[ses-*/]eeg/*_eeg.*`",
        "- `<CLINICAL_BIDS_ROOT>/dataset_or_site_id/sub-*/[ses-*/]eeg/*_events.tsv[.gz]`",
        "- Optional but recommended: participant-level TSV/JSON with demographics.",
        "",
        "## Required severity CSV columns",
        "- `subject_id` (required)",
        "- Clinical outcomes (one or more), e.g. `PANSS_total`, `MMSE`, `HAMD`",
        "- `age` (recommended)",
        "- `sex` (recommended)",
        "",
        "## Expected clinical event map format",
        "- YAML with per-dataset entries:",
        "```yaml",
        "datasets:",
        "  <dataset_id>:",
        "    event_filter: \"<pandas query selecting probe events>\"",
        "    load_column: \"<column name for load>\"",
        "    load_regex: \"<optional regex if load_column is text>\"",
        "    load_sign: 1.0",
        "```",
        "",
        "## Inspection output snapshot",
        "```json",
        json.dumps(inspection, indent=2),
        "```",
    ]
    _write_text(path, "\n".join(lines) + "\n")
    return path


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inspection = _event_column_snapshot(args.clinical_bids_root) if args.clinical_bids_root.exists() else {"n_event_files": 0, "snapshots": []}
    _write_json(args.out_dir / "events_inspection.json", inspection)

    if not args.clinical_bids_root.exists():
        stop = _write_stop_reason(
            args.out_dir,
            reason=(
                f"Missing CLINICAL_BIDS_ROOT: `{args.clinical_bids_root}`. "
                "Create/populate this path with BIDS EEG clinical cohort data."
            ),
            inspection=inspection,
        )
        _write_json(
            args.out_dir / "clinical_apply_summary.json",
            {
                "status": "SKIP",
                "reason": f"clinical root missing: {args.clinical_bids_root}",
                "stop_reason": str(stop),
            },
        )
        return 0

    if not args.clinical_severity_csv.exists():
        stop = _write_stop_reason(
            args.out_dir,
            reason=(
                f"Missing CLINICAL_SEVERITY_CSV: `{args.clinical_severity_csv}`. "
                "Provide subject-level clinical outcomes keyed by subject_id."
            ),
            inspection=inspection,
        )
        _write_json(
            args.out_dir / "clinical_apply_summary.json",
            {
                "status": "SKIP",
                "reason": f"clinical severity csv missing: {args.clinical_severity_csv}",
                "stop_reason": str(stop),
            },
        )
        return 0

    try:
        sev = pd.read_csv(args.clinical_severity_csv)
    except Exception as exc:
        stop = _write_stop_reason(
            args.out_dir,
            reason=f"Failed to read clinical severity CSV `{args.clinical_severity_csv}`: {exc}",
            inspection=inspection,
        )
        _write_json(args.out_dir / "clinical_apply_summary.json", {"status": "SKIP", "reason": str(exc), "stop_reason": str(stop)})
        return 0

    required_cols = {"subject_id"}
    if not required_cols.issubset(set(sev.columns)):
        stop = _write_stop_reason(
            args.out_dir,
            reason=(
                f"Clinical severity CSV is missing required columns: {sorted(required_cols - set(sev.columns))}. "
                f"Present columns: {list(sev.columns)}"
            ),
            inspection=inspection,
        )
        _write_json(args.out_dir / "clinical_apply_summary.json", {"status": "SKIP", "reason": "severity columns missing", "stop_reason": str(stop)})
        return 0

    if not args.event_map.exists():
        stop = _write_stop_reason(
            args.out_dir,
            reason=f"Clinical event map missing: `{args.event_map}`",
            inspection=inspection,
        )
        _write_json(args.out_dir / "clinical_apply_summary.json", {"status": "SKIP", "reason": "clinical event map missing", "stop_reason": str(stop)})
        return 0

    try:
        emap = yaml.safe_load(args.event_map.read_text(encoding="utf-8"))
    except Exception as exc:
        stop = _write_stop_reason(args.out_dir, reason=f"Clinical event map parse failed: {exc}", inspection=inspection)
        _write_json(args.out_dir / "clinical_apply_summary.json", {"status": "SKIP", "reason": "clinical event map parse failed", "stop_reason": str(stop)})
        return 0

    datasets_cfg = (emap or {}).get("datasets", {}) if isinstance(emap, dict) else {}
    if not isinstance(datasets_cfg, dict) or not datasets_cfg:
        stop = _write_stop_reason(
            args.out_dir,
            reason=(
                "Clinical event mapping is missing dataset-level entries. "
                "Provide explicit event_filter/load_column per clinical dataset to avoid ambiguous parsing."
            ),
            inspection=inspection,
        )
        _write_json(args.out_dir / "clinical_apply_summary.json", {"status": "SKIP", "reason": "missing explicit clinical mappings", "stop_reason": str(stop)})
        return 0

    # Placeholder outputs for environments where clinical datasets exist but
    # full extraction/model stages are not configured yet.
    (args.out_dir / "deviation_scores.csv").write_text("subject_id,deviation_score\n", encoding="utf-8")
    (args.out_dir / "clinical_regression_results.csv").write_text("outcome,beta,p_value,q_value\n", encoding="utf-8")
    (args.out_dir / "group_auc.csv").write_text("label,auc,ci95_lo,ci95_hi\n", encoding="utf-8")

    _write_text(
        args.out_dir / "events_inspection.md",
        "# Clinical Events Inspection\n\n"
        "Clinical root exists and event files were detected.\n"
        "This run preserved fail-closed behavior and emitted placeholder outputs only;\n"
        "extend `scripts/clinical_apply.py` with site-specific extraction/model commands when clinical mapping is finalized.\n",
    )

    _write_json(
        args.out_dir / "clinical_apply_summary.json",
        {
            "status": "SKIP",
            "reason": "clinical pipeline scaffold present; full extraction/model execution requires finalized site-specific mapping",
            "n_event_files": inspection.get("n_event_files", 0),
            "severity_columns": list(sev.columns),
            "event_map": str(args.event_map),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
