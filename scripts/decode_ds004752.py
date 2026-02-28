#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


def _write(path: Path, txt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _stop(path: Path, reason: str, diag: Dict[str, Any]) -> None:
    _write(
        path,
        "\n".join(
            [
                "# STOP_REASON_ds004752",
                "",
                "## Why skipped",
                reason,
                "",
                "## Diagnostics",
                "```json",
                json.dumps(diag, indent=2),
                "```",
                "",
            ]
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--out_yaml", type=Path, required=True)
    ap.add_argument("--out_summary", type=Path, required=True)
    ap.add_argument("--out_candidate", type=Path, required=True)
    ap.add_argument("--stop_reason", type=Path, required=True)
    args = ap.parse_args()

    events = sorted(args.dataset_root.rglob("*_task-verbalWM*_events.tsv"))
    rows: List[Dict[str, Any]] = []
    cols_seen = set()
    subj_ok = set()
    modality_counts = {"eeg": 0, "ieeg": 0}

    for p in events:
        mod = "ieeg" if "/ieeg/" in p.as_posix() else ("eeg" if "/eeg/" in p.as_posix() else "other")
        if mod in modality_counts:
            modality_counts[mod] += 1
        try:
            df = pd.read_csv(p, sep="\t")
        except Exception:
            continue
        cols_seen.update(df.columns.tolist())
        sid = p.name.split("_", 1)[0]
        has_cols = all(c in df.columns for c in ["onset", "duration", "SetSize"])
        n = int(len(df))
        n_finite = int(pd.to_numeric(df.get("SetSize"), errors="coerce").notna().sum()) if "SetSize" in df.columns else 0
        if has_cols and n_finite >= max(10, int(0.6 * max(1, n))):
            subj_ok.add(sid)
        rows.append(
            {
                "events_file": str(p),
                "modality": mod,
                "n_rows": n,
                "has_onset": bool("onset" in df.columns),
                "has_duration": bool("duration" in df.columns),
                "has_SetSize": bool("SetSize" in df.columns),
                "has_ResponseTime": bool("ResponseTime" in df.columns),
                "has_Correct": bool("Correct" in df.columns),
                "setsize_nonmissing": n_finite,
            }
        )

    cand_df = pd.DataFrame(rows)
    cand_df.to_csv(args.out_candidate, index=False)

    diag = {
        "n_events_files": len(events),
        "modalities": modality_counts,
        "columns_seen": sorted(cols_seen),
        "n_subjects_with_valid_setsize": len(subj_ok),
    }

    if len(events) == 0:
        reason = "no verbalWM events files found"
        _stop(args.stop_reason, reason, diag)
        _write_json(args.out_summary, {"dataset_id": "ds004752", "status": "SKIP", "reason": reason, "diagnostics": diag})
        return 0

    required = {"onset", "duration", "SetSize"}
    if not required.issubset(cols_seen) or len(subj_ok) < 10:
        reason = "missing required columns or insufficient subjects for stable load mapping"
        _stop(args.stop_reason, reason, diag)
        _write_json(args.out_summary, {"dataset_id": "ds004752", "status": "SKIP", "reason": reason, "diagnostics": diag})
        return 0

    mapping = {
        "event_filter": "SetSize.notna()",
        "load_column": "SetSize",
        "rt_column": "ResponseTime",
        "task_contains": "verbalWM",
        "load_sign": 1.0,
    }
    payload = {"defaults": {}, "datasets": {"ds004752": mapping}}
    args.out_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.out_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    _write_json(args.out_summary, {"dataset_id": "ds004752", "status": "PASS", "reason": "", "mapping": mapping, "diagnostics": diag})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
