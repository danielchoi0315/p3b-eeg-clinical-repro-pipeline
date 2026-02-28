#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
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
                "# STOP_REASON_ds007262",
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


def _start_val(x: str) -> float:
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)\s*-", str(x))
    return float(m.group(1)) if m else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--out_yaml", type=Path, required=True)
    ap.add_argument("--out_summary", type=Path, required=True)
    ap.add_argument("--out_candidate", type=Path, required=True)
    ap.add_argument("--stop_reason", type=Path, required=True)
    args = ap.parse_args()

    events = sorted(args.dataset_root.rglob("*_task-arithmetic_events.tsv"))
    rows: List[Dict[str, Any]] = []
    cols_seen = set()
    subj_levels: Dict[str, List[str]] = {}

    for p in events:
        try:
            df = pd.read_csv(p, sep="\t")
        except Exception:
            continue
        cols_seen.update(df.columns.tolist())
        sid = p.name.split("_", 1)[0]

        lev = []
        if "difficulty_range" in df.columns:
            lev = sorted(set(df["difficulty_range"].dropna().astype(str).tolist()))
            lev = [x for x in lev if re.match(r"^[0-9]+(?:\.[0-9]+)?-[0-9]+(?:\.[0-9]+)?$", x)]
        subj_levels[sid] = lev

        rows.append(
            {
                "events_file": str(p),
                "n_rows": int(len(df)),
                "has_difficulty_range": bool("difficulty_range" in df.columns),
                "has_trial_type": bool("trial_type" in df.columns),
                "n_levels_difficulty_range": int(len(lev)),
                "n_accuracy_nonmissing": int(pd.to_numeric(df.get("response_accuracy"), errors="coerce").notna().sum()) if "response_accuracy" in df.columns else 0,
            }
        )

    pd.DataFrame(rows).to_csv(args.out_candidate, index=False)

    level_union = sorted(set(sum(subj_levels.values(), [])), key=_start_val)
    level_map = {lv: i + 1 for i, lv in enumerate(level_union)}
    n_subj_good = sum(1 for v in subj_levels.values() if len(v) >= 4)
    diag = {
        "n_events_files": len(events),
        "columns_seen": sorted(cols_seen),
        "level_union": level_union,
        "n_subjects_good": n_subj_good,
    }

    if len(events) == 0:
        reason = "no arithmetic events files found"
        _stop(args.stop_reason, reason, diag)
        _write_json(args.out_summary, {"dataset_id": "ds007262", "status": "SKIP", "reason": reason, "diagnostics": diag})
        return 0

    if "difficulty_range" not in cols_seen or len(level_union) < 4 or n_subj_good < 8:
        reason = "difficulty_range mapping is ambiguous or unstable across subjects"
        _stop(args.stop_reason, reason, diag)
        _write_json(args.out_summary, {"dataset_id": "ds007262", "status": "SKIP", "reason": reason, "diagnostics": diag})
        return 0

    mapping = {
        "event_filter": "difficulty_range.notna()",
        "load_column": "difficulty_range",
        "load_value_map": level_map,
        "rt_strategy": "auto",
        "task_contains": "arithmetic",
        "load_sign": 1.0,
    }
    payload = {"defaults": {}, "datasets": {"ds007262": mapping}}
    args.out_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.out_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    _write_json(args.out_summary, {"dataset_id": "ds007262", "status": "PASS", "reason": "", "mapping": mapping, "diagnostics": diag})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
