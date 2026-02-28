#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class CandidateResult:
    column: str
    status: str
    n_files: int
    n_subjects_all_three: int
    n_subjects_any_stim: int
    standard_count: int
    target_count: int
    novel_count: int
    standard_values: List[str]
    target_values: List[str]
    novel_values: List[str]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _subject_id_from_path(path: Path) -> str:
    m = re.search(r"sub-([A-Za-z0-9]+)", str(path))
    return m.group(1) if m else ""


def _norm(v: object) -> str:
    return str(v).strip()


def _classify_value(txt: str) -> str:
    s = txt.strip().lower()
    if re.search(r"\bstandard\b", s):
        return "standard"
    if re.search(r"\btarget\b", s):
        return "target"
    if re.search(r"\bnovel\b", s):
        return "novel"
    return ""


def _evaluate_column(events: List[Tuple[Path, pd.DataFrame]], col: str) -> CandidateResult:
    values_by_class = {"standard": set(), "target": set(), "novel": set()}
    counts = Counter()
    subj_counts = defaultdict(Counter)
    files_with_any = 0

    for fp, df in events:
        if col not in df.columns:
            continue
        subj = _subject_id_from_path(fp)
        c_file = Counter()
        for raw in df[col].dropna().astype(str):
            cls = _classify_value(raw)
            if not cls:
                continue
            val = _norm(raw)
            values_by_class[cls].add(val)
            counts[cls] += 1
            c_file[cls] += 1
            if subj:
                subj_counts[subj][cls] += 1
        if sum(c_file.values()) > 0:
            files_with_any += 1

    n_subj_any = sum(1 for _, c in subj_counts.items() if sum(c.values()) > 0)
    n_subj_all3 = sum(1 for _, c in subj_counts.items() if c["standard"] > 0 and c["target"] > 0 and c["novel"] > 0)

    ok_exact_three = bool(values_by_class["standard"] and values_by_class["target"] and values_by_class["novel"])
    ok_ratio = counts["standard"] > counts["target"] and counts["standard"] > counts["novel"]
    ok_subject_stability = n_subj_all3 >= 10

    status = "PASS" if ok_exact_three and ok_ratio and ok_subject_stability else "FAIL"
    return CandidateResult(
        column=col,
        status=status,
        n_files=int(files_with_any),
        n_subjects_all_three=int(n_subj_all3),
        n_subjects_any_stim=int(n_subj_any),
        standard_count=int(counts["standard"]),
        target_count=int(counts["target"]),
        novel_count=int(counts["novel"]),
        standard_values=sorted(values_by_class["standard"]),
        target_values=sorted(values_by_class["target"]),
        novel_values=sorted(values_by_class["novel"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--stop_reason", type=Path, required=True)
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_reason: Path = args.stop_reason
    mapping_json = out_dir / "pd_mapping_decode.json"
    summary_json = out_dir / "mapping_decode_summary.json"

    events_paths = sorted(args.dataset_root.rglob("*_events.tsv"))
    if not events_paths:
        reason = "No *_events.tsv files found"
        _write_text(stop_reason, f"# STOP_REASON\n\n{reason}\n")
        payload = {"status": "SKIP", "reason": reason, "n_event_files": 0}
        _write_json(summary_json, payload)
        _write_json(mapping_json, payload)
        return 0

    events: List[Tuple[Path, pd.DataFrame]] = []
    missing_semantics: List[str] = []
    for p in events_paths:
        try:
            df = pd.read_csv(p, sep="\t")
        except Exception:
            continue
        cols = {str(c) for c in df.columns}
        if "onset" not in cols or "duration" not in cols:
            missing_semantics.append(str(p))
        events.append((p, df))

    if missing_semantics:
        reason = "BIDS semantics ambiguous: onset/duration missing in one or more events files"
        _write_text(
            stop_reason,
            "\n".join([
                "# STOP_REASON",
                "",
                reason,
                "",
                "## Files",
                *[f"- {x}" for x in missing_semantics[:50]],
            ])
            + "\n",
        )
        payload = {"status": "SKIP", "reason": reason, "n_event_files": len(events_paths)}
        _write_json(summary_json, payload)
        _write_json(mapping_json, payload)
        return 0

    candidate_cols = [
        "trial_type",
        "event_type",
        "stim_type",
        "value",
        "trial",
        "event",
        "stimulus",
    ]
    present_cols = sorted({c for _, df in events for c in df.columns})
    candidate_cols = [c for c in candidate_cols if c in present_cols]

    results: List[CandidateResult] = [_evaluate_column(events, c) for c in candidate_cols]
    table = pd.DataFrame([r.__dict__ for r in results])
    table.to_csv(out_dir / "CANDIDATE_TABLE.csv", index=False)

    passed = [r for r in results if r.status == "PASS"]
    if not passed:
        reason = "No candidate column passed fail-closed oddball decode criteria"
        details = {
            "criteria": {
                "exact_three_categories": True,
                "plausible_ratio": "standard_count > target_count and standard_count > novel_count",
                "subject_stability": "subjects_with_all_three >= 10",
            },
            "candidate_results": [r.__dict__ for r in results],
        }
        _write_text(
            stop_reason,
            "\n".join([
                "# STOP_REASON",
                "",
                reason,
                "",
                "## Candidate evaluation",
                "```json",
                json.dumps(details, indent=2),
                "```",
            ])
            + "\n",
        )
        payload = {
            "status": "SKIP",
            "reason": reason,
            "n_event_files": len(events_paths),
            "candidate_table": str(out_dir / "CANDIDATE_TABLE.csv"),
        }
        _write_json(summary_json, payload)
        _write_json(mapping_json, payload)
        return 0

    # Preference: trial_type first, then highest n_subjects_all_three.
    passed.sort(key=lambda r: (0 if r.column == "trial_type" else 1, -r.n_subjects_all_three, -r.standard_count))
    best = passed[0]

    mapping_payload = {
        "status": "PASS",
        "dataset_id": "ds003490",
        "column": best.column,
        "stim_standard_values": best.standard_values,
        "stim_target_values": best.target_values,
        "stim_novel_values": best.novel_values,
        "counts": {
            "standard": best.standard_count,
            "target": best.target_count,
            "novel": best.novel_count,
        },
        "n_event_files": int(len(events_paths)),
        "n_subjects_any_stim": int(best.n_subjects_any_stim),
        "n_subjects_all_three": int(best.n_subjects_all_three),
        "criteria": {
            "ratio_ok": True,
            "stability_subjects_ge_10": True,
            "exact_three_classes": True,
        },
    }
    _write_json(mapping_json, mapping_payload)

    summary_payload = {
        "status": "PASS",
        "reason": "",
        "mapping_json": str(mapping_json),
        "candidate_table": str(out_dir / "CANDIDATE_TABLE.csv"),
        **mapping_payload,
    }
    _write_json(summary_json, summary_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
