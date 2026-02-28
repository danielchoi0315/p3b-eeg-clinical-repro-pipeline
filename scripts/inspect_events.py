#!/usr/bin/env python3
"""Inspect Sternberg-family events tables and emit audit markdown reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.lawc_audit import RT_COLUMNS_PRIORITY, load_lawc_event_map


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--event_map", type=Path, default=Path("configs/lawc_event_map.yaml"))
    ap.add_argument("--datasets", type=str, default="ds005095,ds003655,ds004117")
    ap.add_argument("--out_dir", type=Path, required=True)
    return ap.parse_args()


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _find_event_files(dataset_dir: Path) -> List[Path]:
    files = sorted(list(dataset_dir.rglob("*_events.tsv")) + list(dataset_dir.rglob("*_events.tsv.gz")))
    return [p for p in files if p.is_file()]


def _sample_tables(event_files: List[Path], max_files: int = 8) -> List[pd.DataFrame]:
    out = []
    for p in event_files[:max_files]:
        try:
            df = pd.read_csv(p, sep="\t")
        except Exception:
            continue
        df["__source__"] = str(p)
        out.append(df)
    return out


def _likely_categorical_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c.startswith("__"):
            continue
        if df[c].dtype == object:
            cols.append(c)
            continue
        nunq = int(df[c].nunique(dropna=True))
        if nunq <= 20:
            cols.append(c)
    return cols


def _auto_suggestions(df: pd.DataFrame) -> Dict[str, str]:
    suggestions: Dict[str, str] = {}

    event_candidates: List[str] = []
    if "trial_type" in df.columns and df["trial_type"].astype(str).str.contains("probe", case=False, na=False).any():
        event_candidates.append("trial_type.str.contains('^probe:', case=False, na=False)")
    if "task_role" in df.columns and df["task_role"].astype(str).str.contains("probe", case=False, na=False).any():
        event_candidates.append("task_role in ['probe_target', 'probe_not_shown']")
    if "value" in df.columns:
        vv = pd.to_numeric(df["value"], errors="coerce")
        probe_like = vv.isin([31, 32, 61, 62, 91, 92, 121, 122, 151, 152]).mean()
        if probe_like > 0.05:
            event_candidates.append("value in [31,32,61,62,91,92,121,122,151,152]")

    if len(event_candidates) == 1:
        suggestions["event_filter"] = event_candidates[0]
    elif len(event_candidates) > 1:
        suggestions["event_filter_candidates"] = "; ".join(event_candidates)

    load_candidates = [
        c
        for c in ["memory_load", "set_size", "load", "memory_cond", "value", "trial_type"]
        if c in df.columns
    ]
    if len(load_candidates) == 1:
        suggestions["load_column"] = load_candidates[0]
    elif len(load_candidates) > 1:
        suggestions["load_column_candidates"] = ", ".join(load_candidates)

    return suggestions


def _render_report(dataset_id: str, event_files: List[Path], sample_df: pd.DataFrame, spec: Dict[str, str], suggestions: Dict[str, str]) -> str:
    lines: List[str] = []
    lines.append(f"# Events Inspection: {dataset_id}")
    lines.append("")
    lines.append(f"- Event files found: {len(event_files)}")
    lines.append(f"- Sampled files: {min(len(event_files), 8)}")
    lines.append("")

    lines.append("## Config Mapping")
    if spec:
        lines.append(f"- event_filter: `{spec.get('event_filter', '<missing>')}`")
        lines.append(f"- load_column: `{spec.get('load_column', '<missing>')}`")
        lines.append(f"- load_sign: `{spec.get('load_sign', 1.0)}`")
        lines.append(f"- rt_column: `{spec.get('rt_column', '<none>')}`")
        lines.append(f"- rt_strategy: `{spec.get('rt_strategy', '<none>')}`")
    else:
        lines.append("- No explicit dataset mapping present in `configs/lawc_event_map.yaml`.")
    lines.append("")

    lines.append("## Event Columns")
    lines.append("- " + ", ".join(list(sample_df.columns)))
    lines.append("")

    cat_cols = _likely_categorical_columns(sample_df)
    lines.append("## Categorical Values (sample)")
    if not cat_cols:
        lines.append("- <none>")
    for c in cat_cols[:10]:
        vals = sample_df[c].dropna().astype(str).unique().tolist()[:20]
        lines.append(f"- `{c}`: {vals}")
    lines.append("")

    lines.append("## Missingness Summary")
    for c in ["memory_load", "set_size", "load", "memory_cond", "value"] + RT_COLUMNS_PRIORITY:
        if c not in sample_df.columns:
            continue
        nonmissing = float(pd.to_numeric(sample_df[c], errors="coerce").notna().mean())
        lines.append(f"- `{c}` non-missing numeric fraction: {nonmissing:.3f}")
    lines.append("")

    lines.append("## Suggestions")
    if suggestions:
        for k, v in suggestions.items():
            lines.append(f"- {k}: `{v}`")
    else:
        lines.append("- <none>")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    datasets = _split_csv(args.datasets)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    event_map = load_lawc_event_map(args.event_map)
    map_datasets = event_map.get("datasets", {}) or {}

    failures: List[str] = []
    summary: Dict[str, Dict[str, str]] = {}

    for ds in datasets:
        ds_dir = args.data_root / ds
        if not ds_dir.exists():
            failures.append(f"{ds}: dataset directory missing at {ds_dir}")
            continue

        event_files = _find_event_files(ds_dir)
        if not event_files:
            failures.append(f"{ds}: no *_events.tsv files found")
            continue

        tables = _sample_tables(event_files)
        if not tables:
            failures.append(f"{ds}: failed to read sampled events files")
            continue

        sample_df = pd.concat(tables, axis=0, ignore_index=True)
        spec = dict(map_datasets.get(ds, {}))
        suggestions = _auto_suggestions(sample_df)

        # Fail-closed ambiguity rules.
        if not spec:
            failures.append(
                f"{ds}: missing explicit mapping in {args.event_map}; suggestions written to report."
            )
        else:
            ef = str(spec.get("event_filter", "")).strip()
            lc = str(spec.get("load_column", "")).strip()
            if not ef or not lc:
                failures.append(f"{ds}: mapping must define event_filter and load_column")
            else:
                try:
                    selected = sample_df.query(ef, engine="python")
                except Exception as exc:
                    failures.append(f"{ds}: event_filter query failed: {exc}")
                    selected = sample_df.iloc[0:0]
                if selected.empty:
                    failures.append(f"{ds}: event_filter selects zero rows in sampled files")
                if lc and lc not in sample_df.columns:
                    failures.append(f"{ds}: load_column '{lc}' missing from sampled columns")

        report = _render_report(ds, event_files, sample_df, spec, suggestions)
        out_path = args.out_dir / f"events_inspection_{ds}.md"
        out_path.write_text(report, encoding="utf-8")
        summary[ds] = {
            "event_files": str(len(event_files)),
            "report": str(out_path),
        }

    (args.out_dir / "events_inspection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if failures:
        details = "\n".join(f"- {x}" for x in failures)
        raise RuntimeError(f"Events inspection fail-closed errors:\n{details}")

    print("Events inspection complete:")
    for ds in datasets:
        print(f"  - {ds}: {args.out_dir / f'events_inspection_{ds}.md'}")


if __name__ == "__main__":
    main()
