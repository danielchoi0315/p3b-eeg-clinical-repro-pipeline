#!/usr/bin/env python3
"""Decode auditable clinical event/load mappings for ds003523/ds005114.

Sources used (fail-closed):
- BIDS events.tsv (+ events.json sidecars when present)
- code/*.m task scripts
- code/*.xls/*.xlsx metadata inventories (optional diagnostics)

Outputs:
- configs/clinical_event_map_autogen.yaml
- <out_dir>/CANDIDATE_TABLE.csv
- <out_dir>/mapping_decode_summary.json
- <out_dir>/STOP_REASON_<dataset>.md for skipped datasets
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _event_files(dataset_root: Path) -> List[Path]:
    return sorted(list(dataset_root.rglob("*_events.tsv")) + list(dataset_root.rglob("*_events.tsv.gz")))


def _subject_from_path(path: Path) -> str:
    m = re.search(r"sub-([A-Za-z0-9]+)", str(path))
    return m.group(1) if m else ""


def _safe_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    return df.query(str(query), engine="python")


def _norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def _extract_first_int(v: Any) -> float:
    s = str(v)
    m = re.search(r"(\d+)", s)
    if m is None:
        return float("nan")
    return float(int(m.group(1)))


def _parse_matlab_num_expr(expr: str) -> List[int]:
    """Parse small MATLAB numeric expressions like:
    [51]
    [52:56]
    50+[7,12,19]
    50+[8:11,13:18]
    """
    txt = expr.strip().rstrip(";")
    txt = re.sub(r"\s+", "", txt)
    if not txt:
        return []

    m = re.fullmatch(r"(-?\d+)\+\[(.*)\]", txt)
    if m is not None:
        base = int(m.group(1))
        inner = m.group(2)
        vals = _parse_matlab_num_expr(f"[{inner}]")
        return [base + x for x in vals]

    m = re.fullmatch(r"\[(.*)\]", txt)
    if m is not None:
        out: List[int] = []
        inner = m.group(1)
        for part in inner.split(","):
            if not part:
                continue
            if ":" in part:
                a, b = part.split(":", 1)
                ai = int(a)
                bi = int(b)
                step = 1 if ai <= bi else -1
                out.extend(list(range(ai, bi + step, step)))
            else:
                out.append(int(part))
        return out

    if re.fullmatch(r"-?\d+", txt):
        return [int(txt)]
    return []


def _parse_matlab_assignment_numbers(text: str, var_name: str) -> List[int]:
    pat = re.compile(rf"{re.escape(var_name)}\s*=\s*([^;]+);", flags=re.IGNORECASE)
    m = pat.search(text)
    if m is None:
        return []
    return _parse_matlab_num_expr(m.group(1))


def _parse_trialtypes_wm_script(text: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for m in re.finditer(r"'S\s*(\d+)'\s*'([^']+)'", text):
        out[int(m.group(1))] = str(m.group(2))
    return out


@dataclass
class Candidate:
    dataset_id: str
    candidate_id: str
    source: str
    source_files: List[str]
    event_filter: str
    load_column: str
    load_value_map: Dict[str, float] = field(default_factory=dict)
    load_regex: str = ""
    rt_strategy: str = ""
    response_filter: str = ""
    response_accuracy_map: Dict[str, float] = field(default_factory=dict)
    task_relevant_value_codes: List[int] = field(default_factory=list)
    task_relevant_trialtype_regex: str = ""
    notes: str = ""


@dataclass
class CandidateEval:
    candidate: Candidate
    status: str
    reason: str
    metrics: Dict[str, Any]


def _derive_load_from_candidate(df_selected: pd.DataFrame, cand: Candidate) -> np.ndarray:
    if cand.load_column not in df_selected.columns:
        return np.full(len(df_selected), np.nan, dtype=float)

    src = df_selected[cand.load_column]
    if cand.load_value_map:
        map_norm = {_norm_text(k): float(v) for k, v in cand.load_value_map.items() if np.isfinite(float(v))}
        out = np.full(len(src), np.nan, dtype=float)
        for i, x in enumerate(src.astype(str).tolist()):
            key = _norm_text(x)
            if key in map_norm:
                out[i] = map_norm[key]
        return out

    if cand.load_regex:
        ex = src.astype(str).str.extract(str(cand.load_regex), expand=False)
        return pd.to_numeric(ex, errors="coerce").to_numpy(dtype=float)

    arr = pd.to_numeric(src, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(arr).mean() < 0.8:
        arr2 = np.asarray([_extract_first_int(x) for x in src.astype(str).tolist()], dtype=float)
        if np.isfinite(arr2).mean() > np.isfinite(arr).mean():
            arr = arr2
    return arr


def _task_relevant_mask(df: pd.DataFrame, cand: Candidate) -> np.ndarray:
    if cand.task_relevant_value_codes and "value" in df.columns:
        vals = np.asarray([_extract_first_int(x) for x in df["value"].astype(str).tolist()], dtype=float)
        codes = set(int(x) for x in cand.task_relevant_value_codes)
        return np.asarray([int(v) in codes if np.isfinite(v) else False for v in vals], dtype=bool)

    if cand.task_relevant_trialtype_regex and "trial_type" in df.columns:
        return df["trial_type"].astype(str).str.match(cand.task_relevant_trialtype_regex, case=False, na=False).to_numpy(dtype=bool)

    try:
        selected = _safe_query(df, cand.event_filter)
        return df.index.isin(selected.index).to_numpy(dtype=bool)
    except Exception:
        return np.zeros(len(df), dtype=bool)


def _evaluate_candidate(
    *,
    dataset_root: Path,
    cand: Candidate,
    sampled_subjects: Sequence[str],
    subject_to_files: Dict[str, List[Path]],
) -> CandidateEval:
    per_subject_rows: List[Dict[str, Any]] = []
    all_load: List[float] = []
    level_sets: Dict[str, Tuple[float, ...]] = {}
    freq_rows: List[Tuple[str, float, float]] = []

    for sid in sampled_subjects:
        files = subject_to_files.get(sid, [])
        if not files:
            continue
        dfs = []
        for fp in files:
            try:
                df = pd.read_csv(fp, sep="\t")
            except Exception:
                continue
            dfs.append(df)
        if not dfs:
            continue
        sdf = pd.concat(dfs, axis=0, ignore_index=True)

        try:
            selected = _safe_query(sdf, cand.event_filter).copy()
        except Exception as exc:
            return CandidateEval(
                candidate=cand,
                status="SKIP",
                reason=f"event_filter query failed: {exc}",
                metrics={"query_error": str(exc)},
            )

        task_mask = _task_relevant_mask(sdf, cand)
        n_task = int(task_mask.sum())
        n_selected = int(len(selected))
        if n_task > 0:
            n_selected_task = int(task_mask[selected.index].sum())
            coverage = float(n_selected_task / max(1, n_task))
        else:
            n_selected_task = 0
            coverage = float("nan")

        load = _derive_load_from_candidate(selected, cand)
        finite = np.isfinite(load)
        levels = tuple(sorted(set(load[finite].tolist())))
        if finite.any():
            all_load.extend(load[finite].tolist())
            level_sets[sid] = levels
            vc = pd.Series(load[finite]).value_counts(normalize=True)
            for k, p in vc.items():
                freq_rows.append((sid, float(k), float(p)))

        per_subject_rows.append(
            {
                "subject_id": sid,
                "n_task_relevant": n_task,
                "n_selected": n_selected,
                "n_selected_task": n_selected_task,
                "coverage": coverage,
                "n_load_finite": int(finite.sum()),
                "levels": json.dumps([float(x) for x in levels]),
            }
        )

    if not per_subject_rows:
        return CandidateEval(
            candidate=cand,
            status="SKIP",
            reason="no readable event tables in sampled subjects",
            metrics={},
        )

    subj_df = pd.DataFrame(per_subject_rows)
    coverage_vals = pd.to_numeric(subj_df["coverage"], errors="coerce")
    coverage_finite = coverage_vals[np.isfinite(coverage_vals)]
    n_subjects_with_task = int(np.isfinite(coverage_vals).sum())
    n_subjects_with_levels = int((pd.to_numeric(subj_df["n_load_finite"], errors="coerce") > 0).sum())

    global_levels = tuple(sorted(set(float(x) for x in all_load if np.isfinite(x))))
    n_levels = len(global_levels)

    same_levels_rate = float("nan")
    if n_subjects_with_levels > 0 and n_levels > 0:
        same = 0
        for sid, lev in level_sets.items():
            if tuple(lev) == tuple(global_levels):
                same += 1
        same_levels_rate = float(same / max(1, len(level_sets)))

    freq_median_l1 = float("nan")
    freq_p90_l1 = float("nan")
    if freq_rows and n_levels > 0:
        fdf = pd.DataFrame(freq_rows, columns=["subject_id", "level", "p"])
        wide = fdf.pivot(index="subject_id", columns="level", values="p").fillna(0.0)
        for lv in global_levels:
            if lv not in wide.columns:
                wide[lv] = 0.0
        wide = wide[list(global_levels)]
        meanp = wide.mean(axis=0)
        l1 = (wide - meanp).abs().sum(axis=1)
        freq_median_l1 = float(l1.median())
        freq_p90_l1 = float(l1.quantile(0.9))

    coverage_median = float(np.nanmedian(coverage_finite)) if len(coverage_finite) else float("nan")
    coverage_mean = float(np.nanmean(coverage_finite)) if len(coverage_finite) else float("nan")
    coverage_min = float(np.nanmin(coverage_finite)) if len(coverage_finite) else float("nan")

    gate_coverage = bool(np.isfinite(coverage_median) and coverage_median >= 0.8)
    gate_levels = bool(2 <= int(n_levels) <= 12)
    gate_subjects = bool(n_subjects_with_levels >= 20)
    gate_consistency = bool(
        np.isfinite(same_levels_rate)
        and same_levels_rate >= 0.95
        and np.isfinite(freq_p90_l1)
        and freq_p90_l1 <= 0.25
    )
    gate_all = gate_coverage and gate_levels and gate_subjects and gate_consistency

    score = float(0.0)
    if np.isfinite(coverage_median):
        score += float(coverage_median)
    if np.isfinite(same_levels_rate):
        score += float(same_levels_rate)
    if np.isfinite(freq_p90_l1):
        score += float(max(0.0, 1.0 - freq_p90_l1))

    metrics: Dict[str, Any] = {
        "n_subjects_sampled": int(len(sampled_subjects)),
        "n_subjects_with_task": int(n_subjects_with_task),
        "n_subjects_with_levels": int(n_subjects_with_levels),
        "coverage_mean": coverage_mean,
        "coverage_median": coverage_median,
        "coverage_min": coverage_min,
        "load_levels": [float(x) for x in global_levels],
        "n_load_levels": int(n_levels),
        "same_levels_rate": same_levels_rate,
        "freq_median_l1": freq_median_l1,
        "freq_p90_l1": freq_p90_l1,
        "gate_coverage": gate_coverage,
        "gate_levels": gate_levels,
        "gate_subjects": gate_subjects,
        "gate_consistency": gate_consistency,
        "gate_all": gate_all,
        "score": score,
    }

    if gate_all:
        return CandidateEval(candidate=cand, status="PASS", reason="", metrics=metrics)

    failed = []
    for g in ["coverage", "levels", "subjects", "consistency"]:
        if not metrics.get(f"gate_{g}", False):
            failed.append(g)
    reason = "failed gates: " + ",".join(failed)
    return CandidateEval(candidate=cand, status="SKIP", reason=reason, metrics=metrics)


def _write_stop_reason(path: Path, dataset_id: str, reason: str, details: Dict[str, Any]) -> None:
    lines = [
        f"# STOP_REASON {dataset_id}",
        "",
        "## Why skipped",
        reason,
        "",
        "## Diagnostics",
        "```json",
        json.dumps(details, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _discover_candidates_for_ds003523(dataset_root: Path) -> List[Candidate]:
    out: List[Candidate] = []
    script = dataset_root / "code" / "Convert2BIDS_mTBICoBRE_WM.m"
    if not script.exists():
        return out
    text = _read_text(script)
    trial_map = _parse_trialtypes_wm_script(text)
    if not trial_map:
        return out

    probe_pairs = [(code, label) for code, label in trial_map.items() if "question mark probe" in label.lower()]
    if len(probe_pairs) < 2:
        return out

    # Keep deterministic order for mapping values.
    labels = [lab for _, lab in sorted(probe_pairs, key=lambda x: x[0])]
    load_value_map = {labels[0]: 0.0, labels[1]: 1.0}
    for extra in labels[2:]:
        if extra not in load_value_map:
            load_value_map[extra] = float(len(load_value_map))

    resp_corr = [c for c, lab in trial_map.items() if "response" in lab.lower() and "correct" in lab.lower()]
    resp_inc = [c for c, lab in trial_map.items() if "response" in lab.lower() and ("inorrect" in lab.lower() or "incorrect" in lab.lower() or "error" in lab.lower())]
    acc_map: Dict[str, float] = {}
    for c in resp_corr:
        acc_map[str(int(c))] = 1.0
    for c in resp_inc:
        acc_map[str(int(c))] = 0.0

    out.append(
        Candidate(
            dataset_id="ds003523",
            candidate_id="wm_probe_trialtype_from_script",
            source="matlab_script_trialTypes+events_tsv",
            source_files=[str(script)],
            event_filter="trial_type.str.contains('Question Mark Probe', case=False, na=False)",
            load_column="trial_type",
            load_value_map=load_value_map,
            rt_strategy="next_response_any",
            response_filter="trial_type.str.contains('Response:', case=False, na=False)",
            response_accuracy_map=acc_map,
            task_relevant_value_codes=sorted([int(c) for c, _ in probe_pairs]),
            task_relevant_trialtype_regex=r"^Question Mark Probe",
            notes="Probe condition derived from Convert2BIDS_mTBICoBRE_WM.m trialTypes table.",
        )
    )
    return out


def _discover_candidates_for_ds005114(dataset_root: Path) -> List[Candidate]:
    out: List[Candidate] = []
    script = dataset_root / "code" / "STEP1_DPX_PreProc.m"
    if not script.exists():
        return out
    text = _read_text(script)

    probe_sets = {
        "Probe_aX_onset": _parse_matlab_assignment_numbers(text, "Probe_aX_onset"),
        "Probe_aY_onset": _parse_matlab_assignment_numbers(text, "Probe_aY_onset"),
        "Probe_bX_onset": _parse_matlab_assignment_numbers(text, "Probe_bX_onset"),
        "Probe_bY_onset": _parse_matlab_assignment_numbers(text, "Probe_bY_onset"),
    }
    probe_codes: List[int] = sorted(set(x for vals in probe_sets.values() for x in vals))

    # Build observed trial_type categories from events (auditable to BIDS-converted labels).
    events = _event_files(dataset_root)
    trial_types: set[str] = set()
    for fp in events[:80]:
        try:
            df = pd.read_csv(fp, sep="\t")
        except Exception:
            continue
        if "trial_type" not in df.columns:
            continue
        m = df["trial_type"].astype(str).str.match(r"^Probe_.*_onset$", case=False, na=False)
        vals = df.loc[m, "trial_type"].astype(str).unique().tolist()
        trial_types.update(vals)
    if not trial_types:
        return out

    canonical_order = ["Probe_aX_onset", "Probe_aY_onset", "Probe_bX_onset", "Probe_bY_onset"]
    present = [x for x in canonical_order if x in trial_types]
    extras = [x for x in sorted(trial_types) if x not in present]
    ordered = present + extras
    load_map = {name: float(i + 1) for i, name in enumerate(ordered)}

    # Probe response correctness mapping from script.
    corr_codes = _parse_matlab_assignment_numbers(text, "Probe_Corr")
    inc_codes = _parse_matlab_assignment_numbers(text, "Probe_Incorr")
    acc_map: Dict[str, float] = {}
    for c in corr_codes:
        acc_map[str(int(c))] = 1.0
    for c in inc_codes:
        acc_map[str(int(c))] = 0.0

    literal_list = ",".join([repr(x) for x in ordered])
    out.append(
        Candidate(
            dataset_id="ds005114",
            candidate_id="dpx_probe_onset_trialtype_from_script",
            source="matlab_script_probe_arrays+events_tsv",
            source_files=[str(script)],
            event_filter=f"trial_type in [{literal_list}]",
            load_column="trial_type",
            load_value_map=load_map,
            rt_strategy="next_response_any",
            response_filter="trial_type.str.contains('Response: Probe', case=False, na=False)",
            response_accuracy_map=acc_map,
            task_relevant_value_codes=probe_codes,
            task_relevant_trialtype_regex=r"^Probe_.*_onset$",
            notes="Probe condition derived from STEP1_DPX_PreProc.m trigger arrays.",
        )
    )
    return out


def _discover_candidates(dataset_id: str, dataset_root: Path) -> List[Candidate]:
    if dataset_id == "ds003523":
        return _discover_candidates_for_ds003523(dataset_root)
    if dataset_id == "ds005114":
        return _discover_candidates_for_ds005114(dataset_root)
    return []


def _write_candidate_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        pd.DataFrame(
            columns=[
                "dataset_id",
                "candidate_id",
                "status",
                "reason",
                "event_filter",
                "load_column",
                "n_load_levels",
                "coverage_median",
                "same_levels_rate",
                "freq_p90_l1",
                "gate_all",
                "score",
                "source",
                "source_files",
                "load_value_map",
            ]
        ).to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical_bids_root", type=Path, required=True)
    ap.add_argument("--datasets", type=str, default="ds003523,ds005114")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--out_event_map", type=Path, required=True)
    ap.add_argument("--base_event_map", type=Path, default=Path("configs/lawc_event_map.yaml"))
    ap.add_argument("--sample_subjects", type=int, default=20)
    ap.add_argument("--random_seed", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_event_map.parent.mkdir(parents=True, exist_ok=True)

    datasets = _split_csv(args.datasets)
    rng = random.Random(int(args.random_seed))

    defaults: Dict[str, Any] = {}
    if args.base_event_map.exists():
        try:
            base = yaml.safe_load(args.base_event_map.read_text(encoding="utf-8")) or {}
            if isinstance(base, dict):
                defaults = dict(base.get("defaults") or {})
        except Exception:
            defaults = {}

    event_map_payload: Dict[str, Any] = {"defaults": defaults, "datasets": {}}
    summary_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []

    for ds in datasets:
        ds_root = args.clinical_bids_root / ds
        if not ds_root.exists():
            reason = f"dataset folder missing: {ds_root}"
            stop = args.out_dir / f"STOP_REASON_{ds}.md"
            _write_stop_reason(stop, ds, reason, {"dataset_root": str(ds_root)})
            summary_rows.append(
                {
                    "dataset_id": ds,
                    "status": "SKIP",
                    "reason": reason,
                    "stop_reason": str(stop),
                    "selected_candidate_id": "",
                }
            )
            continue

        events = _event_files(ds_root)
        if not events:
            reason = "no events.tsv files found"
            stop = args.out_dir / f"STOP_REASON_{ds}.md"
            _write_stop_reason(stop, ds, reason, {"dataset_root": str(ds_root)})
            summary_rows.append(
                {"dataset_id": ds, "status": "SKIP", "reason": reason, "stop_reason": str(stop), "selected_candidate_id": ""}
            )
            continue

        subject_to_files: Dict[str, List[Path]] = {}
        for fp in events:
            sid = _subject_from_path(fp)
            if not sid:
                continue
            subject_to_files.setdefault(sid, []).append(fp)
        subjects = sorted(subject_to_files.keys())
        if not subjects:
            reason = "could not derive subject IDs from event paths"
            stop = args.out_dir / f"STOP_REASON_{ds}.md"
            _write_stop_reason(stop, ds, reason, {"dataset_root": str(ds_root)})
            summary_rows.append(
                {"dataset_id": ds, "status": "SKIP", "reason": reason, "stop_reason": str(stop), "selected_candidate_id": ""}
            )
            continue

        sample_n = min(len(subjects), max(20, int(args.sample_subjects)))
        sampled_subjects = rng.sample(subjects, sample_n) if len(subjects) > sample_n else subjects

        # Source diagnostics: sidecars and codebooks
        events_json_count = int(len(list(ds_root.rglob("*_events.json"))))
        xls_files = sorted(list((ds_root / "code").rglob("*.xls")) + list((ds_root / "code").rglob("*.xlsx"))) if (ds_root / "code").exists() else []

        cands = _discover_candidates(ds, ds_root)
        if not cands:
            reason = "no auditable candidate mapping could be derived from events/code sources"
            stop = args.out_dir / f"STOP_REASON_{ds}.md"
            _write_stop_reason(
                stop,
                ds,
                reason,
                {
                    "dataset_root": str(ds_root),
                    "events_json_count": events_json_count,
                    "xls_files": [str(x) for x in xls_files[:20]],
                    "code_dir_exists": bool((ds_root / "code").exists()),
                },
            )
            summary_rows.append(
                {
                    "dataset_id": ds,
                    "status": "SKIP",
                    "reason": reason,
                    "stop_reason": str(stop),
                    "selected_candidate_id": "",
                }
            )
            continue

        evals: List[CandidateEval] = []
        for cand in cands:
            ev = _evaluate_candidate(
                dataset_root=ds_root,
                cand=cand,
                sampled_subjects=sampled_subjects,
                subject_to_files=subject_to_files,
            )
            evals.append(ev)
            candidate_rows.append(
                {
                    "dataset_id": ds,
                    "candidate_id": cand.candidate_id,
                    "status": ev.status,
                    "reason": ev.reason,
                    "event_filter": cand.event_filter,
                    "load_column": cand.load_column,
                    "n_load_levels": ev.metrics.get("n_load_levels", np.nan),
                    "coverage_median": ev.metrics.get("coverage_median", np.nan),
                    "same_levels_rate": ev.metrics.get("same_levels_rate", np.nan),
                    "freq_p90_l1": ev.metrics.get("freq_p90_l1", np.nan),
                    "gate_all": ev.metrics.get("gate_all", False),
                    "score": ev.metrics.get("score", np.nan),
                    "source": cand.source,
                    "source_files": json.dumps(cand.source_files),
                    "load_value_map": json.dumps(cand.load_value_map, sort_keys=True),
                }
            )

        passing = [e for e in evals if e.status == "PASS" and bool(e.metrics.get("gate_all", False))]
        passing = sorted(passing, key=lambda x: float(x.metrics.get("score", -1e9)), reverse=True)

        if not passing:
            reason = "no mapping candidate passed strict gates"
            stop = args.out_dir / f"STOP_REASON_{ds}.md"
            _write_stop_reason(
                stop,
                ds,
                reason,
                {
                    "sampled_subjects": sampled_subjects,
                    "events_json_count": events_json_count,
                    "xls_files": [str(x) for x in xls_files[:20]],
                    "candidate_metrics": [dict(candidate_id=e.candidate.candidate_id, status=e.status, reason=e.reason, metrics=e.metrics) for e in evals],
                },
            )
            summary_rows.append(
                {
                    "dataset_id": ds,
                    "status": "SKIP",
                    "reason": reason,
                    "stop_reason": str(stop),
                    "selected_candidate_id": "",
                }
            )
            continue

        if len(passing) > 1:
            top = float(passing[0].metrics.get("score", np.nan))
            second = float(passing[1].metrics.get("score", np.nan))
            if np.isfinite(top) and np.isfinite(second) and abs(top - second) <= 0.02:
                reason = "ambiguous: multiple candidates passed with similar scores"
                stop = args.out_dir / f"STOP_REASON_{ds}.md"
                _write_stop_reason(
                    stop,
                    ds,
                    reason,
                    {
                        "top_candidates": [
                            {"candidate_id": p.candidate.candidate_id, "score": p.metrics.get("score"), "metrics": p.metrics}
                            for p in passing[:3]
                        ]
                    },
                )
                summary_rows.append(
                    {
                        "dataset_id": ds,
                        "status": "SKIP",
                        "reason": reason,
                        "stop_reason": str(stop),
                        "selected_candidate_id": "",
                    }
                )
                continue

        chosen = passing[0]
        c = chosen.candidate
        mapping = {
            "event_filter": c.event_filter,
            "load_column": c.load_column,
            "load_value_map": c.load_value_map,
            "rt_strategy": c.rt_strategy,
            "response_filter": c.response_filter,
            "response_accuracy_map": c.response_accuracy_map,
        }
        event_map_payload["datasets"][ds] = mapping
        summary_rows.append(
            {
                "dataset_id": ds,
                "status": "PASS",
                "reason": "",
                "selected_candidate_id": c.candidate_id,
                "selected_source": c.source,
                "mapping": mapping,
                "metrics": chosen.metrics,
                "sampled_subjects": sampled_subjects,
                "events_json_count": events_json_count,
                "xls_files": [str(x) for x in xls_files[:20]],
            }
        )

    candidate_csv = args.out_dir / "CANDIDATE_TABLE.csv"
    _write_candidate_table(candidate_csv, candidate_rows)

    args.out_event_map.write_text(yaml.safe_dump(event_map_payload, sort_keys=True), encoding="utf-8")

    overall_status = "PASS" if any(r.get("status") == "PASS" for r in summary_rows) else "SKIP"
    summary = {
        "status": overall_status,
        "clinical_bids_root": str(args.clinical_bids_root),
        "datasets_requested": datasets,
        "out_event_map": str(args.out_event_map),
        "candidate_table_csv": str(candidate_csv),
        "rows": summary_rows,
    }
    summary_json = args.out_dir / "mapping_decode_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Human-readable markdown
    md = args.out_dir / "mapping_decode_summary.md"
    lines = [
        "# Clinical Mapping Decode Summary",
        "",
        f"- Status: `{overall_status}`",
        f"- Event map: `{args.out_event_map}`",
        f"- Candidate table: `{candidate_csv}`",
        "",
        "| Dataset | Status | Candidate | Reason |",
        "|---|---|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r.get('dataset_id')} | {r.get('status')} | {r.get('selected_candidate_id', '')} | {r.get('reason', '')} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"STATUS={overall_status}")
    print(f"EVENT_MAP={args.out_event_map}")
    print(f"SUMMARY_JSON={summary_json}")
    print(f"CANDIDATE_TABLE={candidate_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

