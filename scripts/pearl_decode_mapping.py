#!/usr/bin/env python3
"""Decode fail-closed Sternberg mapping for ds004796 (PEARL-Neuro).

Auditable sources only:
- BIDS events.tsv (+ sidecars if present)
- sourcedata Sternberg logs
- optional code/xls if present (recorded in diagnostics)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


def _event_files_sternberg(dataset_root: Path) -> List[Path]:
    pats = [
        "*task-sternberg*_events.tsv",
        "*task-sternberg*_events.tsv.gz",
    ]
    out: List[Path] = []
    for pat in pats:
        out.extend(dataset_root.rglob(pat))
    return sorted(set(out))


def _subject_from_path(path: Path) -> str:
    m = re.search(r"sub-([A-Za-z0-9]+)", str(path))
    return m.group(1) if m else ""


def _safe_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    return df.query(query, engine="python")


def _normalize_event_code(x: Any) -> str:
    return re.sub(r"\s+", "", str(x).strip().upper())


def _locate_log(dataset_root: Path, subject: str) -> Optional[Path]:
    sid = str(subject).strip()
    if not sid:
        return None
    tokens = [sid]
    if sid.isdigit():
        tokens.append(sid.zfill(2))
        tokens.append(sid.zfill(3))
    tokens = list(dict.fromkeys(tokens))
    cands: List[Path] = []
    for tok in tokens:
        d = dataset_root / "sourcedata" / f"sub-{tok}" / "logfiles"
        if not d.exists():
            continue
        cands.extend(sorted(d.glob(f"sub-{tok}_task-sternberg_events.txt")))
        cands.extend(sorted(d.glob(f"sub-{tok}_task-sternberg*events*.txt")))
    if not cands:
        return None
    for p in cands:
        if p.name.endswith("_task-sternberg_events.txt"):
            return p
    return cands[0]


def _parse_sternberg_log(log_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(log_path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = df.dropna(how="all").reset_index(drop=True)
    if "Event Type" not in df.columns or "Code" not in df.columns:
        raise RuntimeError(f"log missing Event Type/Code columns: {log_path}")

    if "TTime" in df.columns:
        df["TTime"] = pd.to_numeric(df["TTime"], errors="coerce")

    recs: List[Dict[str, Any]] = []
    current_mem: Optional[List[str]] = None

    for _, row in df.iterrows():
        et = str(row.get("Event Type", "")).strip().lower()
        code = str(row.get("Code", "")).strip()

        if et == "picture":
            letters = [x.upper() for x in re.findall(r"[A-Za-z]", code)]
            if len(letters) >= 2 and ("+" in code or " " in code):
                current_mem = letters
                continue
            if re.fullmatch(r"[A-Za-z]", code):
                if not current_mem:
                    continue
                probe = code.upper()
                recs.append(
                    {
                        "load": float(len(current_mem)),
                        "probe_letter": probe,
                        "is_target": float(probe in current_mem),
                        "response_code": np.nan,
                        "rt": np.nan,
                    }
                )
                continue

        if et == "response":
            if not recs:
                continue
            idx = len(recs) - 1
            while idx >= 0 and np.isfinite(float(recs[idx].get("response_code", np.nan))):
                idx -= 1
            if idx < 0:
                continue
            rv = pd.to_numeric(pd.Series([row.get("Code")]), errors="coerce").iloc[0]
            if not np.isfinite(rv):
                m = re.search(r"(\d+)", str(row.get("Code", "")))
                if m:
                    rv = float(m.group(1))
            if np.isfinite(rv):
                recs[idx]["response_code"] = float(rv)
            tv = pd.to_numeric(pd.Series([row.get("TTime")]), errors="coerce").iloc[0]
            if np.isfinite(tv):
                recs[idx]["rt"] = float(tv)

    if not recs:
        raise RuntimeError(f"no probe records parsed from {log_path}")
    out = pd.DataFrame(recs)
    out = out[np.isfinite(pd.to_numeric(out["load"], errors="coerce"))].reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"parsed zero finite loads from {log_path}")

    # Infer yes/no response coding if possible.
    yes_code = None
    yes_score = float("nan")
    fit = out.dropna(subset=["response_code", "is_target"]).copy()
    if not fit.empty:
        fit["response_code"] = pd.to_numeric(fit["response_code"], errors="coerce")
        fit = fit[np.isfinite(fit["response_code"])].copy()
        codes = sorted(set(fit["response_code"].astype(int).tolist()))
        if len(codes) >= 2 and fit["is_target"].nunique() >= 2:
            scored: List[Tuple[int, float]] = []
            tgt = fit["is_target"].to_numpy(dtype=bool)
            for c in codes:
                pred_yes = fit["response_code"].astype(int).to_numpy() == int(c)
                acc = float((pred_yes == tgt).mean())
                scored.append((int(c), acc))
            scored.sort(key=lambda x: x[1], reverse=True)
            if len(scored) >= 2 and scored[0][1] >= 0.55 and (scored[0][1] - scored[1][1]) >= 0.05:
                yes_code = int(scored[0][0])
                yes_score = float(scored[0][1])

    acc = np.full(len(out), np.nan, dtype=float)
    if yes_code is not None:
        r = pd.to_numeric(out["response_code"], errors="coerce").to_numpy(dtype=float)
        t = pd.to_numeric(out["is_target"], errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(r) & np.isfinite(t)
        pred_yes = r == float(yes_code)
        acc[good] = (pred_yes[good] == (t[good] > 0.5)).astype(float)
    out["accuracy"] = acc

    rt = pd.to_numeric(out["rt"], errors="coerce").to_numpy(dtype=float)
    finite = rt[np.isfinite(rt)]
    rt_units = "no-finite-rt"
    if finite.size:
        med = float(np.median(finite))
        if med > 50.0:
            rt = rt / 1000.0
            rt_units = "ms->s"
        else:
            rt_units = "s"
    out["rt"] = rt

    diag = {
        "log_path": str(log_path),
        "n_probe_records": int(len(out)),
        "load_levels": sorted(set(pd.to_numeric(out["load"], errors="coerce").dropna().astype(int).tolist())),
        "inferred_yes_code": yes_code,
        "inferred_yes_code_accuracy": yes_score,
        "rt_units": rt_units,
        "rt_nonmissing_rate": float(np.isfinite(pd.to_numeric(out["rt"], errors="coerce")).mean()),
        "accuracy_nonmissing_rate": float(np.isfinite(pd.to_numeric(out["accuracy"], errors="coerce")).mean()),
    }
    return out, diag


def _distribution_l1(counts: Sequence[float], ref: Sequence[float]) -> float:
    a = np.asarray(counts, dtype=float)
    b = np.asarray(ref, dtype=float)
    if a.sum() <= 0 or b.sum() <= 0:
        return float("inf")
    a = a / a.sum()
    b = b / b.sum()
    return float(np.abs(a - b).sum())


def _write_stop_reason(path: Path, reason: str, diag: Dict[str, Any]) -> None:
    lines = [
        "# STOP_REASON",
        "",
        "## Why mapping was skipped",
        reason,
        "",
        "## Diagnostics",
        "```json",
        json.dumps(diag, indent=2),
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--event_map_out", type=Path, default=Path("configs/pearl_event_map.yaml"))
    ap.add_argument("--dataset_id", type=str, default="ds004796")
    ap.add_argument("--sample_subjects", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_csv = out_dir / "CANDIDATE_TABLE.csv"
    summary_json = out_dir / "mapping_decode_summary.json"
    stop_md = out_dir.parent / "STOP_REASON.md"

    dataset_root = args.dataset_root
    dataset_id = str(args.dataset_id)
    if not dataset_root.exists():
        payload = {"status": "SKIP", "reason": f"dataset_root missing: {dataset_root}"}
        summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_stop_reason(stop_md, payload["reason"], payload)
        print(json.dumps(payload, indent=2))
        return

    event_files = _event_files_sternberg(dataset_root)
    if not event_files:
        payload = {"status": "SKIP", "reason": "no sternberg *_events.tsv files found"}
        summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_stop_reason(stop_md, payload["reason"], payload)
        print(json.dumps(payload, indent=2))
        return

    subject_files: Dict[str, List[Path]] = {}
    for fp in event_files:
        sid = _subject_from_path(fp)
        if not sid:
            continue
        subject_files.setdefault(sid, []).append(fp)

    subjects = sorted(subject_files.keys())
    if len(subjects) == 0:
        payload = {"status": "SKIP", "reason": "no subject IDs parsed from sternberg event files"}
        summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_stop_reason(stop_md, payload["reason"], payload)
        print(json.dumps(payload, indent=2))
        return

    rng = random.Random(int(args.seed))
    sample_n = min(len(subjects), max(int(args.sample_subjects), 1))
    sampled_subjects = sorted(rng.sample(subjects, sample_n))

    # Candidate anchored to S12 probe marker (dataset descriptor Table 5).
    event_filter = "event_type.str.contains('S\\\\s*12', case=False, na=False, regex=True)"
    response_filter = "trial_type.str.contains('response', case=False, na=False)"

    rows: List[Dict[str, Any]] = []
    valid_subject_rows: List[Dict[str, Any]] = []
    logs_seen: List[str] = []

    for sid in sampled_subjects:
        files = sorted(subject_files.get(sid, []))
        if not files:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "subject_id": sid,
                    "status": "FAIL",
                    "reason": "no sternberg events.tsv files",
                }
            )
            continue

        # ds004796 has one sternberg EEG file per subject; fail closed if ambiguous.
        if len(files) != 1:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "subject_id": sid,
                    "status": "FAIL",
                    "reason": f"ambiguous sternberg events files (n={len(files)})",
                    "events_files": "|".join(str(x) for x in files),
                }
            )
            continue

        fp = files[0]
        try:
            df = pd.read_csv(fp, sep="\t")
        except Exception as exc:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "subject_id": sid,
                    "status": "FAIL",
                    "reason": f"events read failure: {exc}",
                    "events_file": str(fp),
                }
            )
            continue

        cols_lc = {str(c).lower(): c for c in df.columns}
        if "event_type" not in cols_lc:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "subject_id": sid,
                    "status": "FAIL",
                    "reason": "event_type column missing",
                    "events_file": str(fp),
                }
            )
            continue

        et_col = cols_lc["event_type"]
        s12_mask = df[et_col].astype(str).str.contains(r"\bS\s*12\b", case=False, regex=True, na=False)
        n_s12_ref = int(s12_mask.sum())
        if n_s12_ref <= 0:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "subject_id": sid,
                    "status": "FAIL",
                    "reason": "no S12 markers in event_type",
                    "events_file": str(fp),
                }
            )
            continue

        qdf = df.rename(columns={et_col: "event_type"})
        try:
            selected = _safe_query(qdf, event_filter)
        except Exception as exc:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "subject_id": sid,
                    "status": "FAIL",
                    "reason": f"event_filter query failed: {exc}",
                    "events_file": str(fp),
                }
            )
            continue

        n_selected = int(len(selected))
        coverage = float(n_selected / max(1, n_s12_ref))

        log_path = _locate_log(dataset_root, sid)
        if log_path is None:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "subject_id": sid,
                    "status": "FAIL",
                    "reason": "sourcedata sternberg log missing",
                    "events_file": str(fp),
                }
            )
            continue

        try:
            load_df, load_diag = _parse_sternberg_log(log_path)
        except Exception as exc:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "subject_id": sid,
                    "status": "FAIL",
                    "reason": f"log parse failure: {exc}",
                    "events_file": str(fp),
                    "log_file": str(log_path),
                }
            )
            continue

        logs_seen.append(str(log_path))
        n_load = int(len(load_df))
        load_vals = pd.to_numeric(load_df["load"], errors="coerce").to_numpy(dtype=float)
        load_levels = sorted(set(load_vals[np.isfinite(load_vals)].astype(int).tolist()))
        n_levels = int(len(load_levels))

        if n_selected != n_load:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "subject_id": sid,
                    "status": "FAIL",
                    "reason": f"probe/log count mismatch selected={n_selected} load_records={n_load}",
                    "events_file": str(fp),
                    "log_file": str(log_path),
                    "coverage_vs_s12": coverage,
                    "n_s12_ref": n_s12_ref,
                    "n_selected": n_selected,
                    "n_load_records": n_load,
                }
            )
            continue

        status = "PASS"
        reason = ""
        if coverage < 0.8:
            status = "FAIL"
            reason = "event_filter coverage below 0.8 for S12 probes"
        elif not (2 <= n_levels <= 12):
            status = "FAIL"
            reason = f"derived load levels out of range (n={n_levels})"

        row = {
            "dataset_id": dataset_id,
            "subject_id": sid,
            "status": status,
            "reason": reason,
            "events_file": str(fp),
            "log_file": str(log_path),
            "coverage_vs_s12": coverage,
            "n_s12_ref": n_s12_ref,
            "n_selected": n_selected,
            "n_load_records": n_load,
            "n_load_levels": n_levels,
            "load_levels": "|".join(str(x) for x in load_levels),
            "load_level_counts": json.dumps(
                {str(int(k)): int(v) for k, v in pd.Series(load_vals).dropna().astype(int).value_counts().sort_index().items()}
            ),
            "rt_nonmissing_rate_log": float(np.isfinite(pd.to_numeric(load_df["rt"], errors="coerce")).mean()),
            "acc_nonmissing_rate_log": float(np.isfinite(pd.to_numeric(load_df["accuracy"], errors="coerce")).mean()),
            "inferred_yes_code": load_diag.get("inferred_yes_code"),
        }
        rows.append(row)
        if status == "PASS":
            valid_subject_rows.append(row)

    cand_df = pd.DataFrame(rows)
    cand_df.to_csv(candidate_csv, index=False)

    # Stability gates across sampled subjects:
    pass_rows = [r for r in valid_subject_rows if r.get("status") == "PASS"]
    gate_reason = ""
    stable_pass = True
    if len(pass_rows) < min(20, sample_n):
        stable_pass = False
        gate_reason = (
            f"insufficient subjects passing per-subject gates ({len(pass_rows)}/{min(20, sample_n)} required)"
        )

    if stable_pass:
        level_sets = [tuple(int(x) for x in str(r.get("load_levels", "")).split("|") if str(x).strip()) for r in pass_rows]
        mode_set = max(set(level_sets), key=level_sets.count)
        frac_mode = float(level_sets.count(mode_set) / max(1, len(level_sets)))
        if frac_mode < 0.8:
            stable_pass = False
            gate_reason = f"load level-set inconsistency across subjects (mode_fraction={frac_mode:.3f})"

    if stable_pass:
        freq_vecs: List[np.ndarray] = []
        for r in pass_rows:
            try:
                d = json.loads(str(r.get("load_level_counts", "{}")))
            except Exception:
                continue
            keys = sorted(int(k) for k in d.keys())
            if not keys:
                continue
            vec = np.asarray([float(d.get(str(k), 0.0)) for k in keys], dtype=float)
            freq_vecs.append(vec)
        if not freq_vecs:
            stable_pass = False
            gate_reason = "no load frequency vectors available for stability check"
        else:
            ref = np.median(np.vstack([v / max(v.sum(), 1.0) for v in freq_vecs]), axis=0)
            l1_vals = [float(np.abs((v / max(v.sum(), 1.0)) - ref).sum()) for v in freq_vecs]
            l1_med = float(np.median(l1_vals))
            if l1_med > 0.35:
                stable_pass = False
                gate_reason = f"subject load-frequency instability (median L1={l1_med:.3f} > 0.35)"

    mapping = {
        "event_filter": event_filter,
        "load_column": "event_type",
        "load_source": "sourcedata_sternberg_log",
        "load_sign": 1.0,
        "rt_strategy": "next_response_any",
        "response_filter": response_filter,
        "accuracy_source": "sourcedata_sternberg_log",
    }

    status = "PASS" if stable_pass else "SKIP"
    reason = "" if stable_pass else gate_reason

    if status == "PASS":
        cfg = {
            "defaults": {
                "min_trials_per_subject": 20,
                "min_rt_nonmissing_rate": 0.3,
                "control_p_threshold": 0.2,
                "control_rho_threshold": 0.0,
                "control_n_shuffles": 128,
            },
            "datasets": {dataset_id: mapping},
        }
        args.event_map_out.parent.mkdir(parents=True, exist_ok=True)
        args.event_map_out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    else:
        _write_stop_reason(
            stop_md,
            reason,
            {
                "dataset_id": dataset_id,
                "sample_subjects": sampled_subjects,
                "n_event_files": len(event_files),
                "candidate_table": str(candidate_csv),
            },
        )

    summary = {
        "status": status,
        "reason": reason,
        "dataset_id": dataset_id,
        "dataset_root": str(dataset_root),
        "n_event_files": len(event_files),
        "n_subjects_with_events": len(subjects),
        "sampled_subjects": sampled_subjects,
        "n_subjects_passed_gates": len(pass_rows),
        "event_map_out": str(args.event_map_out),
        "mapping": mapping if status == "PASS" else {},
        "candidate_table": str(candidate_csv),
        "logs_seen_sample": sorted(set(logs_seen))[:20],
        "stop_reason_path": str(stop_md) if stop_md.exists() else "",
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
