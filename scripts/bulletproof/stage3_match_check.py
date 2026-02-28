#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common import (
    REQUIRED_CONFIRMATORY_DATASETS,
    canonicalize_metric_key,
    ensure_out_tree,
    ensure_stage_status,
    is_confirmatory_metric_key,
    parse_dataset_hashes_payload,
    read_json,
    stop_reason,
    write_text,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path, required=True)
    return ap.parse_args()


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _dataset_from_key(key: str) -> str:
    for part in str(key).split("."):
        if part.startswith("ds") and len(part) == 8 and part[2:].isdigit():
            return part
    return ""


def _is_int_key(k: str) -> bool:
    return k.endswith(".n") or k.endswith(".n_perm_done") or k.endswith(".n_boot_done") or k.endswith(".n_subjects")


def _is_rho_key(k: str) -> bool:
    kk = k.lower()
    return kk.endswith(".rho") or "rho" in kk


def _is_auc_key(k: str) -> bool:
    return "auc" in k.lower()


def _is_pq_key(k: str) -> bool:
    kk = k.lower()
    return kk.endswith(".p") or kk.endswith(".q")


def _load_expected(audit: Path) -> Tuple[Dict[str, float], Dict[str, str]]:
    em = audit / "expected_confirmatory_metrics.json"
    eh = audit / "expected_dataset_hashes.json"
    if not em.exists() or not eh.exists():
        return {}, {}

    metrics_payload = read_json(em)
    raw = metrics_payload.get("metrics", {}) if isinstance(metrics_payload.get("metrics"), dict) else {}
    metrics: Dict[str, float] = {}
    for k, v in raw.items():
        ck = canonicalize_metric_key(str(k))
        if not ck or not is_confirmatory_metric_key(ck):
            continue
        fv = _to_float(v)
        if math.isfinite(fv):
            metrics[ck] = fv

    commit_payload = read_json(eh)
    commit_map_raw = commit_payload.get("datasets", {}) if isinstance(commit_payload.get("datasets"), dict) else {}
    commits = {str(k): str(v) for k, v in commit_map_raw.items() if str(k) in REQUIRED_CONFIRMATORY_DATASETS and str(v).strip()}
    return metrics, commits


def _load_observed(out_root: Path) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    paths = ensure_out_tree(out_root)
    repro = paths["REPRO_FROM_SCRATCH"]

    metrics: Dict[str, float] = {}
    per_dataset: Dict[str, Dict[str, Any]] = {}
    file_diffs: List[Dict[str, Any]] = []

    for ds in REQUIRED_CONFIRMATORY_DATASETS:
        ds_dir = repro / ds
        done = ds_dir / "DONE.json"
        res = ds_dir / "results.json"
        mcsv = ds_dir / "metrics.csv"
        if not done.exists() or not res.exists() or not mcsv.exists():
            file_diffs.append(
                {
                    "type": "file_missing",
                    "dataset_id": ds,
                    "missing": [
                        str(p)
                        for p in [done, res, mcsv]
                        if not p.exists()
                    ],
                }
            )
            continue

        try:
            payload = read_json(res)
        except Exception:
            file_diffs.append({"type": "file_missing", "dataset_id": ds, "missing": [str(res)], "reason": "json_parse_failed"})
            continue

        per_dataset[ds] = payload
        m = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
        for k, v in m.items():
            ck = canonicalize_metric_key(str(k))
            if not ck or not is_confirmatory_metric_key(ck):
                continue
            fv = _to_float(v)
            if math.isfinite(fv):
                metrics[ck] = fv

        # expose job-level counts as fallback metric keys
        n_subjects = _to_float(payload.get("n_subjects"))
        n_trials = _to_float(payload.get("n_trials"))
        n_perm_done = _to_float(payload.get("n_perm_done"))
        n_boot_done = _to_float(payload.get("n_boot_done"))

        if math.isfinite(n_subjects):
            metrics[f"{ds}.n_subjects"] = n_subjects
        if math.isfinite(n_trials):
            metrics[f"{ds}.n_trials"] = n_trials
        if math.isfinite(n_perm_done):
            metrics[f"{ds}.n_perm_done"] = n_perm_done
        if math.isfinite(n_boot_done):
            metrics[f"{ds}.n_boot_done"] = n_boot_done

    return metrics, per_dataset, file_diffs


def _load_observed_commits(audit: Path) -> Dict[str, str]:
    p = audit / "dataset_hashes.json"
    if not p.exists():
        return {}
    payload = read_json(p)
    return parse_dataset_hashes_payload(payload)


def _pq_tolerance(key: str, per_dataset: Dict[str, Dict[str, Any]]) -> float:
    ds = _dataset_from_key(key)
    if not ds:
        return 1e-4
    row = per_dataset.get(ds, {})
    deterministic = bool(row.get("deterministic_seed", False))
    n_perm_done = int(max(1, round(_to_float(row.get("n_perm_done", 0)))))
    if deterministic:
        return 0.0
    return float(2.0 / float(n_perm_done))


def _render_diff_report(out_root: Path, expected_source: str, diffs: List[Dict[str, Any]]) -> None:
    top = diffs[:50]
    lines = [
        "# DIFF_REPORT",
        "",
        f"Expected source: `{expected_source}`",
        f"Total diffs: `{len(diffs)}`",
        f"Top shown: `{len(top)}`",
        "",
        "## Summary",
        f"- missing_metric: `{sum(1 for d in diffs if d.get('type') == 'missing_metric')}`",
        f"- numeric_drift: `{sum(1 for d in diffs if d.get('type') == 'numeric_drift')}`",
        f"- file_missing: `{sum(1 for d in diffs if d.get('type') == 'file_missing')}`",
        f"- dataset_commit_mismatch: `{sum(1 for d in diffs if d.get('type') == 'dataset_commit_mismatch')}`",
        "",
        "## Top 50 diffs",
        "",
        "| type | metric_or_dataset | expected | observed | abs_diff | tolerance | details |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    for d in top:
        metric_or_dataset = d.get("metric", d.get("dataset_id", ""))
        details = d.get("details", "")
        lines.append(
            "| {t} | {m} | {e} | {o} | {ad} | {tol} | {det} |".format(
                t=d.get("type", ""),
                m=metric_or_dataset,
                e=d.get("expected", ""),
                o=d.get("observed", ""),
                ad=d.get("abs_diff", ""),
                tol=d.get("tolerance", ""),
                det=str(details).replace("|", "/"),
            )
        )

    write_text(out_root / "DIFF_REPORT.md", "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    out_root = args.out_root
    paths = ensure_out_tree(out_root)
    audit = paths["AUDIT"]

    expected_metrics, expected_commits = _load_expected(audit)
    expected_source = str(audit / "expected_kit_ref")

    if not expected_metrics or not expected_commits:
        stop_reason(
            audit / "STOP_REASON_stage3_match_check.md",
            "stage3_match_check",
            "Expected kit parse outputs are missing. Stage3 cannot run without Stage0 expected metrics + dataset commit map.",
            diagnostics={
                "expected_metrics_file": str(audit / "expected_confirmatory_metrics.json"),
                "expected_commits_file": str(audit / "expected_dataset_hashes.json"),
            },
        )
        ensure_stage_status(audit, "stage3_match_check", "FAIL", {"reason": "missing_expected_payload"})
        return 1

    observed_metrics, per_dataset, file_diffs = _load_observed(out_root)
    observed_commits = _load_observed_commits(audit)

    diffs: List[Dict[str, Any]] = []
    diffs.extend(file_diffs)

    # Exact commit match required for confirmatory datasets.
    for ds in REQUIRED_CONFIRMATORY_DATASETS:
        exp = str(expected_commits.get(ds, "")).strip()
        obs = str(observed_commits.get(ds, "")).strip()
        if exp != obs:
            diffs.append(
                {
                    "type": "dataset_commit_mismatch",
                    "dataset_id": ds,
                    "expected": exp,
                    "observed": obs,
                    "details": "Expected and observed git commit differ.",
                }
            )

    # Compare only confirmatory metrics listed in expected kit.
    for key in sorted(expected_metrics.keys()):
        if not is_confirmatory_metric_key(key):
            continue

        exp = float(expected_metrics[key])
        obs = observed_metrics.get(key)
        if obs is None or not math.isfinite(float(obs)):
            diffs.append(
                {
                    "type": "missing_metric",
                    "metric": key,
                    "expected": exp,
                    "observed": "",
                    "details": "Expected confirmatory metric missing in Stage2 outputs.",
                }
            )
            continue

        obs_f = float(obs)
        if _is_int_key(key):
            if int(round(obs_f)) != int(round(exp)):
                diffs.append(
                    {
                        "type": "numeric_drift",
                        "metric": key,
                        "expected": int(round(exp)),
                        "observed": int(round(obs_f)),
                        "abs_diff": abs(int(round(obs_f)) - int(round(exp))),
                        "tolerance": 0,
                        "details": "Integer metric must match exactly.",
                    }
                )
            continue

        if _is_rho_key(key):
            tol = 1e-6
        elif _is_auc_key(key):
            tol = 1e-6
        elif _is_pq_key(key):
            tol = _pq_tolerance(key, per_dataset)
        else:
            tol = 1e-6

        delta = abs(obs_f - exp)
        if delta > tol:
            diffs.append(
                {
                    "type": "numeric_drift",
                    "metric": key,
                    "expected": exp,
                    "observed": obs_f,
                    "abs_diff": delta,
                    "tolerance": tol,
                    "details": "Float metric drift beyond tolerance.",
                }
            )

    report_lines = [
        "# CONFIRMATORY_MATCH_REPORT",
        "",
        f"- expected_source: `{expected_source}`",
        f"- expected_metrics: `{len(expected_metrics)}`",
        f"- observed_metrics: `{len(observed_metrics)}`",
        f"- commit_rows_expected: `{len(expected_commits)}`",
        f"- commit_rows_observed: `{len(observed_commits)}`",
        "",
    ]

    if diffs:
        _render_diff_report(out_root, expected_source, diffs)
        report_lines.extend(
            [
                "## Status",
                "FAIL",
                "",
                f"- n_diffs: `{len(diffs)}`",
                f"- diff_report: `{out_root / 'DIFF_REPORT.md'}`",
            ]
        )
        write_text(audit / "CONFIRMATORY_MATCH_REPORT.md", "\n".join(report_lines) + "\n")
        stop_reason(
            audit / "STOP_REASON_stage3_match_check.md",
            "stage3_match_check",
            "Confirmatory strict match failed.",
            diagnostics={"n_diffs": len(diffs), "diff_report": str(out_root / "DIFF_REPORT.md")},
        )
        ensure_stage_status(
            audit,
            "stage3_match_check",
            "FAIL",
            {"n_diffs": len(diffs), "diff_report": str(out_root / "DIFF_REPORT.md")},
        )
        return 1

    report_lines.extend(["## Status", "PASS"])
    write_text(audit / "CONFIRMATORY_MATCH_REPORT.md", "\n".join(report_lines) + "\n")
    ensure_stage_status(
        audit,
        "stage3_match_check",
        "PASS",
        {"matched_metrics": len(expected_metrics), "expected_source": expected_source},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
