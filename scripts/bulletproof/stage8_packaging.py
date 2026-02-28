#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from common import copytree_files, ensure_out_tree, ensure_stage_status, parse_csv_rows, sha256_file, stop_reason, write_text, zip_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path, required=True)
    return ap.parse_args()


def _collect_stage_summaries(audit: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for p in sorted(audit.glob("stage*_summary.json")):
        try:
            out[p.stem.replace("_summary", "")] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return out


def _build_final_results_table(repro: Path, dst_csv: Path) -> int:
    rows: List[Dict[str, str]] = []

    lawc = repro / "PACK_CORE_LAWC" / "lawc_ultradeep" / "lawc_audit" / "locked_test_results.csv"
    for r in parse_csv_rows(lawc):
        ds = str(r.get("dataset_id", "")).strip()
        if not ds:
            continue
        rows.append({"metric": f"lawc.{ds}.observed_median", "value": str(r.get("observed_median", r.get("median_subject_rho", "")))})
        rows.append({"metric": f"lawc.{ds}.p", "value": str(r.get("p_value", r.get("perm_p", "")))})
        rows.append({"metric": f"lawc.{ds}.q", "value": str(r.get("q_value", r.get("perm_q", "")))})
        rows.append({"metric": f"lawc.{ds}.n", "value": str(r.get("n_subjects_used", r.get("n", "")))})
        rows.append({"metric": f"lawc.{ds}.perms", "value": str(r.get("n_perm", ""))})

    clin = repro / "PACK_CLINICAL" / "clinical_endpoints_all.csv"
    for r in parse_csv_rows(clin):
        ds = str(r.get("dataset_id", "")).strip()
        ep = str(r.get("endpoint", "")).strip()
        if not ds or not ep:
            continue
        rows.append({"metric": f"clinical.{ds}.{ep}.estimate", "value": str(r.get("estimate", ""))})
        rows.append({"metric": f"clinical.{ds}.{ep}.perm_p", "value": str(r.get("perm_p", ""))})
        rows.append({"metric": f"clinical.{ds}.{ep}.perm_q", "value": str(r.get("perm_q", r.get("perm_q_global", "")))})
        rows.append({"metric": f"clinical.{ds}.{ep}.n", "value": str(r.get("n", ""))})

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with dst_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> int:
    args = parse_args()
    out_root = args.out_root
    paths = ensure_out_tree(out_root)
    audit = paths["AUDIT"]

    summaries = _collect_stage_summaries(audit)

    mk = paths["MANUSCRIPT_KIT_UPDATED"]
    ov = paths["OVERLEAF_UPDATED"]
    pr = paths["PRISM_ARTIST_PACK"]
    outzip = paths["OUTZIP"]

    repro = paths["REPRO_FROM_SCRATCH"]
    if not repro.exists():
        stop_reason(audit / "STOP_REASON_stage8_packaging.md", "stage8_packaging", "repro outputs missing")
        ensure_stage_status(audit, "stage8_packaging", "FAIL", {"reason": "repro_missing"})
        return 1

    # Build updated ManuscriptKit structure.
    n_metrics = _build_final_results_table(repro, mk / "FINAL_RESULTS_TABLE.csv")
    copytree_files(repro / "PACK_CORE_LAWC", mk / "PACK_CORE_LAWC")
    copytree_files(repro / "PACK_MECHANISM", mk / "PACK_MECHANISM")
    copytree_files(repro / "PACK_CLINICAL", mk / "PACK_CLINICAL")
    copytree_files(paths["CLINICAL_STABILITY"], mk / "CLINICAL_STABILITY")
    copytree_files(paths["RELIABILITY"], mk / "RELIABILITY")
    (mk / "PROVENANCE").mkdir(parents=True, exist_ok=True)
    for p in [audit / "dataset_hashes.json", audit / "preflight_env.json", audit / "CONFIRMATORY_MATCH_REPORT.md"]:
        if p.exists():
            (mk / "PROVENANCE" / p.name).write_text(p.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")

    # Overleaf package (lightweight source files + figures).
    tables_dir = ov / "tables"
    figs_dir = ov / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    src_table = mk / "FINAL_RESULTS_TABLE.csv"
    if src_table.exists():
        (tables_dir / "FINAL_RESULTS_TABLE.csv").write_text(src_table.read_text(encoding="utf-8"), encoding="utf-8")
    for p in repro.rglob("*.png"):
        rel = p.name
        (figs_dir / rel).write_bytes(p.read_bytes())
    write_text(
        ov / "README_OVERLEAF.md",
        "# Overleaf Updated\n\nIncludes refreshed tables/figures from bulletproof run.\n",
    )

    # Prism artist pack.
    prism_csvs = list(repro.rglob("*.csv")) + list(paths["CLINICAL_STABILITY"].rglob("*.csv")) + list(paths["RELIABILITY"].rglob("*.csv"))
    prism_csvs = sorted({p.resolve() for p in prism_csvs if p.is_file()})
    (pr / "csv").mkdir(parents=True, exist_ok=True)
    map_rows: List[str] = []
    for i, p in enumerate(prism_csvs, start=1):
        dst = pr / "csv" / p.name
        dst.write_bytes(p.read_bytes())
        map_rows.append(f"| PANEL_{i} | {dst.name} | infer from header |")
    write_text(
        pr / "FIGURE_SPECS.md",
        "# Figure Specs\n\nUse CSV headers as Prism column names. Keep axis scales identical to manuscript defaults.\n",
    )
    write_text(
        pr / "PANEL_MAP.md",
        "# Panel Map\n\n| panel | csv | columns |\n|---|---|---|\n" + "\n".join(map_rows) + "\n",
    )

    # Zip deliverables.
    mk_zip = outzip / "MANUSCRIPT_KIT_UPDATED.zip"
    ov_zip = outzip / "OVERLEAF_UPDATED.zip"
    pr_zip = outzip / "PRISM_ARTIST_PACK.zip"
    zip_dir(mk, mk_zip)
    zip_dir(ov, ov_zip)
    zip_dir(pr, pr_zip)

    # Audit report.
    stop_files = sorted([p for p in out_root.rglob("STOP_REASON*.md") if p.is_file()])
    status_lines = [f"- {k}: `{v.get('status')}`" for k, v in sorted(summaries.items())]
    report_lines = [
        "# BULLETPROOF AUDIT REPORT",
        "",
        "## Confirmatory Match",
        f"- status: `{summaries.get('stage3_match_check', {}).get('status', '<missing>')}`",
        f"- report: `{audit / 'CONFIRMATORY_MATCH_REPORT.md'}`",
        "",
        "## Robustness Grid",
        f"- status: `{summaries.get('stage4_robustness_grid', {}).get('status', '<missing>')}`",
        f"- table: `{paths['ROBUSTNESS_GRID'] / 'lawc_robustness.csv'}`",
        "",
        "## Reliability",
        f"- status: `{summaries.get('stage5_reliability', {}).get('status', '<missing>')}`",
        "",
        "## Clinical Calibration",
        f"- status: `{summaries.get('stage6_clinical_stability', {}).get('status', '<missing>')}`",
        f"- report: `{paths['CLINICAL_STABILITY'] / 'clinical_stability_report.md'}`",
        "",
        "## Endpoint Hierarchy / Multiplicity",
        "- Law-C: within-subject Spearman + permutation + BH-FDR.",
        "- Clinical families: BH-FDR within ds004504, ds004584, ds007020; global q included in combined table.",
        "",
        "## Stage Status",
        *status_lines,
        "",
        "## STOP Reasons",
    ]
    if stop_files:
        report_lines.extend([f"- `{p}`" for p in stop_files])
    else:
        report_lines.append("- none")

    report_lines.extend(
        [
            "",
            "## Packaging",
            f"- metrics_written: `{n_metrics}`",
            f"- MANUSCRIPT_KIT_UPDATED.zip sha256: `{sha256_file(mk_zip)}`",
            f"- OVERLEAF_UPDATED.zip sha256: `{sha256_file(ov_zip)}`",
            f"- PRISM_ARTIST_PACK.zip sha256: `{sha256_file(pr_zip)}`",
        ]
    )
    write_text(audit / "BULLETPROOF_AUDIT_REPORT.md", "\n".join(report_lines) + "\n")

    ensure_stage_status(
        audit,
        "stage8_packaging",
        "PASS",
        {
            "manuscript_zip": str(mk_zip),
            "overleaf_zip": str(ov_zip),
            "prism_zip": str(pr_zip),
            "audit_report": str(audit / "BULLETPROOF_AUDIT_REPORT.md"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
