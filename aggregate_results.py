#!/usr/bin/env python3
"""Aggregate multi-seed outputs and emit AUDIT_SUMMARY.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--stage_manifest", type=Path, default=None)
    ap.add_argument("--allow_deterministic", action="store_true")
    return ap.parse_args()


def _safe_cmd(cmd: list[str], *, cwd: Optional[Path] = None) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, cwd=str(cwd) if cwd else None)
        return out.decode("utf-8", errors="replace").strip()
    except Exception as exc:
        return f"<unavailable: {exc}>"


def _read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ci95(values: List[float]) -> List[float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]


def _mean(values: List[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _compute_repo_fingerprint(repo_root: Path) -> Dict[str, Any]:
    git_head = _safe_cmd(["git", "rev-parse", "HEAD"], cwd=repo_root)
    git_ok = not git_head.startswith("<unavailable:")

    payload: Dict[str, Any] = {
        "repo_root": str(repo_root),
        "git_head": git_head if git_ok else None,
        "repo_fingerprint_sha256": None,
        "file_count_hashed": 0,
    }

    if git_ok:
        return payload

    patterns = ("*.py", "*.sh", "*.yaml", "*.yml", "*.txt")
    files: List[Path] = []
    for pat in patterns:
        files.extend(repo_root.rglob(pat))

    keep = []
    for p in files:
        if not p.is_file():
            continue
        rel = p.relative_to(repo_root)
        if rel.parts and rel.parts[0] == ".git":
            continue
        if "__pycache__" in rel.parts:
            continue
        keep.append(p)

    keep = sorted(set(keep), key=lambda p: str(p.relative_to(repo_root)))
    h = hashlib.sha256()
    for p in keep:
        rel = str(p.relative_to(repo_root)).encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")

    payload["repo_fingerprint_sha256"] = h.hexdigest()
    payload["file_count_hashed"] = len(keep)
    return payload


def _collect_seed_results(out_root: Path, seeds: List[int]) -> Dict[str, Any]:
    per_seed: List[Dict[str, Any]] = []

    med_vals: List[float] = []
    nll_vals: List[float] = []
    zstd_vals: List[float] = []
    zstab_vals: List[float] = []
    rt_vals_healthy: List[float] = []
    rt_vals_clinical: List[float] = []

    seed_checksums: List[str] = []
    bootstrap_hashes: List[str] = []

    for seed in seeds:
        run_id = f"seed_{seed}"
        seed_root = out_root / f"seed_{seed}"

        med_path = seed_root / "reports" / "mediation" / run_id / "mediation_summary.json"
        norm_path = seed_root / "reports" / "normative" / run_id / "normative_metrics.json"
        rt_path = seed_root / "reports" / "normative" / run_id / "rt_linkage_summary.json"

        med = _read_json(med_path)
        norm = _read_json(norm_path)
        rt = _read_json(rt_path)

        med_mean = _to_float(med.get("effect_mean")) if med else float("nan")
        if np.isfinite(med_mean):
            med_vals.append(med_mean)

        healthy_nll = _to_float((norm.get("healthy") or {}).get("nll")) if norm else float("nan")
        if np.isfinite(healthy_nll):
            nll_vals.append(healthy_nll)

        healthy_zstd = _to_float(((norm.get("healthy") or {}).get("calibration") or {}).get("z_std")) if norm else float("nan")
        if np.isfinite(healthy_zstd):
            zstd_vals.append(healthy_zstd)

        healthy_zstab = _to_float(((norm.get("healthy") or {}).get("z_stability") or {}).get("subject_mean_std")) if norm else float("nan")
        if np.isfinite(healthy_zstab):
            zstab_vals.append(healthy_zstab)

        rt_h = _to_float(((rt.get("linkage") or {}).get("healthy") or {}).get("mean_beta_margin")) if rt else float("nan")
        if np.isfinite(rt_h):
            rt_vals_healthy.append(rt_h)

        rt_c = _to_float(((rt.get("linkage") or {}).get("clinical") or {}).get("mean_beta_margin")) if rt else float("nan")
        if np.isfinite(rt_c):
            rt_vals_clinical.append(rt_c)

        checksum = str((norm or {}).get("seed_effect_checksum", ""))
        if checksum:
            seed_checksums.append(checksum)

        bootstrap_hash = str(((norm or {}).get("bootstrap") or {}).get("subject_bootstrap_indices_hash", ""))
        if bootstrap_hash:
            bootstrap_hashes.append(bootstrap_hash)

        per_seed.append(
            {
                "seed": seed,
                "module03_mediation_summary": med,
                "module04_normative_metrics": norm,
                "module04_rt_linkage": rt,
            }
        )

    unique_checksums = sorted(set(seed_checksums))
    deterministic_fail_flag = len(seed_checksums) > 1 and len(unique_checksums) == 1

    aggregate = {
        "mechanism_mediation_effect": {
            "n_seeds": len(med_vals),
            "mean": _mean(med_vals),
            "ci95": _ci95(med_vals),
            "values": med_vals,
        },
        "normative_metrics": {
            "n_seeds": len(nll_vals),
            "healthy_nll": {"mean": _mean(nll_vals), "ci95": _ci95(nll_vals), "values": nll_vals},
            "healthy_calibration_z_std": {"mean": _mean(zstd_vals), "ci95": _ci95(zstd_vals), "values": zstd_vals},
            "healthy_z_stability_subject_mean_std": {
                "mean": _mean(zstab_vals),
                "ci95": _ci95(zstab_vals),
                "values": zstab_vals,
            },
            "seed_effect_checksums": seed_checksums,
            "bootstrap_indices_hashes": bootstrap_hashes,
            "deterministic_seed_checksum_fail_flag": bool(deterministic_fail_flag),
            "deterministic_seed_checksum_warning": (
                "All seed_effect_checksums are identical across seeds."
                if deterministic_fail_flag
                else ""
            ),
        },
        "rt_linkage_effect": {
            "healthy_mean_beta_margin": {
                "n_seeds": len(rt_vals_healthy),
                "mean": _mean(rt_vals_healthy),
                "ci95": _ci95(rt_vals_healthy),
                "values": rt_vals_healthy,
            },
            "clinical_mean_beta_margin": {
                "n_seeds": len(rt_vals_clinical),
                "mean": _mean(rt_vals_clinical),
                "ci95": _ci95(rt_vals_clinical),
                "values": rt_vals_clinical,
            },
        },
        "per_seed": per_seed,
    }
    return aggregate


def _audit_summary_md(
    *,
    out_root: Path,
    seeds: List[int],
    aggregate: Dict[str, Any],
    stage_manifest: Dict[str, Any] | None,
    repo_fingerprint: Dict[str, Any],
    lawc_audit: Dict[str, Any] | None,
) -> str:
    git_commit = repo_fingerprint.get("git_head") or "<unavailable>"
    pip_freeze = _safe_cmd(["python", "-m", "pip", "freeze"])
    conda_list = _safe_cmd(["conda", "list"])
    cuda_info = _safe_cmd(["nvidia-smi", "-L"])

    dataset_hash_lines = []
    if stage_manifest and isinstance(stage_manifest.get("datasets"), list):
        for ds in stage_manifest["datasets"]:
            dataset_hash_lines.append(
                f"- {ds.get('id')}: pinned={ds.get('pinned_hash')} checked_out={ds.get('checked_out_hash')}"
            )
    if not dataset_hash_lines:
        dataset_hash_lines.append("- <unavailable>")

    seed_table_rows = ["| Seed | Mediation Mean | Healthy NLL | Healthy RT beta | Seed checksum |", "|---|---:|---:|---:|---|"]
    for row in aggregate.get("per_seed", []):
        seed = row.get("seed")
        med = row.get("module03_mediation_summary") or {}
        norm = row.get("module04_normative_metrics") or {}
        rt = row.get("module04_rt_linkage") or {}

        med_mean = _to_float(med.get("effect_mean"))
        healthy_nll = _to_float((norm.get("healthy") or {}).get("nll"))
        healthy_beta = _to_float(((rt.get("linkage") or {}).get("healthy") or {}).get("mean_beta_margin"))
        checksum = str(norm.get("seed_effect_checksum", ""))[:12]

        seed_table_rows.append(f"| {seed} | {med_mean:.6g} | {healthy_nll:.6g} | {healthy_beta:.6g} | {checksum} |")

    agg_json = json.dumps(aggregate, indent=2)

    md = []
    md.append("# AUDIT SUMMARY")
    md.append("")
    md.append(f"- Output root: `{out_root}`")
    md.append(f"- Seeds: `{','.join(str(s) for s in seeds)}`")
    md.append(f"- Pipeline git commit: `{git_commit}`")
    md.append("")
    md.append("## Repo Fingerprint")
    md.append("```json")
    md.append(json.dumps(repo_fingerprint, indent=2))
    md.append("```")
    md.append("")
    md.append("## Dataset Hashes")
    md.extend(dataset_hash_lines)
    md.append("")

    if lawc_audit is not None:
        md.append("## Law-C Audit")
        md.append(f"- pass: `{lawc_audit.get('pass')}`")
        for row in lawc_audit.get("datasets", []):
            md.append(
                "- "
                f"{row.get('dataset_id')}: median_rho={_to_float(row.get('median_rho')):.6g}, "
                f"p={_to_float(row.get('p_value')):.6g}, q={_to_float(row.get('q_value')):.6g}, "
                f"x_control={row.get('x_control_degrade_pass')}, y_control={row.get('y_control_degrade_pass')}"
            )
        md.append("")

    md.append("## Per-Seed Snapshot")
    md.extend(seed_table_rows)
    md.append("")
    md.append("## Aggregate Metrics")
    md.append("```json")
    md.append(agg_json)
    md.append("```")
    md.append("")
    md.append("## CUDA Info")
    md.append("```text")
    md.append(cuda_info)
    md.append("```")
    md.append("")
    md.append("## Pip Freeze")
    md.append("```text")
    md.append(pip_freeze)
    md.append("```")
    md.append("")
    md.append("## Conda List")
    md.append("```text")
    md.append(conda_list)
    md.append("```")
    md.append("")
    return "\n".join(md)


def main() -> None:
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    aggregate = _collect_seed_results(args.out_root, seeds)

    lawc_json_path = args.out_root / "lawc_audit" / "locked_test_results.json"
    lawc_audit = _read_json(lawc_json_path)

    repo_root = Path(__file__).resolve().parent
    repo_fingerprint = _compute_repo_fingerprint(repo_root)
    repo_fingerprint_path = args.out_root / "AUDIT" / "repo_fingerprint.json"
    repo_fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    repo_fingerprint_path.write_text(json.dumps(repo_fingerprint, indent=2), encoding="utf-8")

    aggregate["lawc_audit"] = lawc_audit
    aggregate["repo_fingerprint"] = repo_fingerprint

    agg_path = args.out_root / "aggregate_results.json"
    agg_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    stage_manifest = None
    if args.stage_manifest and args.stage_manifest.exists():
        try:
            stage_manifest = json.loads(args.stage_manifest.read_text(encoding="utf-8"))
        except Exception:
            stage_manifest = None

    audit_md = _audit_summary_md(
        out_root=args.out_root,
        seeds=seeds,
        aggregate=aggregate,
        stage_manifest=stage_manifest,
        repo_fingerprint=repo_fingerprint,
        lawc_audit=lawc_audit,
    )
    audit_path = args.out_root / "AUDIT_SUMMARY.md"
    audit_path.write_text(audit_md, encoding="utf-8")

    print(f"Wrote aggregate results: {agg_path}")
    print(f"Wrote audit summary: {audit_path}")
    print(f"Wrote repo fingerprint: {repo_fingerprint_path}")

    det_fail = bool(
        ((aggregate.get("normative_metrics") or {}).get("deterministic_seed_checksum_fail_flag", False))
    )
    if det_fail and not bool(args.allow_deterministic):
        raise RuntimeError(
            "Deterministic fail-closed: all normative seed_effect_checksums are identical across seeds. "
            "Re-run with meaningful stochasticity or pass --allow_deterministic to override."
        )


if __name__ == "__main__":
    main()
