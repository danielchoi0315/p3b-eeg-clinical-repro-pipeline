#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from common import (
    CORE_LAWC_DATASETS,
    CLINICAL_DATASETS,
    MECHANISM_DATASETS,
    detect_slurm,
    ensure_out_tree,
    ensure_stage_status,
    find_expected_dataset_hashes_file,
    find_expected_results_table_file,
    find_repo,
    iso_now,
    out_root_default,
    parse_dataset_hashes_payload,
    parse_expected_confirmatory_metrics,
    resolve_expected_kit,
    sha256_file,
    stop_reason,
    unpack_zip,
    write_json,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path, default=out_root_default())
    ap.add_argument("--expected_kit", type=Path, default=None)
    return ap.parse_args()


def _cmd_text(cmd: List[str]) -> str:
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    return r.stdout


def _validate_expected_metrics(metrics: Dict[str, float]) -> List[str]:
    missing: List[str] = []
    for ds in CORE_LAWC_DATASETS:
        for suf in ["rho", "p", "q", "n", "n_perm_done"]:
            key = f"core_lawc.{ds}.{suf}"
            if key not in metrics:
                missing.append(key)

    for ds in CLINICAL_DATASETS:
        if not any(k.startswith(f"clinical.{ds}.") for k in metrics):
            missing.append(f"clinical.{ds}.*")

    for ds in MECHANISM_DATASETS:
        if not any(k.startswith(f"mechanism.{ds}.") or k.startswith("mechanism.") for k in metrics):
            missing.append(f"mechanism.{ds}.*")

    return missing


def _find_expectedkit_metrics_csv(expected_ref: Path) -> Path | None:
    preferred = [
        expected_ref / "expected_confirmatory_metrics.csv",
        expected_ref / "EXPECTED_CONFIRMATORY_METRICS.csv",
    ]
    for p in preferred:
        if p.exists():
            return p
    cands = sorted(expected_ref.rglob("expected_confirmatory_metrics.csv"))
    if cands:
        return cands[0]
    cands2 = sorted([p for p in expected_ref.rglob("*.csv") if "confirmatory" in p.name.lower() and "metric" in p.name.lower()])
    return cands2[0] if cands2 else None


def main() -> int:
    args = parse_args()
    out_root = args.out_root
    paths = ensure_out_tree(out_root)
    audit = paths["AUDIT"]

    repo = find_repo()
    if repo is None:
        stop_reason(audit / "STOP_REASON.md", "stage0_preflight", "Could not locate repository root with scripts/bulletproof/master.py")
        ensure_stage_status(audit, "stage0_preflight", "FAIL", {"reason": "repo_not_found"})
        return 2

    expected_kit, search_hits = resolve_expected_kit(args.expected_kit)
    if expected_kit is None:
        stop_reason(
            audit / "STOP_REASON_expected_kit_missing.md",
            "stage0_preflight",
            "Expected kit zip not found. Stage0 requires --expected_kit PATH or discoverable MANUSCRIPT kit zip under /filesystemHcog/runs.",
            diagnostics={
                "how_to_supply": "bash scripts/bulletproof/run_master.sh --out_root <OUT_ROOT> --expected_kit /path/to/MANUSCRIPT_KIT.zip",
                "search_hits": [str(p) for p in search_hits[:20]],
            },
        )
        ensure_stage_status(audit, "stage0_preflight", "FAIL", {"reason": "expected_kit_missing"})
        return 1

    expected_kit = expected_kit.resolve()
    expected_kit_hash = sha256_file(expected_kit)

    expected_ref = audit / "expected_kit_ref"
    unpack_zip(expected_kit, expected_ref)

    ds_hashes_path = find_expected_dataset_hashes_file(expected_ref)
    expectedkit_metrics_path = _find_expectedkit_metrics_csv(expected_ref)
    results_table_path = expectedkit_metrics_path or find_expected_results_table_file(expected_ref)

    missing_parts: List[str] = []
    if ds_hashes_path is None:
        missing_parts.append("dataset_hashes.json")
    if results_table_path is None:
        missing_parts.append("expected_confirmatory_metrics.csv or FINAL_RESULTS_TABLE.csv")

    expected_commits: Dict[str, str] = {}
    expected_metrics: Dict[str, float] = {}

    if ds_hashes_path is not None:
        payload = json.loads(ds_hashes_path.read_text(encoding="utf-8"))
        expected_commits = parse_dataset_hashes_payload(payload)

    for ds in CORE_LAWC_DATASETS + MECHANISM_DATASETS + CLINICAL_DATASETS:
        commit = expected_commits.get(ds, "")
        if not commit:
            missing_parts.append(f"dataset_hash:{ds}")

    if results_table_path is not None:
        expected_metrics = parse_expected_confirmatory_metrics(results_table_path)
        metric_missing = _validate_expected_metrics(expected_metrics)
        missing_parts.extend([f"confirmatory_metric:{m}" for m in metric_missing])

    if missing_parts:
        stop_reason(
            audit / "STOP_REASON_expected_kit_incomplete.md",
            "stage0_preflight",
            "Expected kit is incomplete for strict confirmatory matching.",
            diagnostics={
                "expected_kit": str(expected_kit),
                "missing_parts": missing_parts,
                "dataset_hashes_path": str(ds_hashes_path) if ds_hashes_path else "",
                "results_table_path": str(results_table_path) if results_table_path else "",
                "expectedkit_metrics_path": str(expectedkit_metrics_path) if expectedkit_metrics_path else "",
            },
        )
        ensure_stage_status(
            audit,
            "stage0_preflight",
            "FAIL",
            {
                "reason": "expected_kit_incomplete",
                "expected_kit": str(expected_kit),
                "missing_parts": missing_parts,
            },
        )
        return 1

    write_json(audit / "expected_dataset_hashes.json", {"datasets": expected_commits})
    write_json(audit / "expected_confirmatory_metrics.json", {"metrics": expected_metrics})

    expected_manifest = {
        "expected_kit_zip": str(expected_kit),
        "expected_kit_sha256": expected_kit_hash,
        "expected_kit_ref": str(expected_ref),
        "dataset_hashes_path": str(ds_hashes_path),
        "confirmatory_table_path": str(results_table_path),
        "expectedkit_metrics_path": str(expectedkit_metrics_path) if expectedkit_metrics_path else "",
        "mode": "expectedkit" if expectedkit_metrics_path else "manuscriptkit",
        "n_expected_commits": len(expected_commits),
        "n_expected_confirmatory_metrics": len(expected_metrics),
    }
    write_json(audit / "expected_kit_manifest.json", expected_manifest)

    env = {
        "timestamp_utc": iso_now(),
        "hostname": platform.node(),
        "uname": platform.uname()._asdict(),
        "python": {
            "executable": shutil.which("python3") or "",
            "version": platform.python_version(),
        },
        "nvidia_smi": _cmd_text(["bash", "-lc", "command -v nvidia-smi && nvidia-smi || true"]),
        "pip_freeze": _cmd_text(["bash", "-lc", "python3 -m pip freeze || true"]),
        "repo_root": str(repo),
        "out_root": str(out_root),
        "slurm": detect_slurm(),
        "expected_kit": {
            "path": str(expected_kit),
            "sha256": expected_kit_hash,
            "search_hits": [str(p) for p in search_hits[:20]],
            "manifest": expected_manifest,
        },
        "storage_policy": {
            "SLURM_TMPDIR": os.environ.get("SLURM_TMPDIR", ""),
            "filesystemHcog_exists": Path("/filesystemHcog").exists(),
        },
    }
    write_json(audit / "preflight_env.json", env)

    ensure_stage_status(
        audit,
        "stage0_preflight",
        "PASS",
        {
            "repo_root": str(repo),
            "expected_kit": str(expected_kit),
            "expected_kit_sha256": expected_kit_hash,
            "n_expected_confirmatory_metrics": len(expected_metrics),
            "slurm_reachable": bool(env["slurm"]["reachable"]),
        },
    )

    status_payload = {
        "repo_root": str(repo),
        "out_root": str(out_root),
        "expected_kit": str(expected_kit),
        "expected_kit_sha256": expected_kit_hash,
    }
    write_json(audit / "run_status.json", status_payload)
    print(json.dumps(status_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
