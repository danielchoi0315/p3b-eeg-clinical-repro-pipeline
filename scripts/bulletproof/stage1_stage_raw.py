#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

from common import (
    REQUIRED_CONFIRMATORY_DATASETS,
    ensure_out_tree,
    ensure_stage_status,
    find_repo,
    parse_dataset_hashes_payload,
    read_json,
    run_cmd,
    stage_dataset,
    stop_reason,
    write_json,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, default=Path("/lambda/nfs/HCog/filesystemHcog/openneuro"))
    ap.add_argument("--expected_kit", type=Path, default=None)
    return ap.parse_args()


def _load_expected_commits(audit: Path) -> Dict[str, str]:
    p1 = audit / "expected_dataset_hashes.json"
    if p1.exists():
        payload = read_json(p1)
        ds_map = payload.get("datasets", {})
        if isinstance(ds_map, dict):
            return {str(k): str(v) for k, v in ds_map.items() if str(k) and str(v)}

    p2 = audit / "expected_kit_ref" / "PROVENANCE" / "dataset_hashes.json"
    if not p2.exists():
        p2 = audit / "expected_kit_ref" / "dataset_hashes.json"
    if p2.exists():
        payload = read_json(p2)
        return parse_dataset_hashes_payload(payload)

    return {}


def _git_url(dataset_id: str) -> str:
    return f"https://github.com/OpenNeuroDatasets/{dataset_id}.git"


def main() -> int:
    args = parse_args()
    out_root = args.out_root
    paths = ensure_out_tree(out_root)
    audit = paths["AUDIT"]
    log_path = audit / "stage1_stage_raw.log"

    repo = find_repo()
    if repo is None:
        stop_reason(audit / "STOP_REASON_stage1_stage_raw.md", "stage1_stage_raw", "Repository not found")
        ensure_stage_status(audit, "stage1_stage_raw", "FAIL", {"reason": "repo_not_found"})
        return 2

    expected_commits = _load_expected_commits(audit)
    missing_expected = [ds for ds in REQUIRED_CONFIRMATORY_DATASETS if not expected_commits.get(ds)]
    if missing_expected:
        stop_reason(
            audit / "STOP_REASON_dataset_commit_mismatch.md",
            "stage1_stage_raw",
            "Expected dataset commit SHAs are missing from expected kit reference.",
            diagnostics={"missing_expected_commits": missing_expected},
        )
        ensure_stage_status(audit, "stage1_stage_raw", "FAIL", {"reason": "missing_expected_commits", "datasets": missing_expected})
        return 1

    datalad_bin = ""
    datalad_cands = [
        Path(os.environ.get("DATALAD_BIN", "")),
        Path("/usr/bin/datalad"),
        Path("/lambda/nfs/HCog/filesystemHcog/venvs/research_pipeline/bin/datalad"),
        repo / ".venv" / "bin" / "datalad",
        repo / ".venv_bulletproof" / "bin" / "datalad",
    ]
    for p in sorted(Path("/lambda/nfs/HCog/filesystemHcog/venvs").glob("*/bin/datalad")):
        datalad_cands.append(p)
    for cand in datalad_cands:
        if not str(cand).strip():
            continue
        if not cand.exists():
            continue
        chk = run_cmd([str(cand), "--version"], allow_fail=True)
        if chk.rc == 0:
            datalad_bin = str(cand)
            break

    staged: List[Dict[str, Any]] = []
    failures: List[str] = []
    commit_mismatches: List[Dict[str, str]] = []

    for did in REQUIRED_CONFIRMATORY_DATASETS:
        expected = str(expected_commits.get(did, "")).strip()
        try:
            rec = stage_dataset(
                dataset_id=did,
                git_url=_git_url(did),
                data_root=args.data_root,
                pinned_hash=expected,
                datalad_bin=datalad_bin or None,
                log_path=log_path,
            )
            rec["expected_commit"] = expected

            if rec.get("checked_out_commit", "") != expected:
                commit_mismatches.append(
                    {
                        "dataset_id": did,
                        "expected": expected,
                        "observed": str(rec.get("checked_out_commit", "")),
                    }
                )

            if not rec.get("dataset_description_exists", False):
                failures.append(f"{did}: missing dataset_description.json")
            if not rec.get("participants_exists", False):
                failures.append(f"{did}: missing participants.tsv")
            if int(rec.get("eeg_header_count", 0)) <= 0:
                failures.append(f"{did}: no EEG headers found after staging")

            staged.append(rec)
        except Exception as exc:
            failures.append(f"{did}: {exc}")

    payload = {
        "data_root": str(args.data_root),
        "datasets": staged,
        "failures": failures,
        "commit_mismatches": commit_mismatches,
    }
    write_json(audit / "dataset_hashes.json", payload)

    if commit_mismatches:
        stop_reason(
            audit / "STOP_REASON_dataset_commit_mismatch.md",
            "stage1_stage_raw",
            "One or more datasets could not be checked out at the expected commit.",
            diagnostics={"commit_mismatches": commit_mismatches, "log": str(log_path)},
        )
        ensure_stage_status(audit, "stage1_stage_raw", "FAIL", {"reason": "dataset_commit_mismatch", "n": len(commit_mismatches)})
        return 1

    if failures:
        stop_reason(
            audit / "STOP_REASON_stage1_stage_raw.md",
            "stage1_stage_raw",
            "One or more required datasets failed staging/validation.",
            diagnostics={"failures": failures},
        )
        ensure_stage_status(audit, "stage1_stage_raw", "FAIL", {"failures": failures})
        return 1

    ensure_stage_status(
        audit,
        "stage1_stage_raw",
        "PASS",
        {
            "n_datasets": len(staged),
            "data_root": str(args.data_root),
            "commit_match_verified": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
