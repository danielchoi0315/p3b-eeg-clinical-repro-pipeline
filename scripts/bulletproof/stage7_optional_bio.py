#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import command_env, ensure_out_tree, ensure_stage_status, find_repo, run_cmd, stop_reason


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, default=Path("/lambda/nfs/HCog/filesystemHcog/openneuro"))
    return ap.parse_args()


def _run_opt(repo: Path, cmd: list[str], log: Path, stop: Path, title: str) -> dict:
    try:
        run_cmd(cmd, cwd=repo, env=command_env(repo), log_path=log)
        return {"status": "PASS", "cmd": cmd, "log": str(log)}
    except Exception as exc:
        stop_reason(stop, title, str(exc), diagnostics={"cmd": cmd, "log": str(log)})
        return {"status": "SKIP", "reason": str(exc), "cmd": cmd, "log": str(log), "stop_reason": str(stop)}


def main() -> int:
    args = parse_args()
    out_root = args.out_root
    paths = ensure_out_tree(out_root)
    audit = paths["AUDIT"]
    out_dir = paths["PACK_BIO"]
    out_dir.mkdir(parents=True, exist_ok=True)

    stage3 = audit / "stage3_match_check_summary.json"
    if not stage3.exists() or json.loads(stage3.read_text(encoding="utf-8")).get("status") != "PASS":
        stop_reason(audit / "STOP_REASON_stage7_optional_bio.md", "stage7_optional_bio", "Blocked because stage3 is not PASS")
        ensure_stage_status(audit, "stage7_optional_bio", "SKIP", {"reason": "blocked_by_stage3"})
        return 0

    repo = find_repo()
    if repo is None:
        stop_reason(audit / "STOP_REASON_stage7_optional_bio.md", "stage7_optional_bio", "repo_not_found")
        ensure_stage_status(audit, "stage7_optional_bio", "FAIL", {"reason": "repo_not_found"})
        return 1

    results = {}

    ds004752 = args.data_root / "ds004752"
    if ds004752.exists():
        results["ds004752"] = _run_opt(
            repo,
            ["python3", str(repo / "scripts" / "decode_ds004752.py"), "--data_root", str(args.data_root), "--out_dir", str(out_dir / "ds004752")],
            out_dir / "ds004752.log",
            out_dir / "STOP_REASON_ds004752.md",
            "stage7_optional_bio_ds004752",
        )
    else:
        stop_reason(out_dir / "STOP_REASON_ds004752.md", "stage7_optional_bio_ds004752", "dataset missing")
        results["ds004752"] = {"status": "SKIP", "reason": "dataset_missing"}

    ds007262 = args.data_root / "ds007262"
    if ds007262.exists():
        results["ds007262"] = _run_opt(
            repo,
            ["python3", str(repo / "scripts" / "decode_ds007262.py"), "--data_root", str(args.data_root), "--out_dir", str(out_dir / "ds007262")],
            out_dir / "ds007262.log",
            out_dir / "STOP_REASON_ds007262.md",
            "stage7_optional_bio_ds007262",
        )
    else:
        stop_reason(out_dir / "STOP_REASON_ds007262.md", "stage7_optional_bio_ds007262", "dataset missing")
        results["ds007262"] = {"status": "SKIP", "reason": "dataset_missing"}

    (out_dir / "optional_bio_status.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ensure_stage_status(audit, "stage7_optional_bio", "PASS", {"status_json": str(out_dir / "optional_bio_status.json")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
