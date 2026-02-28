#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import ensure_out_tree, ensure_stage_status, sha256_file, stop_reason, tar_results_only, write_text


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path, required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_root = args.out_root
    paths = ensure_out_tree(out_root)
    audit = paths["AUDIT"]

    tar_path = paths["TARBALLS"] / "results_only.tar.gz"
    try:
        n = tar_results_only(out_root, tar_path)
        digest = sha256_file(tar_path)
        write_text(paths["TARBALLS"] / "results_only.tar.gz.sha256", f"{digest}  {tar_path.name}\n")
    except Exception as exc:
        stop_reason(audit / "STOP_REASON_stage9_tarball.md", "stage9_tarball", str(exc))
        ensure_stage_status(audit, "stage9_tarball", "FAIL", {"error": str(exc)})
        return 1

    ensure_stage_status(
        audit,
        "stage9_tarball",
        "PASS",
        {"tarball": str(tar_path), "sha256": digest, "n_files": n},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
