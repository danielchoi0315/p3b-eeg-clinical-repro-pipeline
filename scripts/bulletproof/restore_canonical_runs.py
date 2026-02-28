#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

REQUIRED_PATTERNS = {
    "canonical_v2": "*_NN_FINAL_MEGA_V2_BIO*",
    "master_v1": "*_NN_FINAL_MASTER_V1*",
    "postfinal_tighten": "*_POSTFINAL_TIGHTEN*",
}

ARCHIVE_PATTERNS = [
    "results_only.tar.gz",
    "*POSTFINAL_TIGHTEN*.zip",
    "*NN_FINAL_MASTER_V1*.zip",
    "*NN_FINAL_MEGA_V2_BIO*.zip",
]


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_stop_reason(path: Path, title: str, why: str, diagnostics: Dict[str, Any]) -> None:
    lines = [
        f"# STOP_REASON {title}",
        "",
        "## Why",
        why,
        "",
        "## Diagnostics",
        "```json",
        json.dumps(diagnostics, indent=2),
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scan_required_dirs(runs_root: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for key, pat in REQUIRED_PATTERNS.items():
        cands = [p.resolve() for p in runs_root.glob(pat) if p.is_dir()]
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        out[key] = cands
    return out


def _all_present(found: Dict[str, List[Path]]) -> bool:
    return all(found.get(k) for k in REQUIRED_PATTERNS.keys())


def _find_archives(search_roots: List[Path]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for root in search_roots:
        if not root.exists():
            continue
        qroot = shlex.quote(str(root))
        query = (
            f"find {qroot} -type f \\( "
            "-name 'results_only.tar.gz' "
            "-o -name '*POSTFINAL_TIGHTEN*.zip' "
            "-o -name '*NN_FINAL_MASTER_V1*.zip' "
            "-o -name '*NN_FINAL_MEGA_V2_BIO*.zip' "
            "\\) -printf '%T@ %p\\n' 2>/dev/null"
        )
        p = subprocess.run(["bash", "-lc", query], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        for line in p.stdout.splitlines():
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            try:
                mt = float(parts[0])
            except Exception:
                continue
            path = Path(parts[1]).expanduser()
            if not path.exists() or not path.is_file():
                continue
            sp = str(path.resolve())
            if sp in seen:
                continue
            seen.add(sp)
            hits.append({"path": sp, "mtime": mt, "name": path.name})

    hits.sort(key=lambda x: float(x.get("mtime", 0.0)), reverse=True)
    return hits


def _name_matches_required(name: str) -> Dict[str, bool]:
    out: Dict[str, bool] = {k: False for k in REQUIRED_PATTERNS}
    for k, pat in REQUIRED_PATTERNS.items():
        if fnmatch.fnmatch(name, pat):
            out[k] = True
    return out


def _scan_archive_members_for_required(names: List[str]) -> Dict[str, bool]:
    out: Dict[str, bool] = {k: False for k in REQUIRED_PATTERNS}
    for raw in names:
        p = PurePosixPath(raw)
        for part in p.parts:
            mm = _name_matches_required(part)
            for k, v in mm.items():
                out[k] = out[k] or v
        mm2 = _name_matches_required(p.name)
        for k, v in mm2.items():
            out[k] = out[k] or v
    return out


def _restore_from_tarball(tarball: Path, runs_root: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"tarball": str(tarball), "used": False, "member_pattern_hits": {k: False for k in REQUIRED_PATTERNS}}
    try:
        with tarfile.open(tarball, "r:gz") as tf:
            members = tf.getnames()
            hits = _scan_archive_members_for_required(members)
            info["member_pattern_hits"] = hits
            info["n_members"] = len(members)
            if not any(hits.values()):
                return info
            tf.extractall(runs_root)
            info["used"] = True
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _find_required_dirs_under(root: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {k: [] for k in REQUIRED_PATTERNS}
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        nm = p.name
        for k, pat in REQUIRED_PATTERNS.items():
            if fnmatch.fnmatch(nm, pat):
                out[k].append(p.resolve())
    for k in out:
        out[k].sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _restore_from_zips(zip_paths: List[Path], runs_root: Path) -> Dict[str, Any]:
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    restore_root = runs_root / f"_RESTORED_FROM_ZIPS_{ts}"
    restore_root.mkdir(parents=True, exist_ok=True)

    extracted: List[Dict[str, Any]] = []
    found_pool: Dict[str, List[Path]] = {k: [] for k in REQUIRED_PATTERNS}

    for zp in zip_paths:
        out = restore_root / _safe_name(zp.stem)
        out.mkdir(parents=True, exist_ok=True)
        rec: Dict[str, Any] = {"zip": str(zp), "extract_dir": str(out), "ok": False}
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(out)
            rec["ok"] = True
            found_here = _find_required_dirs_under(out)
            rec["found_required_dirs"] = {k: [str(p) for p in v] for k, v in found_here.items() if v}
            for k, vv in found_here.items():
                found_pool[k].extend(vv)
        except Exception as exc:
            rec["error"] = str(exc)
        extracted.append(rec)

    symlinks: Dict[str, str] = {}
    for k, paths in found_pool.items():
        if not paths:
            continue
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        tgt = paths[0]
        link = runs_root / tgt.name
        try:
            if link.exists() or link.is_symlink():
                if link.resolve() == tgt.resolve():
                    symlinks[k] = str(link)
                    continue
                # keep existing directory untouched
                continue
            os.symlink(str(tgt), str(link))
            symlinks[k] = str(link)
        except Exception:
            continue

    return {
        "restore_root": str(restore_root),
        "zip_extractions": extracted,
        "symlinks": symlinks,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=Path, required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    runs_root = args.runs_root.expanduser().resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    found0 = _scan_required_dirs(runs_root)
    if _all_present(found0):
        payload = {k: str(v[0]) for k, v in found0.items()}
        print(json.dumps({"status": "already_present", "paths": payload}, indent=2))
        return 0

    search_roots = [runs_root, Path("/lambda/nfs/HCog")]
    home = Path(os.environ.get("HOME", "")).expanduser()
    if str(home):
        search_roots.append(home)
    search_roots.append(Path("/scratch"))

    archives = _find_archives(search_roots)
    archive_paths = [Path(a["path"]) for a in archives]
    tarballs = [p for p in archive_paths if p.name == "results_only.tar.gz"]
    zips = [p for p in archive_paths if p.suffix.lower() == ".zip"]

    manifest: Dict[str, Any] = {
        "timestamp_utc": _iso_now(),
        "runs_root": str(runs_root),
        "search_roots": [str(p) for p in search_roots],
        "archives_found": archives[:200],
        "tar_restore": {},
        "zip_restore": {},
    }

    # Preferred restore via results_only tarball
    if tarballs:
        tar_info = _restore_from_tarball(tarballs[0], runs_root)
        manifest["tar_restore"] = tar_info

    found1 = _scan_required_dirs(runs_root)
    if not _all_present(found1) and zips:
        manifest["zip_restore"] = _restore_from_zips(zips, runs_root)

    found2 = _scan_required_dirs(runs_root)
    manifest["final_found"] = {k: [str(p) for p in v[:5]] for k, v in found2.items()}

    _write_json(runs_root / "_RESTORE_MANIFEST.json", manifest)

    if _all_present(found2):
        payload = {k: str(v[0]) for k, v in found2.items()}
        print(json.dumps({"status": "restored", "paths": payload, "manifest": str(runs_root / '_RESTORE_MANIFEST.json')}, indent=2))
        return 0

    missing = [k for k, v in found2.items() if not v]
    incoming = runs_root / "_INCOMING" / "results_only.tar.gz"
    stop = runs_root / "STOP_REASON_restore_missing_archives.md"
    _write_stop_reason(
        stop,
        "restore_canonical_runs",
        "Could not restore all required canonical run roots from available archives.",
        {
            "missing_required_roots": missing,
            "required_patterns": REQUIRED_PATTERNS,
            "searched_paths": [str(p) for p in search_roots],
            "archives_found_top20": archives[:20],
            "import_commands": {
                "scp": f"scp user@host:/path/to/results_only.tar.gz {incoming}",
                "globus": f"Place results_only.tar.gz under {runs_root / '_INCOMING'} then rerun restore.",
            },
            "manifest": str(runs_root / "_RESTORE_MANIFEST.json"),
        },
    )
    print(str(stop))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
