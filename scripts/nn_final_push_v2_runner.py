#!/usr/bin/env python3
"""NN_FINAL_PUSH_V2 end-to-end orchestrator.

Stages
0) preflight
1) compile_gate
2) stage_datasets
3) core_lawc_ultradeep
4) mechanism_deep_auditfix
5) clinical_PD_ds003490_FULL
6) clinical_Dementia_ds004504_REFRESH
7) final_report
8) zip_bundle

Fail-closed rules:
- Hard FAIL on compile_gate and staging failures.
- BIDS oddball mapping must be explicit; ambiguous mapping -> SKIP with STOP_REASON.
- PD SKIP may not emit endpoint or figure artefacts.
- NaN q-values in mechanism/clinical tables are disallowed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from common.lawc_audit import bh_fdr  # noqa: E402

try:
    from aggregate_results import _compute_repo_fingerprint
except Exception:
    _compute_repo_fingerprint = None


DATASET_URLS: Dict[str, str] = {
    "ds003490": "https://github.com/OpenNeuroDatasets/ds003490.git",
    "ds004504": "https://github.com/OpenNeuroDatasets/ds004504.git",
}

REQUIRED_DATASETS: List[str] = ["ds003490", "ds004504"]

CORE_STERNBERG_DATASETS: List[str] = ["ds005095", "ds003655", "ds004117"]

STAGE_ORDER: List[str] = [
    "preflight",
    "compile_gate",
    "stage_datasets",
    "core_lawc_ultradeep",
    "mechanism_deep_auditfix",
    "clinical_PD_ds003490_FULL",
    "clinical_Dementia_ds004504_REFRESH",
    "final_report",
    "zip_bundle",
]


@dataclass
class RunContext:
    out_root: Path
    audit_dir: Path
    outzip_dir: Path
    pack_core: Path
    pack_mechanism: Path
    pack_pd: Path
    pack_dementia: Path

    data_root: Path
    features_root_core: Path
    features_root_mechanism: Path
    config: Path
    lawc_event_map: Path
    mechanism_event_map: Path
    reuse_out_root: Optional[Path]

    wall_hours: float
    lawc_n_perm: int
    mechanism_n_perm: int
    pd_n_perm: int
    dementia_n_perm: int

    gpu_parallel_procs: int
    cpu_workers: int
    resume: bool

    start_ts: float
    deadline_ts: float

    stage_records: List[Dict[str, Any]]
    stage_status: Dict[str, str]
    monitor_proc: Optional[subprocess.Popen]
    monitor_handle: Optional[Any]
    runtime_env: Dict[str, str]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _tail(path: Path, n: int = 160) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max(1, int(n)) :])


def _run_cmd(
    cmd: List[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    allow_fail: bool = False,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{_iso_now()}] CMD: {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            check=False,
        )
    rc = int(proc.returncode)
    if rc != 0 and not allow_fail:
        return rc
    return rc


def _record_stage(
    ctx: RunContext,
    *,
    stage: str,
    status: str,
    rc: int,
    started: float,
    log_path: Path,
    summary_path: Path,
    command: str,
    outputs: Optional[Sequence[Path]] = None,
    error: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ended = time.time()
    rec = {
        "stage": stage,
        "status": status,
        "returncode": int(rc),
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "ended_at": datetime.fromtimestamp(ended, tz=timezone.utc).isoformat(),
        "elapsed_sec": float(max(0.0, ended - started)),
        "log": str(log_path),
        "summary": str(summary_path),
        "command": command,
        "outputs": [str(p) for p in (outputs or [])],
        "error": error,
    }
    if extra:
        rec.update(extra)
    _write_json(summary_path, rec)
    _write_text(ctx.audit_dir / f"{stage}.status", status + "\n")
    ctx.stage_records.append(rec)
    ctx.stage_status[stage] = status
    return rec


def _seconds_left(ctx: RunContext) -> float:
    return float(ctx.deadline_ts - time.time())


def _wall_guard(ctx: RunContext, *, reserve_sec: float) -> bool:
    return _seconds_left(ctx) > float(reserve_sec)


def _start_gpu_monitor(ctx: RunContext) -> None:
    out_csv = ctx.audit_dir / "nvidia_smi_1hz.csv"
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
        "-l",
        "1",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    handle = out_csv.open("w", encoding="utf-8")
    ctx.monitor_handle = handle
    ctx.monitor_proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=handle,
        stderr=handle,
        text=True,
    )


def _stop_gpu_monitor(ctx: RunContext) -> None:
    p = ctx.monitor_proc
    if p is not None:
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=8)
        finally:
            ctx.monitor_proc = None
    h = ctx.monitor_handle
    if h is not None:
        try:
            h.close()
        except Exception:
            pass
        ctx.monitor_handle = None


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_seeds(spec: str) -> List[int]:
    out: List[int] = []
    for raw in str(spec).split(","):
        t = raw.strip()
        if not t:
            continue
        if "-" in t:
            a, b = t.split("-", 1)
            ai = int(a.strip())
            bi = int(b.strip())
            if ai <= bi:
                out.extend(range(ai, bi + 1))
            else:
                out.extend(range(ai, bi - 1, -1))
        else:
            out.append(int(t))
    return sorted(set(out))


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stage_preflight(ctx: RunContext) -> Dict[str, Any]:
    stage = "preflight"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"
    outputs: List[Path] = []
    error = ""

    try:
        ctx.out_root.mkdir(parents=True, exist_ok=True)
        ctx.audit_dir.mkdir(parents=True, exist_ok=True)
        ctx.outzip_dir.mkdir(parents=True, exist_ok=True)
        ctx.pack_core.mkdir(parents=True, exist_ok=True)
        ctx.pack_mechanism.mkdir(parents=True, exist_ok=True)
        ctx.pack_pd.mkdir(parents=True, exist_ok=True)
        ctx.pack_dementia.mkdir(parents=True, exist_ok=True)

        pyv = ctx.audit_dir / "python_version.txt"
        _write_text(pyv, sys.version + "\n")
        outputs.append(pyv)

        pip_freeze = ctx.audit_dir / "pip_freeze.txt"
        p = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
        _write_text(pip_freeze, p.stdout if p.returncode == 0 else p.stdout + "\n" + p.stderr)
        outputs.append(pip_freeze)

        nv_l = ctx.audit_dir / "nvidia_smi_L.txt"
        p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
        _write_text(nv_l, p.stdout if p.returncode == 0 else p.stdout + "\n" + p.stderr)
        outputs.append(nv_l)

        nv_s = ctx.audit_dir / "nvidia_smi_snapshot.csv"
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        _write_text(nv_s, p.stdout if p.returncode == 0 else p.stdout + "\n" + p.stderr)
        outputs.append(nv_s)

        repo_fp = (
            _compute_repo_fingerprint(REPO_ROOT)
            if _compute_repo_fingerprint is not None
            else {
                "repo_root": str(REPO_ROOT),
                "git_head": None,
                "repo_fingerprint_sha256": None,
                "file_count_hashed": 0,
            }
        )
        repo_fp_path = ctx.audit_dir / "repo_fingerprint.json"
        _write_json(repo_fp_path, repo_fp)
        outputs.append(repo_fp_path)

        _start_gpu_monitor(ctx)

        return _record_stage(
            ctx,
            stage=stage,
            status="PASS",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="preflight",
            outputs=outputs,
        )
    except Exception as exc:
        error = str(exc)
        _write_text(log_path, traceback.format_exc())
        _stop_gpu_monitor(ctx)
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="preflight",
            outputs=outputs,
            error=error,
        )


def _stage_compile_gate(ctx: RunContext) -> Dict[str, Any]:
    stage = "compile_gate"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"
    cmd = ["bash", "-lc", "find . -name '*.py' -print0 | xargs -0 python -m py_compile"]
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, env=ctx.runtime_env)
    status = "PASS" if rc == 0 else "FAIL"
    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=rc,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="find . -name '*.py' -print0 | xargs -0 python -m py_compile",
        error="" if status == "PASS" else "compile gate failed",
    )


def _count_event_files(dataset_root: Path) -> int:
    return int(len(list(dataset_root.rglob("*_events.tsv"))) + len(list(dataset_root.rglob("*_events.tsv.gz"))))


_EEG_PAT = re.compile(r"_eeg\.(edf|bdf|set|vhdr|eeg|fif|gdf|cnt|fdt)(\.gz)?$", re.IGNORECASE)


def _count_eeg_payload_files(dataset_root: Path) -> int:
    n = 0
    for p in dataset_root.rglob("*"):
        if not p.is_file():
            continue
        if _EEG_PAT.search(p.name):
            n += 1
    return int(n)


def _count_eeg_broken_symlinks(dataset_root: Path) -> int:
    n = 0
    for p in dataset_root.rglob("*"):
        try:
            if p.is_symlink() and _EEG_PAT.search(p.name) and not p.exists():
                n += 1
        except Exception:
            continue
    return int(n)


def _count_sternberg_logs(dataset_root: Path) -> int:
    n = 0
    for p in dataset_root.rglob("*"):
        if not p.is_file():
            continue
        nm = p.name.lower()
        if "sternberg" in nm and "events" in nm and (nm.endswith(".txt") or nm.endswith(".csv")):
            n += 1
    return int(n)


def _dataset_ready(dataset_id: str, dataset_root: Path) -> bool:
    if not dataset_root.exists() or not (dataset_root / "participants.tsv").exists():
        return False

    eeg_n = _count_eeg_payload_files(dataset_root)
    if eeg_n <= 0:
        return False

    # Rest-only dataset has no BIDS events.tsv requirement.
    if dataset_id == "ds004504":
        return True

    events_n = _count_event_files(dataset_root)
    if events_n <= 0:
        return False

    # mechanism requires pupil payload in raw dataset
    if dataset_id == "ds003838":
        pupil_n = int(len(list(dataset_root.rglob("*_pupil.tsv"))) + len(list(dataset_root.rglob("*_pupil.tsv.gz"))))
        if pupil_n <= 0:
            return False

    return True


def _git_head(path: Path) -> Optional[str]:
    try:
        p = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if p.returncode == 0:
            return p.stdout.strip()
    except Exception:
        pass
    return None


def _remote_head(url: str) -> str:
    try:
        p = subprocess.run(["git", "ls-remote", url, "HEAD"], capture_output=True, text=True, check=False)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip().split()[0]
    except Exception:
        pass
    return "<unavailable>"


def _sha256_manifest(dataset_root: Path, out_path: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(dataset_root.rglob("*")):
        if not p.is_file():
            continue
        if ".git" in p.parts:
            continue
        rel = p.relative_to(dataset_root).as_posix()
        h = hashlib.sha256()
        try:
            with p.open("rb") as f:
                while True:
                    b = f.read(1024 * 1024)
                    if not b:
                        break
                    h.update(b)
            rows.append({"path": rel, "sha256": h.hexdigest(), "size": int(p.stat().st_size)})
        except Exception:
            rows.append({"path": rel, "sha256": None, "size": int(p.stat().st_size)})
    payload = {"dataset_root": str(dataset_root), "n_files": int(len(rows)), "files": rows}
    _write_json(out_path, payload)
    return payload


def _copy_tree_merge(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        q = dst / rel
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)


def _bh_qvals(pvals: Sequence[float]) -> List[float]:
    arr = np.asarray(list(pvals), dtype=float)
    arr = np.where(np.isfinite(arr), arr, 1.0)
    if arr.size == 0:
        return []
    q = multipletests(arr, alpha=0.05, method="fdr_bh")[1]
    return [float(x) for x in q]


def _clone_with_datalad(dataset_id: str, ds_root: Path, url: str, log_path: Path, env: Dict[str, str]) -> Tuple[int, str]:
    if shutil.which("datalad") is None:
        return 1, "datalad unavailable"

    if not (ds_root / ".git").exists():
        rc = _run_cmd(["datalad", "clone", url, str(ds_root)], cwd=REPO_ROOT, log_path=log_path, env=env, allow_fail=True)
        if rc != 0:
            return rc, "datalad clone failed"

    rc_n = _run_cmd(["datalad", "get", "-n", "."], cwd=ds_root, log_path=log_path, env=env, allow_fail=True)
    rc_g = _run_cmd(["datalad", "get", "-r", "."], cwd=ds_root, log_path=log_path, env=env, allow_fail=True)
    if rc_n == 0 and rc_g == 0:
        return 0, "datalad"
    return max(rc_n, rc_g), "datalad get failed"


def _clone_with_git_annex(dataset_id: str, ds_root: Path, url: str, log_path: Path, env: Dict[str, str], jobs: int) -> Tuple[int, str]:
    if not (ds_root / ".git").exists():
        rc = _run_cmd(["git", "clone", url, str(ds_root)], cwd=REPO_ROOT, log_path=log_path, env=env, allow_fail=True)
        if rc != 0:
            return rc, "git clone failed"

    # Try datalad get from a git clone only if possible; otherwise continue with git-annex get.
    if shutil.which("datalad") is not None:
        _run_cmd(["datalad", "get", "-n", "."], cwd=ds_root, log_path=log_path, env=env, allow_fail=True)
        _run_cmd(["datalad", "get", "-r", "."], cwd=ds_root, log_path=log_path, env=env, allow_fail=True)
        if _dataset_ready(dataset_id, ds_root):
            return 0, "git_clone+datalad_get"

    if shutil.which("git-annex") is None:
        return 1, "git-annex unavailable"

    # In plain git clones, annex may need explicit init before get.
    has_uuid = False
    try:
        p = subprocess.run(["git", "-C", str(ds_root), "config", "--get", "annex.uuid"], capture_output=True, text=True, check=False)
        has_uuid = p.returncode == 0 and bool(p.stdout.strip())
    except Exception:
        has_uuid = False
    if not has_uuid:
        rc = _run_cmd(["git", "annex", "init", f"nn_final_push_{dataset_id}"], cwd=ds_root, log_path=log_path, env=env, allow_fail=True)
        if rc != 0:
            return rc, "git annex init failed"

    # Pull only required EEG payload files to avoid hanging on irrelevant annex objects.
    p = subprocess.run(
        ["git", "annex", "find", "--not", "--in=here"],
        cwd=str(ds_root),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    missing = [x.strip() for x in str(p.stdout).splitlines() if x.strip()]
    if p.returncode != 0:
        return int(p.returncode), "git annex find missing failed"

    eeg_need_re = re.compile(
        r"(^|/)sub-[^/]+(/ses-[^/]+)?/eeg/.*_eeg\.(edf|bdf|set|vhdr|eeg|fif|gdf|cnt|fdt|vmrk)(\.gz)?$",
        flags=re.IGNORECASE,
    )

    if dataset_id in {"ds003490", "ds004504"}:
        missing = [m for m in missing if eeg_need_re.search(m)]

    # For datasets that should have events in working tree, retain matching events if annexed.
    if dataset_id in {"ds005095", "ds003655", "ds004117", "ds003838", "ds004796", "ds003490"}:
        event_need = [m for m in missing if m.lower().endswith("_events.tsv") or m.lower().endswith("_events.tsv.gz")]
        missing = sorted(set(missing + event_need))

    if not missing:
        return 0, "git_clone+annex_get(nothing-missing)"

    batch_size = 8
    for i in range(0, len(missing), batch_size):
        batch = missing[i : i + batch_size]
        # Prefer generic annex routing first; fallback to explicit public remote.
        rc_get = _run_cmd(
            ["git", "annex", "get", "-J", str(max(1, int(jobs))), "--", *batch],
            cwd=ds_root,
            log_path=log_path,
            env=env,
            allow_fail=True,
        )
        if rc_get != 0:
            rc_get = _run_cmd(
                ["git", "annex", "get", "--from", "s3-PUBLIC", "-J", str(max(1, int(jobs))), "--", *batch],
                cwd=ds_root,
                log_path=log_path,
                env=env,
                allow_fail=True,
            )
        if rc_get != 0:
            return rc_get, "git annex get batch failed"
    return 0, "git_clone+annex_get"


def _stage_stage_datasets(ctx: RunContext) -> Dict[str, Any]:
    stage = "stage_datasets"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"
    out_hashes = ctx.audit_dir / "dataset_hashes.json"

    results: List[Dict[str, Any]] = []
    failures: List[str] = []

    try:
        ctx.data_root.mkdir(parents=True, exist_ok=True)
        manifest_dir = ctx.audit_dir / "sha256_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        for dataset_id in REQUIRED_DATASETS:
            url = DATASET_URLS[dataset_id]
            ds_root = ctx.data_root / dataset_id
            method = "existing"
            used_fallback = False

            if not _dataset_ready(dataset_id, ds_root):
                rc, method_try = _clone_with_datalad(dataset_id, ds_root, url, log_path, ctx.runtime_env)
                method = method_try
                if rc != 0 or not _dataset_ready(dataset_id, ds_root):
                    used_fallback = True
                    rc2, method_try2 = _clone_with_git_annex(
                        dataset_id,
                        ds_root,
                        url,
                        log_path,
                        ctx.runtime_env,
                        jobs=max(1, min(16, ctx.cpu_workers // 2)),
                    )
                    method = method_try2
                    if rc2 != 0:
                        failures.append(f"{dataset_id}: fallback retrieval failed ({method_try2})")

            ready = _dataset_ready(dataset_id, ds_root)
            if not ready:
                failures.append(f"{dataset_id}: payload not ready after staging attempts")

            event_n = _count_event_files(ds_root) if ds_root.exists() else 0
            eeg_n = _count_eeg_payload_files(ds_root) if ds_root.exists() else 0
            broken_links = _count_eeg_broken_symlinks(ds_root) if ds_root.exists() else 0

            git_exists = (ds_root / ".git").exists()
            git_head = _git_head(ds_root) if git_exists else None
            manifest_path: Optional[Path] = None
            if git_exists and not git_head:
                failures.append(f"{dataset_id}: git repository exists but HEAD commit could not be resolved")
            if not git_exists and ds_root.exists():
                manifest_path = manifest_dir / f"{dataset_id}_sha256_manifest.json"
                _sha256_manifest(ds_root, manifest_path)

            results.append(
                {
                    "dataset_id": dataset_id,
                    "git_url": url,
                    "path": str(ds_root),
                    "git_head": git_head,
                    "sha256_manifest": str(manifest_path) if manifest_path else None,
                    "remote_head_commit": _remote_head(url),
                    "status": "PASS" if ready else "FAIL",
                    "staging_method": method,
                    "fallback_used": bool(used_fallback),
                    "n_event_files": int(event_n),
                    "n_eeg_files": int(eeg_n),
                    "n_broken_eeg_symlinks": int(broken_links),
                }
            )

        status = "PASS" if not failures else "FAIL"
        error = "" if status == "PASS" else " ; ".join(sorted(set(failures)))

        payload = {
            "timestamp_utc": _iso_now(),
            "data_root": str(ctx.data_root),
            "datasets": results,
            "status": status,
            "error": error,
        }
        _write_json(out_hashes, payload)

        if status == "FAIL":
            stop = ctx.audit_dir / "STOP_REASON.md"
            lines = [
                "# STOP_REASON",
                f"- Stage: `{stage}`",
                "- Reason: dataset staging failed for one or more required datasets.",
                "",
                "## Failures",
            ]
            for f in sorted(set(failures)):
                lines.append(f"- {f}")
            lines.extend(["", "## Log tail", "```text", _tail(log_path, 240), "```"])
            _write_text(stop, "\n".join(lines) + "\n")

        return _record_stage(
            ctx,
            stage=stage,
            status=status,
            rc=0 if status == "PASS" else 1,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="dataset staging (datalad/git-annex fail-closed)",
            outputs=[out_hashes],
            error=error,
            extra={"dataset_results": results},
        )
    except Exception as exc:
        _write_text(log_path, traceback.format_exc())
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="dataset staging",
            outputs=[out_hashes],
            error=str(exc),
        )


def _stage_core_lawc_ultradeep(ctx: RunContext) -> Dict[str, Any]:
    stage = "core_lawc_ultradeep"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    out_dir = ctx.pack_core
    out_dir.mkdir(parents=True, exist_ok=True)
    lawc_root = out_dir / "lawc_ultradeep"
    effects_root = out_dir / "effect_sizes"

    rc_lawc = 0
    rc_eff = 0
    reused = False

    if ctx.reuse_out_root:
        src_lawc = ctx.reuse_out_root / "PACK_CORE_LAWC" / "lawc_ultradeep"
        src_eff = ctx.reuse_out_root / "PACK_CORE_LAWC" / "effect_sizes"
        if src_lawc.exists() and src_eff.exists():
            _copy_tree_merge(src_lawc, lawc_root)
            _copy_tree_merge(src_eff, effects_root)
            reused = True
            _write_text(log_path, f"[{_iso_now()}] Reused core outputs from {ctx.reuse_out_root}\n")

    if not reused:
        cmd_lawc = [
            sys.executable,
            "05_audit_lawc.py",
            "--features_root",
            str(ctx.features_root_core),
            "--out_root",
            str(lawc_root),
            "--event_map",
            str(ctx.lawc_event_map),
            "--datasets",
            ",".join(CORE_STERNBERG_DATASETS),
            "--n_perm",
            str(max(50000, int(ctx.lawc_n_perm))),
            "--workers",
            str(max(1, ctx.cpu_workers)),
        ]
        rc_lawc = _run_cmd(cmd_lawc, cwd=REPO_ROOT, log_path=log_path, env=ctx.runtime_env)

        cmd_eff = [
            sys.executable,
            "scripts/effect_size_pack.py",
            "--features_root",
            str(ctx.features_root_core),
            "--datasets",
            ",".join(CORE_STERNBERG_DATASETS),
            "--out_dir",
            str(effects_root),
            "--n_boot",
            "5000",
            "--seed",
            "123",
        ]
        rc_eff = _run_cmd(cmd_eff, cwd=REPO_ROOT, log_path=log_path, env=ctx.runtime_env)

    lawc_json = lawc_root / "lawc_audit" / "locked_test_results.json"
    lawc_csv = lawc_root / "lawc_audit" / "locked_test_results.csv"
    lawc_neg = lawc_root / "lawc_audit" / "negative_controls.csv"
    effect_csv = effects_root / "effect_size_summary.csv"

    pass_lawc_gate = False
    if lawc_json.exists():
        payload = _read_json_if_exists(lawc_json) or {}
        pass_lawc_gate = bool(payload.get("pass", False))

    required = [lawc_json, lawc_csv, lawc_neg, effect_csv]
    missing = [str(p) for p in required if not p.exists()]

    status = "PASS"
    error = ""
    if rc_lawc != 0 or rc_eff != 0:
        status = "FAIL"
        error = f"lawc/effect command failed rc_lawc={rc_lawc} rc_effect={rc_eff}"
    elif missing:
        status = "FAIL"
        error = f"missing required outputs: {missing}"
    elif not pass_lawc_gate:
        status = "FAIL"
        error = "Law-C locked endpoint did not pass"

    cmd_desc = (
        f"reuse_core_from={ctx.reuse_out_root}"
        if reused
        else "run 05_audit_lawc.py + scripts/effect_size_pack.py"
    )

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=0 if status == "PASS" else 1,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command=cmd_desc,
        outputs=required,
        error=error,
    )


def _mechanism_seed_spec(ctx: RunContext) -> str:
    # Target 0-99 when enough budget remains; fallback 0-49.
    if _seconds_left(ctx) >= 5.0 * 3600.0:
        return "0-99"
    return "0-49"


def _mechanism_auditfix(out_dir: Path) -> Tuple[bool, str]:
    table_csv = out_dir / "Table_mechanism_effects.csv"
    aggregate_json = out_dir / "aggregate_mechanism.json"
    sanity_md = out_dir / "MECHANISM_SANITY.md"

    if not table_csv.exists() or not aggregate_json.exists():
        return False, "missing mechanism table or aggregate json"

    df = pd.read_csv(table_csv)
    if df.empty or "metric" not in df.columns or "p_value" not in df.columns:
        return False, "mechanism table missing required columns"

    pvals = pd.to_numeric(df["p_value"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    qvals = _bh_qvals([float(x) for x in pvals])
    if len(qvals) != len(df):
        return False, "failed to compute q-values"
    df["q_value"] = np.asarray(qvals, dtype=float)
    if not np.all(np.isfinite(pd.to_numeric(df["q_value"], errors="coerce").to_numpy(dtype=float))):
        return False, "NaN q-values after BH-FDR"
    df.to_csv(table_csv, index=False)

    agg = _read_json_if_exists(aggregate_json) or {}
    q_map = {str(m): float(q) for m, q in zip(df["metric"].astype(str), qvals)}
    agg_q = dict(agg.get("q_values", {}))
    for k, v in q_map.items():
        agg_q[k] = float(v)
    agg["q_values"] = agg_q
    _write_json(aggregate_json, agg)

    obs = dict((agg.get("observed_medians") or {}))
    ci = dict((agg.get("observed_ci95") or {}))
    p = dict((agg.get("p_values") or {}))
    a = float(obs.get("a", np.nan))
    b = float(obs.get("b", np.nan))
    ab = float(obs.get("ab", np.nan))
    ab_prod = float(a * b) if np.isfinite(a) and np.isfinite(b) else float("nan")
    sign_match = bool(np.sign(ab_prod) == np.sign(ab)) if np.isfinite(ab_prod) and np.isfinite(ab) else False
    ratio = float(ab / ab_prod) if np.isfinite(ab_prod) and abs(ab_prod) > 1e-12 and np.isfinite(ab) else float("nan")

    lines = [
        "# MECHANISM_SANITY",
        "",
        "## Definitions",
        "- `a`: effect of load on pupil proxy (units: pupil AUC per load increment).",
        "- `b`: effect of pupil proxy on P3 amplitude controlling load (units: uV per pupil AUC).",
        "- `c_prime`: direct effect of load on P3 controlling pupil (units: uV per load increment).",
        "- `ab`: indirect/mediation term reported by mechanism module.",
        "",
        "## Arithmetic Sanity",
        f"- Observed `a`: {a}",
        f"- Observed `b`: {b}",
        f"- Reported `ab`: {ab}",
        f"- Product `a*b`: {ab_prod}",
        f"- Sign match (`ab` vs `a*b`): {sign_match}",
        f"- Ratio `ab/(a*b)`: {ratio}",
        "",
        "## CI / P-Value Basis",
        "- CI values are taken from `aggregate_mechanism.json` (`observed_ci95`) and correspond to the same mechanism model output as the p-values.",
        f"- `ab` CI95: {ci.get('ab')}",
        f"- `ab` p-value: {p.get('ab')}",
    ]
    _write_text(sanity_md, "\n".join(lines) + "\n")

    neg_summary = out_dir / "mechanism_negative_controls_summary.json"
    payload = {
        "n_metrics": int(len(df)),
        "control_pupil_degrade_true": int(pd.to_numeric(df.get("control_pupil_degrade", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(bool).sum()),
        "control_load_degrade_true": int(pd.to_numeric(df.get("control_load_degrade", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(bool).sum()),
    }
    _write_json(neg_summary, payload)

    return True, ""


def _stage_mechanism_deep_auditfix(ctx: RunContext) -> Dict[str, Any]:
    stage = "mechanism_deep_auditfix"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    out_dir = ctx.pack_mechanism
    out_dir.mkdir(parents=True, exist_ok=True)

    reused = False
    if ctx.reuse_out_root:
        src = ctx.reuse_out_root / "PACK_MECHANISM"
        if src.exists():
            _copy_tree_merge(src, out_dir)
            reused = True
            _write_text(log_path, f"[{_iso_now()}] Reused mechanism outputs from {ctx.reuse_out_root}\n")

    rc = 0
    cmd = None
    if not (out_dir / "Table_mechanism_effects.csv").exists() or not (out_dir / "aggregate_mechanism.json").exists():
        seed_spec = _mechanism_seed_spec(ctx)
        cmd = [
            sys.executable,
            "scripts/mechanism_deep.py",
            "--features_root",
            str(ctx.features_root_mechanism),
            "--data_root",
            str(ctx.data_root),
            "--dataset_id",
            "ds003838",
            "--config",
            str(ctx.config),
            "--event_map",
            str(ctx.mechanism_event_map),
            "--out_dir",
            str(out_dir),
            "--seeds",
            seed_spec,
            "--parallel_procs",
            str(max(1, ctx.gpu_parallel_procs)),
            "--n_perm",
            str(max(20000, int(ctx.mechanism_n_perm))),
            "--min_trials",
            "20",
        ]
        if ctx.resume:
            cmd.append("--resume")
        rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, env=ctx.runtime_env)

    required = [
        out_dir / "Table_mechanism_effects.csv",
        out_dir / "aggregate_mechanism.json",
        out_dir / "FIG_load_vs_pupil.png",
        out_dir / "FIG_pupil_vs_p3_partial.png",
        out_dir / "FIG_mediation_ab.png",
        out_dir / "FIG_mechanism_summary.png",
        out_dir / "MECHANISM_SANITY.md",
    ]
    ok_fix, fix_err = _mechanism_auditfix(out_dir)
    missing = [str(p) for p in required if not p.exists()]
    neg_summary = out_dir / "mechanism_negative_controls_summary.json"

    status = "PASS" if rc == 0 and ok_fix and not missing else "FAIL"
    error = "" if status == "PASS" else f"mechanism auditfix failed rc={rc} ok_fix={ok_fix} fix_err={fix_err} missing={missing}"
    outputs = list(required) + ([neg_summary] if neg_summary.exists() else [])

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=0 if status == "PASS" else 1,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="reuse/calc mechanism + q-fix + sanity",
        outputs=outputs,
        error=error,
    )


def _bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray, *, n_boot: int = 2000, seed: int = 0) -> Tuple[float, List[float]]:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    m = np.isfinite(s)
    y = y[m]
    s = s[m]
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan"), [float("nan"), float("nan")]

    auc = float(roc_auc_score(y, s))
    rng = np.random.default_rng(int(seed))
    boots: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, y.size, size=y.size)
        yb = y[idx]
        sb = s[idx]
        if len(np.unique(yb)) < 2:
            continue
        boots.append(float(roc_auc_score(yb, sb)))
    if not boots:
        return auc, [float("nan"), float("nan")]
    arr = np.asarray(boots, dtype=float)
    return auc, [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]


def _perm_p_auc(y_true: np.ndarray, y_score: np.ndarray, *, n_perm: int, seed: int) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    m = np.isfinite(s)
    y = y[m]
    s = s[m]
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")

    obs = float(roc_auc_score(y, s))
    rng = np.random.default_rng(int(seed))
    null = np.full(int(n_perm), np.nan, dtype=float)
    for i in range(int(n_perm)):
        yp = y.copy()
        rng.shuffle(yp)
        if len(np.unique(yp)) < 2:
            continue
        null[i] = float(roc_auc_score(yp, s))
    finite = null[np.isfinite(null)]
    if finite.size == 0:
        return float("nan")
    return float((1.0 + np.sum(np.abs(finite - 0.5) >= abs(obs - 0.5))) / (1.0 + finite.size))


def _encode_sex(series: pd.Series) -> np.ndarray:
    vals = series.fillna("").astype(str).str.strip().str.lower()
    out = np.full(len(vals), np.nan, dtype=float)
    for i, v in enumerate(vals):
        if v in {"m", "male", "1"}:
            out[i] = 1.0
        elif v in {"f", "female", "0"}:
            out[i] = 0.0
    return out


def _robust_group_beta(df: pd.DataFrame, score_col: str, group_col: str) -> Tuple[float, int]:
    import statsmodels.api as sm

    work = df.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work[group_col] = pd.to_numeric(work[group_col], errors="coerce")
    if "age" in work.columns:
        work["age"] = pd.to_numeric(work["age"], errors="coerce")
    else:
        work["age"] = np.nan
    if "sex" in work.columns:
        work["sex_num"] = _encode_sex(work["sex"])
    else:
        work["sex_num"] = np.nan

    fit = work[[score_col, group_col, "age", "sex_num"]].dropna().copy()
    if len(fit) < 8 or fit[group_col].nunique() < 2:
        return float("nan"), int(len(fit))

    y = fit[score_col].astype(float)
    X = sm.add_constant(fit[[group_col, "age", "sex_num"]].astype(float), has_constant="add")
    model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    res = model.fit()
    return float(res.params.get(group_col, np.nan)), int(len(fit))


def _perm_p_group_beta(df: pd.DataFrame, score_col: str, group_col: str, *, n_perm: int, seed: int) -> float:
    obs, n_fit = _robust_group_beta(df, score_col, group_col)
    if not np.isfinite(obs) or n_fit <= 0:
        return float("nan")

    rng = np.random.default_rng(int(seed))
    arr = pd.to_numeric(df[group_col], errors="coerce").to_numpy(dtype=float)
    null = np.full(int(n_perm), np.nan, dtype=float)
    for i in range(int(n_perm)):
        perm = arr.copy()
        rng.shuffle(perm)
        tmp = df.copy()
        tmp[group_col] = perm
        b, _ = _robust_group_beta(tmp, score_col, group_col)
        null[i] = b

    finite = null[np.isfinite(null)]
    if finite.size == 0:
        return float("nan")
    return float((1.0 + np.sum(np.abs(finite) >= abs(obs))) / (1.0 + finite.size))


def _read_raw_any(path: Path):
    suf = path.suffix.lower()
    if suf == ".vhdr":
        return mne.io.read_raw_brainvision(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".edf":
        return mne.io.read_raw_edf(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".bdf":
        return mne.io.read_raw_bdf(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".set":
        return mne.io.read_raw_eeglab(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".fif":
        return mne.io.read_raw_fif(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".gdf":
        return mne.io.read_raw_gdf(path.as_posix(), preload=True, verbose="ERROR")
    raise RuntimeError(f"unsupported EEG suffix: {path.suffix}")


def _safe_subject(v: Any) -> str:
    s = str(v).strip()
    return s[4:] if s.startswith("sub-") else s


def _stage_clinical_pd_full(ctx: RunContext) -> Dict[str, Any]:
    stage = "clinical_PD_ds003490_FULL"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    out_dir = ctx.pack_pd
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir = out_dir / "mapping_decode"
    mapping_dir.mkdir(parents=True, exist_ok=True)

    ds_root = ctx.data_root / "ds003490"
    stop_reason = out_dir / "STOP_REASON.md"
    mapping_json = mapping_dir / "pd_mapping_decode.json"
    mapping_summary = mapping_dir / "mapping_decode_summary.json"
    run_summary = out_dir / "pd_run_summary.json"

    cmd_decode = [
        sys.executable,
        "scripts/pd_decode_mapping_ds003490.py",
        "--dataset_root",
        str(ds_root),
        "--out_dir",
        str(mapping_dir),
        "--stop_reason",
        str(stop_reason),
    ]
    rc_decode = _run_cmd(cmd_decode, cwd=REPO_ROOT, log_path=log_path, env=ctx.runtime_env, allow_fail=True)
    dec = _read_json_if_exists(mapping_summary) or {}
    dec_status = str(dec.get("status", "FAIL")).upper()
    if rc_decode != 0 or dec_status not in {"PASS", "SKIP"}:
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command=" ".join(cmd_decode),
            outputs=[mapping_summary, stop_reason],
            error=f"pd mapping decode failed rc={rc_decode} status={dec_status}",
        )
    if dec_status == "SKIP":
        for p in [out_dir / "pd_endpoints.csv", out_dir / "FIG_pd_primary_auc_roc.png", out_dir / "FIG_pd_deviation_by_group_medstate.png", out_dir / "FIG_pd_on_off_paired.png"]:
            if p.exists():
                p.unlink()
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="pd mapping decode",
            outputs=[mapping_summary, stop_reason],
            error=str(dec.get("reason", "mapping skip")),
        )

    cmd_run = [
        sys.executable,
        "scripts/pd_run_ds003490.py",
        "--dataset_root",
        str(ds_root),
        "--mapping_json",
        str(mapping_json),
        "--out_dir",
        str(out_dir),
        "--n_perm",
        str(max(20000, int(ctx.pd_n_perm))),
        "--stop_reason",
        str(stop_reason),
    ]
    rc_run = _run_cmd(cmd_run, cwd=REPO_ROOT, log_path=log_path, env=ctx.runtime_env, allow_fail=True)
    run = _read_json_if_exists(run_summary) or {}
    run_status = str(run.get("status", "FAIL")).upper()

    required = [
        out_dir / "pd_deviation_scores.csv",
        out_dir / "pd_endpoints.csv",
        out_dir / "FIG_pd_primary_auc_roc.png",
        out_dir / "FIG_pd_deviation_by_group_medstate.png",
        out_dir / "FIG_pd_on_off_paired.png",
    ]
    if rc_run != 0 or run_status == "FAIL":
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command=" ".join(cmd_run),
            outputs=[run_summary, stop_reason],
            error=f"pd run failed rc={rc_run} status={run_status}",
        )
    if run_status == "SKIP":
        for p in required:
            if p.exists():
                p.unlink()
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command=" ".join(cmd_run),
            outputs=[mapping_summary, run_summary, stop_reason],
            error=str(run.get("reason", "pd stage skip")),
        )

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command=" ".join(cmd_run),
            outputs=[mapping_summary, run_summary],
            error=f"missing required PD outputs: {missing}",
        )

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="pd decode + run full clinical endpoints",
        outputs=[mapping_summary, run_summary, *required],
        error="",
    )


def _stage_clinical_dementia(ctx: RunContext) -> Dict[str, Any]:
    stage = "clinical_Dementia_ds004504_REFRESH"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    out_dir = ctx.pack_dementia
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_root = ctx.data_root / "ds004504"
    stop_reason = out_dir / "STOP_REASON_ds004504.md"

    participants_tsv = ds_root / "participants.tsv"
    participants_json = ds_root / "participants.json"

    if not ds_root.exists() or not participants_tsv.exists():
        reason = f"dataset or participants missing: {ds_root}"
        _write_text(stop_reason, f"# STOP_REASON ds004504\n\n{reason}\n")
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="dementia spectral",
            outputs=[stop_reason],
            error=reason,
        )

    part = pd.read_csv(participants_tsv, sep="\t")
    part["subject_id"] = part["participant_id"].map(_safe_subject)

    group_label_map = {"A": "AD", "F": "FTD", "C": "CN"}
    if participants_json.exists():
        try:
            pjson = json.loads(participants_json.read_text(encoding="utf-8"))
            lv = (((pjson or {}).get("Group") or {}).get("Levels") or {}) if isinstance(pjson, dict) else {}
            if isinstance(lv, dict) and lv:
                tmp = {}
                for k, v in lv.items():
                    vv = str(v).lower()
                    if "alzheimer" in vv:
                        tmp[str(k)] = "AD"
                    elif "frontotemporal" in vv or "ftd" in vv:
                        tmp[str(k)] = "FTD"
                    elif "healthy" in vv or "control" in vv:
                        tmp[str(k)] = "CN"
                if tmp:
                    group_label_map.update(tmp)
        except Exception:
            pass

    part["group"] = part["Group"].astype(str).map(group_label_map).fillna("UNK")
    part["age"] = pd.to_numeric(part.get("Age"), errors="coerce")
    part["sex"] = part.get("Gender", "").astype(str)
    part["mmse"] = pd.to_numeric(part.get("MMSE"), errors="coerce")

    eeg_files = []
    for p in ds_root.rglob("*"):
        if not p.is_file():
            continue
        if "derivatives" in p.parts:
            continue
        if _EEG_PAT.search(p.name):
            eeg_files.append(p)
    eeg_files = sorted(set(eeg_files))

    if not eeg_files:
        reason = "no resting EEG payload files found"
        _write_text(stop_reason, f"# STOP_REASON ds004504\n\n{reason}\n")
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="dementia spectral",
            outputs=[stop_reason],
            error=reason,
        )

    # One representative run per subject.
    subj_to_file: Dict[str, Path] = {}
    for p in eeg_files:
        m = re.search(r"sub-([A-Za-z0-9]+)", str(p))
        if not m:
            continue
        sid = m.group(1)
        if sid not in subj_to_file:
            subj_to_file[sid] = p

    spectral_rows: List[Dict[str, Any]] = []
    failures: List[str] = []

    for sid, fp in sorted(subj_to_file.items()):
        try:
            raw = _read_raw_any(fp)
            raw.pick_types(eeg=True, eog=False, misc=False, stim=False)
            if len(raw.ch_names) == 0:
                failures.append(f"{sid}: no EEG channels")
                continue
            sf = float(raw.info["sfreq"])
            if sf > 256:
                raw.resample(256, verbose="ERROR")
                sf = float(raw.info["sfreq"])

            data = raw.get_data()
            if data.size == 0:
                failures.append(f"{sid}: empty EEG data")
                continue

            psd, freqs = mne.time_frequency.psd_array_welch(
                data,
                sfreq=sf,
                fmin=1.0,
                fmax=40.0,
                n_fft=1024,
                n_overlap=512,
                average="mean",
                verbose="ERROR",
            )
            pxx = np.nanmean(psd, axis=0)
            f = np.asarray(freqs, dtype=float)

            def _band(a: float, b: float) -> float:
                m = (f >= a) & (f < b)
                if int(m.sum()) < 2:
                    return float("nan")
                return float(np.trapz(pxx[m], f[m]))

            theta = _band(4.0, 8.0)
            alpha = _band(8.0, 12.0)
            total = _band(1.0, 30.0)

            # Spectral slope proxy: linear fit in log10-log10 domain (2-30 Hz).
            msl = (f >= 2.0) & (f <= 30.0) & np.isfinite(pxx) & (pxx > 0)
            if int(msl.sum()) >= 6:
                x = np.log10(f[msl])
                y = np.log10(pxx[msl])
                slope = float(np.polyfit(x, y, deg=1)[0])
            else:
                slope = float("nan")

            spectral_rows.append(
                {
                    "subject_id": sid,
                    "rest_file": str(fp),
                    "theta_alpha_ratio": float(theta / max(alpha, 1e-12)) if np.isfinite(theta) and np.isfinite(alpha) else float("nan"),
                    "rel_alpha": float(alpha / max(total, 1e-12)) if np.isfinite(alpha) and np.isfinite(total) else float("nan"),
                    "spectral_slope": float(slope),
                }
            )
        except Exception as exc:
            failures.append(f"{sid}: {exc}")
            continue

    if not spectral_rows:
        reason = "no usable EEG runs for spectral features"
        _write_text(
            stop_reason,
            "\n".join(
                [
                    "# STOP_REASON ds004504",
                    "",
                    reason,
                    "",
                    "## Sample failures",
                    *[f"- {x}" for x in failures[:20]],
                ]
            )
            + "\n",
        )
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="dementia spectral",
            outputs=[stop_reason],
            error=reason,
        )

    feat_df = pd.DataFrame(spectral_rows)
    feat_df = feat_df.merge(part[["subject_id", "group", "age", "sex", "mmse"]], on="subject_id", how="left")
    feat_df.to_csv(out_dir / "spectral_subject_features.csv", index=False)

    # Control-trained normative deviations per feature.
    import statsmodels.api as sm

    feature_cols = ["theta_alpha_ratio", "rel_alpha", "spectral_slope"]
    dev_df = feat_df.copy()

    for col in feature_cols:
        ctrl = dev_df[(dev_df["group"] == "CN") & np.isfinite(pd.to_numeric(dev_df[col], errors="coerce"))].copy()
        if ctrl.empty:
            dev_df[f"pred_{col}"] = np.nan
            dev_df[f"dev_z_{col}"] = np.nan
            continue

        ctrl["age"] = pd.to_numeric(ctrl["age"], errors="coerce")
        ctrl["sex_num"] = _encode_sex(ctrl["sex"])

        dev_df["age"] = pd.to_numeric(dev_df["age"], errors="coerce")
        dev_df["sex_num"] = _encode_sex(dev_df["sex"])

        fit = ctrl[[col, "age", "sex_num"]].dropna().copy()
        use_cov = len(fit) >= 10

        if use_cov:
            y = fit[col].astype(float)
            X = sm.add_constant(fit[["age", "sex_num"]].astype(float), has_constant="add")
            res = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
            X_eval = sm.add_constant(dev_df[["age", "sex_num"]].astype(float), has_constant="add")
            pred = np.asarray(res.predict(X_eval), dtype=float)

            X_ctrl = sm.add_constant(ctrl[["age", "sex_num"]].astype(float), has_constant="add")
            pred_ctrl = np.asarray(res.predict(X_ctrl), dtype=float)
            resid = pd.to_numeric(ctrl[col], errors="coerce").to_numpy(dtype=float) - pred_ctrl
        else:
            mu = float(pd.to_numeric(ctrl[col], errors="coerce").mean())
            pred = np.full(len(dev_df), mu, dtype=float)
            resid = pd.to_numeric(ctrl[col], errors="coerce").to_numpy(dtype=float) - mu

        resid = resid[np.isfinite(resid)]
        if resid.size == 0:
            sigma = 1.0
        else:
            mad = float(np.median(np.abs(resid - np.median(resid))))
            sigma = float(max(1e-6, 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.std(resid)))
            if not np.isfinite(sigma) or sigma <= 1e-6:
                sigma = 1.0

        dev_df[f"pred_{col}"] = pred
        dev_df[f"dev_z_{col}"] = (pd.to_numeric(dev_df[col], errors="coerce").to_numpy(dtype=float) - pred) / sigma

    # Oriented composite deviation.
    oriented_cols = []
    for col in feature_cols:
        zc = f"dev_z_{col}"
        if zc not in dev_df.columns:
            continue
        ad_med = np.nanmedian(pd.to_numeric(dev_df.loc[dev_df["group"] == "AD", col], errors="coerce").to_numpy(dtype=float))
        cn_med = np.nanmedian(pd.to_numeric(dev_df.loc[dev_df["group"] == "CN", col], errors="coerce").to_numpy(dtype=float))
        sign = 1.0 if (np.isfinite(ad_med) and np.isfinite(cn_med) and ad_med >= cn_med) else -1.0
        oc = f"oriented_{zc}"
        dev_df[oc] = sign * pd.to_numeric(dev_df[zc], errors="coerce")
        oriented_cols.append(oc)

    if oriented_cols:
        dev_df["composite_deviation"] = dev_df[oriented_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    else:
        dev_df["composite_deviation"] = np.nan

    dev_df.to_csv(out_dir / "normative_deviation_scores.csv", index=False)

    # Endpoints + permutations + BH-FDR.
    endpoint_rows: List[Dict[str, Any]] = []

    endpoint_features = [f"dev_z_{c}" for c in feature_cols] + ["composite_deviation"]

    for feat in endpoint_features:
        vals = pd.to_numeric(dev_df[feat], errors="coerce") if feat in dev_df.columns else pd.Series([], dtype=float)
        if vals.empty:
            continue

        def _auc_row(label: str, pos: str, neg: str, seed_off: int) -> Optional[Dict[str, Any]]:
            sub = dev_df[dev_df["group"].isin([pos, neg])].copy()
            sub["x"] = pd.to_numeric(sub[feat], errors="coerce")
            sub = sub[np.isfinite(sub["x"])].copy()
            if sub.empty:
                return None
            y = (sub["group"].astype(str) == pos).astype(int).to_numpy(dtype=int)
            if len(np.unique(y)) < 2:
                return None
            s = sub["x"].to_numpy(dtype=float)
            auc, ci = _bootstrap_auc(y, s, n_boot=2000, seed=200 + seed_off)
            p = _perm_p_auc(y, s, n_perm=max(20000, int(ctx.dementia_n_perm)), seed=300 + seed_off)
            auc_flipped = float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")
            return {
                "endpoint": label,
                "feature": feat,
                "type": "auc",
                "n": int(len(sub)),
                "estimate": float(auc),
                "auc_raw": float(auc),
                "auc_flipped": float(auc_flipped),
                "ci95_lo": float(ci[0]),
                "ci95_hi": float(ci[1]),
                "perm_p": float(p),
            }

        row = _auc_row("AUC_AD_vs_CN", "AD", "CN", 1)
        if row:
            endpoint_rows.append(row)
        row = _auc_row("AUC_FTD_vs_CN", "FTD", "CN", 2)
        if row:
            endpoint_rows.append(row)
        row = _auc_row("AUC_AD_vs_FTD", "AD", "FTD", 3)
        if row:
            endpoint_rows.append(row)

        # Robust regression MMSE ~ deviation + age + sex + group dummies.
        tmp = dev_df.copy()
        tmp["x"] = pd.to_numeric(tmp[feat], errors="coerce")
        tmp["mmse"] = pd.to_numeric(tmp["mmse"], errors="coerce")
        tmp["age"] = pd.to_numeric(tmp["age"], errors="coerce")
        tmp["sex_num"] = _encode_sex(tmp["sex"])
        dummies = pd.get_dummies(tmp["group"], prefix="grp", drop_first=True)
        fit = pd.concat([tmp[["mmse", "x", "age", "sex_num"]], dummies], axis=1).dropna().copy()

        if len(fit) >= 16:
            y = fit["mmse"].astype(float)
            X = fit.drop(columns=["mmse"]).astype(float)
            X = sm.add_constant(X, has_constant="add")
            try:
                res = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
                beta = float(res.params.get("x", np.nan))
            except Exception:
                beta = float("nan")

            # Permutation on MMSE labels.
            rng = np.random.default_rng(411)
            null = np.full(max(20000, int(ctx.dementia_n_perm)), np.nan, dtype=float)
            if np.isfinite(beta):
                arr_mmse = y.to_numpy(dtype=float)
                for i in range(len(null)):
                    yp = arr_mmse.copy()
                    rng.shuffle(yp)
                    try:
                        rnull = sm.RLM(yp, X, M=sm.robust.norms.HuberT()).fit()
                        null[i] = float(rnull.params.get("x", np.nan))
                    except Exception:
                        continue
                finite = null[np.isfinite(null)]
                p_mmse = float((1.0 + np.sum(np.abs(finite) >= abs(beta))) / (1.0 + len(finite))) if finite.size else float("nan")
            else:
                p_mmse = float("nan")
                finite = np.asarray([], dtype=float)

            endpoint_rows.append(
                {
                    "endpoint": "MMSE_regression_beta",
                    "feature": feat,
                    "type": "robust_beta",
                    "n": int(len(fit)),
                    "estimate": float(beta),
                    "auc_raw": float("nan"),
                    "auc_flipped": float("nan"),
                    "ci95_lo": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
                    "ci95_hi": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
                    "perm_p": float(p_mmse),
                }
            )

    end_df = pd.DataFrame(endpoint_rows)
    if not end_df.empty:
        pvals = pd.to_numeric(end_df["perm_p"], errors="coerce").fillna(1.0).to_numpy(dtype=float).tolist()
        end_df["perm_q"] = _bh_qvals([float(x) for x in pvals])
        if not np.all(np.isfinite(pd.to_numeric(end_df["perm_q"], errors="coerce").to_numpy(dtype=float))):
            return _record_stage(
                ctx,
                stage=stage,
                status="FAIL",
                rc=1,
                started=started,
                log_path=log_path,
                summary_path=summary_path,
                command="dementia resting spectral + normative + endpoints",
                outputs=[out_dir / "dementia_endpoints.csv"],
                error="NaN q-values in dementia endpoints",
            )
    else:
        end_df["perm_q"] = []
    end_df.to_csv(out_dir / "dementia_endpoints.csv", index=False)

    # Figures
    fig1 = out_dir / "FIG_dementia_composite_by_group.png"
    fig2 = out_dir / "FIG_dementia_auc_composite.png"
    fig3 = out_dir / "FIG_dementia_mmse_vs_composite.png"

    try:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        order = ["CN", "FTD", "AD"]
        for i, g in enumerate(order, start=1):
            vals = pd.to_numeric(dev_df.loc[dev_df["group"] == g, "composite_deviation"], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            x = np.full(vals.shape, i, dtype=float)
            jitter = np.linspace(-0.15, 0.15, num=len(vals), dtype=float) if len(vals) > 1 else np.asarray([0.0])
            ax.scatter(x + jitter, vals, alpha=0.55, s=14)
            ax.hlines(float(np.median(vals)), i - 0.2, i + 0.2, color="#b22222", linewidth=2.0)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax.set_xticks([1, 2, 3], order)
        ax.set_ylabel("Composite deviation")
        ax.set_title("Resting EEG deviation by group")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(fig1, dpi=160)
        plt.close(fig)
    except Exception:
        pass

    try:
        comp = end_df[(end_df["feature"] == "composite_deviation") & (end_df["type"] == "auc")].copy()
        if not comp.empty:
            fig, ax = plt.subplots(figsize=(7.0, 4.8))
            labs = comp["endpoint"].astype(str).tolist()
            if "auc_flipped" in comp.columns:
                vals = pd.to_numeric(comp["auc_flipped"], errors="coerce").to_numpy(dtype=float)
            else:
                vals = pd.to_numeric(comp["estimate"], errors="coerce").to_numpy(dtype=float)
            ax.bar(np.arange(len(vals)), vals, color=["#224b8f", "#2a9d8f", "#e76f51"][: len(vals)])
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks(np.arange(len(vals)), labs, rotation=20, ha="right")
            ax.set_ylabel("AUC (flipped)")
            ax.set_title("Composite deviation group-separation AUC")
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            fig.savefig(fig2, dpi=160)
            plt.close(fig)
    except Exception:
        pass

    try:
        mm = dev_df[np.isfinite(pd.to_numeric(dev_df["mmse"], errors="coerce")) & np.isfinite(pd.to_numeric(dev_df["composite_deviation"], errors="coerce"))].copy()
        if not mm.empty:
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            cmap = {"CN": "#1b9e77", "FTD": "#d95f02", "AD": "#7570b3", "UNK": "#666666"}
            for g, gg in mm.groupby("group"):
                ax.scatter(
                    pd.to_numeric(gg["composite_deviation"], errors="coerce"),
                    pd.to_numeric(gg["mmse"], errors="coerce"),
                    s=16,
                    alpha=0.6,
                    color=cmap.get(str(g), "#666666"),
                    label=str(g),
                )
            ax.set_xlabel("Composite deviation")
            ax.set_ylabel("MMSE")
            ax.set_title("MMSE vs EEG deviation")
            ax.grid(alpha=0.2)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(fig3, dpi=160)
            plt.close(fig)
    except Exception:
        pass

    incl = {
        "n_subjects_total_features": int(dev_df["subject_id"].nunique()),
        "n_group_CN": int((dev_df["group"] == "CN").sum()),
        "n_group_AD": int((dev_df["group"] == "AD").sum()),
        "n_group_FTD": int((dev_df["group"] == "FTD").sum()),
        "n_mmse_nonmissing": int(np.isfinite(pd.to_numeric(dev_df["mmse"], errors="coerce")).sum()),
        "n_failures_read": int(len(failures)),
    }
    _write_json(out_dir / "inclusion_exclusion_summary.json", incl)

    status = "PASS" if not end_df.empty else "SKIP"
    error = "" if status == "PASS" else "no dementia endpoints computed"

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=0 if status != "FAIL" else 1,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="dementia resting spectral + normative + endpoints",
        outputs=[
            out_dir / "spectral_subject_features.csv",
            out_dir / "normative_deviation_scores.csv",
            out_dir / "dementia_endpoints.csv",
            out_dir / "inclusion_exclusion_summary.json",
            fig1,
            fig2,
            fig3,
        ],
        error=error,
    )


def _status_table(ctx: RunContext) -> str:
    lines = [
        "| Stage | Status | Return code | Runtime (s) |",
        "|---|---|---:|---:|",
    ]
    for st in STAGE_ORDER:
        rec = next((r for r in ctx.stage_records if r.get("stage") == st), None)
        if rec is None:
            lines.append(f"| {st} | NOT_RUN | - | - |")
        else:
            lines.append(f"| {st} | {rec['status']} | {rec['returncode']} | {rec['elapsed_sec']:.1f} |")
    return "\n".join(lines)


def _build_final_report(ctx: RunContext, run_status: str, run_error: str) -> Path:
    report = ctx.audit_dir / "NN_FINAL_PUSH_V2_REPORT.md"

    stage_rows = [
        f"| {r['stage']} | {r['status']} | {r['returncode']} | {r['elapsed_sec']:.1f} | {Path(r['log']).name} | {Path(r['summary']).name} |"
        for r in ctx.stage_records
    ]

    ds_hashes = _read_json_if_exists(ctx.audit_dir / "dataset_hashes.json") or {}
    ds_rows: List[str] = []
    for d in ds_hashes.get("datasets", []):
        commit = d.get("git_head", None)
        commit_txt = "null" if commit is None else str(commit)
        ds_rows.append(
            f"| {d.get('dataset_id')} | {commit_txt} | {d.get('n_event_files')} | {d.get('n_eeg_files')} | {d.get('status')} |"
        )
    if not ds_rows:
        ds_rows = ["| <none> | <none> | <none> | <none> | <none> |"]

    # Core Law-C
    lawc_rows = ["| <none> | <none> | <none> | <none> | <none> | <none> |"]
    lawc_json = ctx.pack_core / "lawc_ultradeep" / "lawc_audit" / "locked_test_results.json"
    if lawc_json.exists():
        try:
            payload = json.loads(lawc_json.read_text(encoding="utf-8"))
            rows = payload.get("datasets", [])
            if rows:
                lawc_rows = []
                for r in rows:
                    lawc_rows.append(
                        f"| {r.get('dataset_id')} | {r.get('median_rho')} | {r.get('p_value')} | {r.get('q_value')} | {r.get('x_control_degrade_pass')} | {r.get('y_control_degrade_pass')} |"
                    )
        except Exception:
            pass

    effect_rows = ["| <none> | <none> | <none> | <none> |"]
    effect_csv = ctx.pack_core / "effect_sizes" / "effect_size_summary.csv"
    if effect_csv.exists():
        try:
            ef = pd.read_csv(effect_csv)
            if not ef.empty:
                effect_rows = []
                for _, r in ef.iterrows():
                    effect_rows.append(
                        f"| {r.get('dataset_id')} | {r.get('n_subjects')} | {r.get('slope_median_uv_per_load')} [{r.get('slope_ci95_lo')}, {r.get('slope_ci95_hi')}] | {r.get('delta_median_uv')} [{r.get('delta_ci95_lo')}, {r.get('delta_ci95_hi')}] |"
                    )
        except Exception:
            pass

    mech_rows = ["| <none> | <none> | <none> | <none> | <none> | <none> |"]
    mech_csv = ctx.pack_mechanism / "Table_mechanism_effects.csv"
    if mech_csv.exists():
        try:
            df = pd.read_csv(mech_csv)
            if not df.empty:
                mech_rows = []
                for _, r in df.iterrows():
                    mech_rows.append(
                        f"| {r.get('metric')} | {r.get('observed_median')} | {r.get('p_value')} | {r.get('q_value')} | {r.get('control_pupil_degrade')} | {r.get('control_load_degrade')} |"
                    )
        except Exception:
            pass

    pd_rows = ["| <none> | <none> | <none> | <none> | <none> | <none> |"]
    pd_end = ctx.pack_pd / "pd_endpoints.csv"
    if pd_end.exists():
        try:
            d = pd.read_csv(pd_end)
            if not d.empty:
                pd_rows = []
                for _, r in d.iterrows():
                    pd_rows.append(
                        f"| {r.get('endpoint')} | {r.get('type')} | {r.get('n')} | {r.get('estimate')} | {r.get('perm_p')} | {r.get('perm_q')} |"
                    )
        except Exception:
            pass
    pd_stop = ctx.pack_pd / "STOP_REASON.md"

    dem_rows = ["| <none> | <none> | <none> | <none> | <none> | <none> | <none> |"]
    dem_end = ctx.pack_dementia / "dementia_endpoints.csv"
    if dem_end.exists():
        try:
            d = pd.read_csv(dem_end)
            if not d.empty:
                dem_rows = []
                for _, r in d.iterrows():
                    dem_rows.append(
                        f"| {r.get('endpoint')} | {r.get('feature')} | {r.get('n')} | {r.get('estimate')} | {r.get('auc_flipped')} | {r.get('perm_p')} | {r.get('perm_q')} |"
                    )
        except Exception:
            pass

    pd_incl = _read_json_if_exists(ctx.pack_pd / "inclusion_exclusion_summary.json") or {}
    dem_incl = _read_json_if_exists(ctx.pack_dementia / "inclusion_exclusion_summary.json") or {}

    negative_lines = []
    lawc_neg = ctx.pack_core / "lawc_ultradeep" / "lawc_audit" / "negative_controls.csv"
    if lawc_neg.exists():
        negative_lines.append(f"- Core Law-C negatives: `{lawc_neg}`")
    mech_neg = ctx.pack_mechanism / "mechanism_negative_controls_summary.json"
    if mech_neg.exists():
        negative_lines.append(f"- Mechanism negatives: `{mech_neg}`")
    if not negative_lines:
        negative_lines.append("- <none>")

    mech_sanity = ctx.pack_mechanism / "MECHANISM_SANITY.md"

    figure_candidates = [
        ctx.pack_core / "effect_sizes" / "FIG_slopes_uv_per_load.png",
        ctx.pack_core / "effect_sizes" / "FIG_delta_uv_high_vs_low.png",
        ctx.pack_core / "effect_sizes" / "FIG_waveforms_by_load.png",
        ctx.pack_mechanism / "FIG_load_vs_pupil.png",
        ctx.pack_mechanism / "FIG_pupil_vs_p3_partial.png",
        ctx.pack_mechanism / "FIG_mediation_ab.png",
        ctx.pack_mechanism / "FIG_mechanism_summary.png",
        ctx.pack_pd / "FIG_pd_deviation_by_group_medstate.png",
        ctx.pack_pd / "FIG_pd_on_off_paired.png",
        ctx.pack_pd / "FIG_pd_primary_auc_roc.png",
        ctx.pack_dementia / "FIG_dementia_composite_by_group.png",
        ctx.pack_dementia / "FIG_dementia_auc_composite.png",
        ctx.pack_dementia / "FIG_dementia_mmse_vs_composite.png",
    ]
    figure_lines = [f"- `{p}`" for p in figure_candidates if p.exists()]
    if not figure_lines:
        figure_lines = ["- <none>"]

    lines = [
        "# NN_FINAL_PUSH_V2 REPORT",
        "",
        f"- Output root: `{ctx.out_root}`",
        f"- Run status: `{run_status}`",
        f"- Resume: `{ctx.resume}`",
        f"- Start: `{datetime.fromtimestamp(ctx.start_ts, tz=timezone.utc).isoformat()}`",
        f"- End: `{datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat()}`",
    ]

    if run_error:
        lines.extend(["", "## Run error", "```text", run_error, "```"])

    lines.extend(
        [
            "",
            "## Stage status",
            "| Stage | Status | Return code | Runtime (s) | Log | Summary |",
            "|---|---|---:|---:|---|---|",
            *stage_rows,
            "",
            "## Dataset hashes/commits",
            "| Dataset | Commit | Event files | EEG files | Status |",
            "|---|---|---:|---:|---|",
            *ds_rows,
            "",
            "## Inclusion/Exclusion",
            f"- PD: `{json.dumps(pd_incl, sort_keys=True)}`",
            f"- Dementia: `{json.dumps(dem_incl, sort_keys=True)}`",
            "",
            "## Negative controls summary",
            *negative_lines,
            "",
            "## Core Law-C locked results",
            "| Dataset | Median rho | p | q | X-control degrade | Y-control degrade |",
            "|---|---:|---:|---:|---|---|",
            *lawc_rows,
            "",
            "## Core effect sizes",
            "| Dataset | N subjects | Slope median [CI95] | Delta median [CI95] |",
            "|---|---:|---|---|",
            *effect_rows,
            "",
            "## Mechanism results",
            "| Metric | Observed median | p | q | Control pupil degrade | Control load degrade |",
            "|---|---:|---:|---:|---|---|",
            *mech_rows,
            "",
            "## Clinical PD endpoints",
            "| Endpoint | Type | N | Estimate | Perm p | Perm q |",
            "|---|---|---:|---:|---:|---:|",
            *pd_rows,
            f"- PD STOP_REASON: `{pd_stop}`" if pd_stop.exists() else "- PD STOP_REASON: <none>",
            "",
            "## Clinical Dementia endpoints",
            "| Endpoint | Feature | N | AUC raw / Beta | AUC flipped | Perm p | Perm q |",
            "|---|---|---:|---:|---:|---:|---:|",
            *dem_rows,
            "",
            "## Mechanism sanity",
            f"- `{mech_sanity}`" if mech_sanity.exists() else "- <none>",
            "",
            "## Figure paths",
            *figure_lines,
            "",
            "## Provenance",
            f"- Repo fingerprint: `{ctx.audit_dir / 'repo_fingerprint.json'}`",
            f"- Dataset hashes: `{ctx.audit_dir / 'dataset_hashes.json'}`",
            f"- GPU monitor: `{ctx.audit_dir / 'nvidia_smi_1hz.csv'}`",
            f"- Config default: `{ctx.config}`",
            f"- Law-C event map: `{ctx.lawc_event_map}`",
            f"- Mechanism event map: `{ctx.mechanism_event_map}`",
            f"- Final bundle: `{ctx.outzip_dir / 'NN_SUBMISSION_PACKET_V2.zip'}`",
        ]
    )

    _write_text(report, "\n".join(lines) + "\n")
    return report


def _stage_final_report(ctx: RunContext, run_status: str, run_error: str) -> Dict[str, Any]:
    stage = "final_report"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    try:
        report = _build_final_report(ctx, run_status, run_error)
        return _record_stage(
            ctx,
            stage=stage,
            status="PASS",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="build final report",
            outputs=[report],
        )
    except Exception as exc:
        _write_text(log_path, traceback.format_exc())
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="build final report",
            error=str(exc),
        )


def _stage_zip_bundle(ctx: RunContext) -> Dict[str, Any]:
    stage = "zip_bundle"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    zpath = ctx.outzip_dir / "NN_SUBMISSION_PACKET_V2.zip"
    include = [ctx.audit_dir, ctx.pack_core, ctx.pack_mechanism, ctx.pack_pd, ctx.pack_dementia]

    added: List[str] = []
    error = ""
    status = "PASS"
    rc = 0

    # Snapshot configs used into AUDIT.
    configs_dir = ctx.audit_dir / "configs_used"
    configs_dir.mkdir(parents=True, exist_ok=True)
    for src in [ctx.config, ctx.lawc_event_map, ctx.mechanism_event_map]:
        if src.exists():
            shutil.copy2(src, configs_dir / src.name)

    try:
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root in include:
                if not root.exists():
                    continue
                for p in sorted(root.rglob("*")):
                    if p.is_dir():
                        continue
                    rel = p.relative_to(ctx.out_root)
                    zf.write(p, rel.as_posix())
                    added.append(rel.as_posix())
    except Exception as exc:
        status = "FAIL"
        rc = 1
        error = str(exc)

    _write_json(summary_path, {"status": status, "zip": str(zpath), "n_files": len(added), "error": error})

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=rc,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="zip submission packet",
        outputs=[zpath] if zpath.exists() else [],
        error=error,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, default=Path("/filesystemHcog/openneuro"))
    ap.add_argument("--out_root", type=Path, default=None)

    ap.add_argument(
        "--features_root_core",
        type=Path,
        default=Path("/filesystemHcog/features_cache_FIX2_20260222_061927"),
    )
    ap.add_argument(
        "--features_root_mechanism",
        type=Path,
        default=Path("/filesystemHcog/features_cache_FIX1_20260222_060109"),
    )

    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--lawc_event_map", type=Path, default=Path("configs/lawc_event_map.yaml"))
    ap.add_argument("--mechanism_event_map", type=Path, default=Path("configs/mechanism_event_map.yaml"))
    ap.add_argument(
        "--reuse_out_root",
        type=Path,
        default=Path("/filesystemHcog/runs/20260222_231912_NN_FINAL_PUSH"),
        help="Existing NN_FINAL_PUSH output root to reuse PASS core/mechanism artefacts from.",
    )

    ap.add_argument("--wall_hours", type=float, default=10.0)
    ap.add_argument("--lawc_n_perm", type=int, default=50000)
    ap.add_argument("--mechanism_n_perm", type=int, default=20000)
    ap.add_argument("--pd_n_perm", type=int, default=20000)
    ap.add_argument("--dementia_n_perm", type=int, default=20000)

    ap.add_argument("--gpu_parallel_procs", type=int, default=6)
    ap.add_argument("--cpu_workers", type=int, default=32)

    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def _print_stage_table(ctx: RunContext) -> None:
    print("PASS_FAIL_SKIP_TABLE_BEGIN", flush=True)
    print(_status_table(ctx), flush=True)
    print("PASS_FAIL_SKIP_TABLE_END", flush=True)


_STAGE_NAME_BY_FN: Dict[str, str] = {
    "_stage_preflight": "preflight",
    "_stage_compile_gate": "compile_gate",
    "_stage_stage_datasets": "stage_datasets",
    "_stage_core_lawc_ultradeep": "core_lawc_ultradeep",
    "_stage_mechanism_deep_auditfix": "mechanism_deep_auditfix",
    "_stage_clinical_pd_full": "clinical_PD_ds003490_FULL",
    "_stage_clinical_dementia": "clinical_Dementia_ds004504_REFRESH",
    "_stage_final_report": "final_report",
    "_stage_zip_bundle": "zip_bundle",
}


def _stage_name_for_fn(fn: Any) -> str:
    name = getattr(fn, "__name__", "")
    return _STAGE_NAME_BY_FN.get(name, name.replace("_stage_", ""))


def _load_resume_record(ctx: RunContext, stage: str) -> Optional[Dict[str, Any]]:
    status_path = ctx.audit_dir / f"{stage}.status"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"
    if not status_path.exists() or not summary_path.exists():
        return None

    status = status_path.read_text(encoding="utf-8", errors="ignore").strip().upper()
    if status not in {"PASS", "SKIP"}:
        return None

    rec = _read_json_if_exists(summary_path) or {}
    rec["stage"] = stage
    rec["status"] = status
    rec.setdefault("returncode", 0 if status in {"PASS", "SKIP"} else 1)
    rec.setdefault("started_at", _iso_now())
    rec.setdefault("ended_at", _iso_now())
    rec.setdefault("elapsed_sec", 0.0)
    rec.setdefault("log", str(ctx.audit_dir / f"{stage}.log"))
    rec.setdefault("summary", str(summary_path))
    rec.setdefault("command", "resume-skip")
    rec.setdefault("outputs", [])
    rec.setdefault("error", "")
    return rec


def main() -> int:
    args = parse_args()

    out_root = args.out_root
    if out_root is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_root = Path("/filesystemHcog/runs") / f"{ts}_NN_FINAL_PUSH_V2"

    if out_root.exists() and not args.resume:
        audit = out_root / "AUDIT"
        has_prior_run = (audit / "run_status.json").exists() or bool(list(audit.glob("*_summary.json")))
        if has_prior_run:
            print(f"ERROR: out_root exists and --resume not set: {out_root}", file=sys.stderr, flush=True)
            return 1

    runtime_env = os.environ.copy()
    runtime_env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'src'}:{runtime_env.get('PYTHONPATH', '')}" if runtime_env.get("PYTHONPATH") else str(REPO_ROOT / "src")
    )

    ctx = RunContext(
        out_root=out_root,
        audit_dir=out_root / "AUDIT",
        outzip_dir=out_root / "OUTZIP",
        pack_core=out_root / "PACK_CORE_LAWC",
        pack_mechanism=out_root / "PACK_MECHANISM",
        pack_pd=out_root / "PACK_CLINICAL_PD",
        pack_dementia=out_root / "PACK_CLINICAL_DEMENTIA",
        data_root=args.data_root,
        features_root_core=args.features_root_core,
        features_root_mechanism=args.features_root_mechanism,
        config=args.config,
        lawc_event_map=args.lawc_event_map,
        mechanism_event_map=args.mechanism_event_map,
        reuse_out_root=(args.reuse_out_root if args.reuse_out_root else None),
        wall_hours=float(args.wall_hours),
        lawc_n_perm=int(args.lawc_n_perm),
        mechanism_n_perm=int(args.mechanism_n_perm),
        pd_n_perm=int(args.pd_n_perm),
        dementia_n_perm=int(args.dementia_n_perm),
        gpu_parallel_procs=int(max(1, args.gpu_parallel_procs)),
        cpu_workers=int(max(1, args.cpu_workers)),
        resume=bool(args.resume),
        start_ts=time.time(),
        deadline_ts=time.time() + float(args.wall_hours) * 3600.0,
        stage_records=[],
        stage_status={},
        monitor_proc=None,
        monitor_handle=None,
        runtime_env=runtime_env,
    )

    ctx.out_root.mkdir(parents=True, exist_ok=True)
    ctx.audit_dir.mkdir(parents=True, exist_ok=True)
    ctx.outzip_dir.mkdir(parents=True, exist_ok=True)

    run_status = "PASS"
    run_error = ""

    core_stages = [
        _stage_preflight,
        _stage_compile_gate,
        _stage_stage_datasets,
        _stage_core_lawc_ultradeep,
        _stage_mechanism_deep_auditfix,
        _stage_clinical_pd_full,
        _stage_clinical_dementia,
    ]

    try:
        for fn in core_stages:
            stage_name = _stage_name_for_fn(fn)

            if ctx.resume:
                resumed = _load_resume_record(ctx, stage_name)
                if resumed is not None:
                    ctx.stage_records.append(resumed)
                    ctx.stage_status[stage_name] = resumed["status"]
                    continue

            # Reserve time for report + zip.
            if fn.__name__ in {"_stage_clinical_pd_full", "_stage_clinical_dementia"} and not _wall_guard(ctx, reserve_sec=1800):
                st = stage_name
                now = time.time()
                lg = ctx.audit_dir / f"{st}.log"
                sm = ctx.audit_dir / f"{st}_summary.json"
                _write_text(lg, f"[{_iso_now()}] SKIP wall-clock budget guard\n")
                _record_stage(
                    ctx,
                    stage=st,
                    status="SKIP",
                    rc=0,
                    started=now,
                    log_path=lg,
                    summary_path=sm,
                    command="budget_guard",
                    error="budget guard",
                )
                continue

            rec = fn(ctx)
            if rec["status"] == "FAIL":
                run_status = "FAIL"
                run_error = rec.get("error", "stage failed")
                break

    except Exception as exc:
        run_status = "FAIL"
        run_error = f"{type(exc).__name__}: {exc}\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    # Build final report regardless of prior status.
    try:
        rec_report = _stage_final_report(ctx, run_status, run_error)
        if rec_report["status"] == "FAIL" and run_status == "PASS":
            run_status = "FAIL"
            run_error = rec_report.get("error", "final report failed")
    except Exception as exc:
        run_status = "FAIL"
        run_error = f"final_report exception: {exc}"

    try:
        rec_zip = _stage_zip_bundle(ctx)
        if rec_zip["status"] == "FAIL" and run_status == "PASS":
            run_status = "FAIL"
            run_error = rec_zip.get("error", "zip failed")
    except Exception as exc:
        run_status = "FAIL"
        run_error = f"zip exception: {exc}"

    _stop_gpu_monitor(ctx)

    report_path = ctx.audit_dir / "NN_FINAL_PUSH_V2_REPORT.md"
    run_status_payload = {
        "status": run_status,
        "error": run_error,
        "out_root": str(ctx.out_root),
        "report": str(report_path),
        "stages": [
            {
                "stage": r["stage"],
                "status": r["status"],
                "returncode": r["returncode"],
                "elapsed_sec": r["elapsed_sec"],
                "log": r["log"],
                "summary": r["summary"],
            }
            for r in ctx.stage_records
        ],
    }
    _write_json(ctx.audit_dir / "run_status.json", run_status_payload)

    print(f"OUT_ROOT={ctx.out_root}", flush=True)
    print(f"REPORT={report_path}", flush=True)
    _print_stage_table(ctx)
    print(f"BUNDLE={ctx.outzip_dir / 'NN_SUBMISSION_PACKET_V2.zip'}", flush=True)

    return 0 if run_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
