#!/usr/bin/env python3
"""NN_FINAL_PUSH end-to-end orchestrator.

Stages
0) preflight
1) compile_gate
2) stage_datasets
3) core_lawc_ultradeep
4) mechanism_deep
5) clinical_PD_ds003490
6) clinical_Dementia_ds004504
7) final_report
8) zip_bundle

Fail-closed rules:
- Hard FAIL on compile_gate and staging failures.
- BIDS event semantics are explicit for oddball mapping; ambiguous mapping -> SKIP with STOP_REASON.
- No silent endpoint substitution for locked ERP windows.
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
    "ds005095": "https://github.com/OpenNeuroDatasets/ds005095.git",
    "ds003655": "https://github.com/OpenNeuroDatasets/ds003655.git",
    "ds004117": "https://github.com/OpenNeuroDatasets/ds004117.git",
    "ds003838": "https://github.com/OpenNeuroDatasets/ds003838.git",
    "ds004796": "https://github.com/OpenNeuroDatasets/ds004796.git",
    "ds003490": "https://github.com/OpenNeuroDatasets/ds003490.git",
    "ds004504": "https://github.com/OpenNeuroDatasets/ds004504.git",
}

REQUIRED_DATASETS: List[str] = [
    "ds005095",
    "ds003655",
    "ds004117",
    "ds003838",
    "ds004796",
    "ds003490",
    "ds004504",
]

CORE_STERNBERG_DATASETS: List[str] = ["ds005095", "ds003655", "ds004117"]

STAGE_ORDER: List[str] = [
    "preflight",
    "compile_gate",
    "stage_datasets",
    "core_lawc_ultradeep",
    "mechanism_deep",
    "clinical_PD_ds003490",
    "clinical_Dementia_ds004504",
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


def _git_head(path: Path) -> str:
    try:
        p = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if p.returncode == 0:
            return p.stdout.strip()
    except Exception:
        pass
    return "<unavailable>"


def _remote_head(url: str) -> str:
    try:
        p = subprocess.run(["git", "ls-remote", url, "HEAD"], capture_output=True, text=True, check=False)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip().split()[0]
    except Exception:
        pass
    return "<unavailable>"


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

        for dataset_id in REQUIRED_DATASETS:
            url = DATASET_URLS[dataset_id]
            ds_root = ctx.data_root / dataset_id
            method = "existing"
            used_fallback = False

            if not _dataset_ready(dataset_id, ds_root):
                # If incomplete payload is present, keep the repo but attempt retrieval repair.
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

            local_commit = _git_head(ds_root) if ds_root.exists() else "<unavailable>"
            remote_commit = _remote_head(url)
            event_n = _count_event_files(ds_root) if ds_root.exists() else 0
            eeg_n = _count_eeg_payload_files(ds_root) if ds_root.exists() else 0
            broken_links = _count_eeg_broken_symlinks(ds_root) if ds_root.exists() else 0
            sternberg_logs = _count_sternberg_logs(ds_root) if ds_root.exists() else 0
            pupil_n = int(len(list(ds_root.rglob("*_pupil.tsv"))) + len(list(ds_root.rglob("*_pupil.tsv.gz")))) if ds_root.exists() else 0

            results.append(
                {
                    "dataset_id": dataset_id,
                    "git_url": url,
                    "path": str(ds_root),
                    "checked_out_commit": local_commit,
                    "remote_head_commit": remote_commit,
                    "status": "PASS" if ready else "FAIL",
                    "staging_method": method,
                    "fallback_used": bool(used_fallback),
                    "n_event_files": int(event_n),
                    "n_eeg_files": int(eeg_n),
                    "n_broken_eeg_symlinks": int(broken_links),
                    "n_sternberg_logs": int(sternberg_logs),
                    "n_pupil_files": int(pupil_n),
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
            command="dataset staging (datalad/git-annex with fail-closed fallback)",
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

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=0 if status == "PASS" else 1,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command=f"{' '.join(cmd_lawc)} && {' '.join(cmd_eff)}",
        outputs=required,
        error=error,
    )


def _mechanism_seed_spec(ctx: RunContext) -> str:
    # Target 0-99 when enough budget remains; fallback 0-49.
    if _seconds_left(ctx) >= 5.0 * 3600.0:
        return "0-99"
    return "0-49"


def _stage_mechanism_deep(ctx: RunContext) -> Dict[str, Any]:
    stage = "mechanism_deep"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    out_dir = ctx.pack_mechanism
    out_dir.mkdir(parents=True, exist_ok=True)

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
    ]
    missing = [str(p) for p in required if not p.exists()]

    status = "PASS" if rc == 0 and not missing else "FAIL"
    error = "" if status == "PASS" else f"mechanism stage failed rc={rc} missing={missing}"

    # Emit explicit negative-control summary for report readability.
    neg_summary = out_dir / "mechanism_negative_controls_summary.json"
    if (out_dir / "Table_mechanism_effects.csv").exists():
        try:
            df = pd.read_csv(out_dir / "Table_mechanism_effects.csv")
            payload = {
                "n_metrics": int(len(df)),
                "control_pupil_degrade_true": int(pd.to_numeric(df.get("control_pupil_degrade", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(bool).sum()),
                "control_load_degrade_true": int(pd.to_numeric(df.get("control_load_degrade", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(bool).sum()),
                "seed_spec_used": seed_spec,
                "n_perm": int(max(20000, int(ctx.mechanism_n_perm))),
            }
            _write_json(neg_summary, payload)
        except Exception:
            pass

    outputs = list(required)
    if neg_summary.exists():
        outputs.append(neg_summary)

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=0 if status == "PASS" else 1,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command=" ".join(cmd),
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


def _stage_clinical_pd(ctx: RunContext) -> Dict[str, Any]:
    stage = "clinical_PD_ds003490"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    out_dir = ctx.pack_pd
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_root = ctx.data_root / "ds003490"
    stop_reason = out_dir / "STOP_REASON_ds003490.md"

    if not ds_root.exists():
        reason = f"dataset root missing: {ds_root}"
        _write_text(stop_reason, f"# STOP_REASON ds003490\n\n{reason}\n")
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="pd clinical",
            outputs=[stop_reason],
            error=reason,
        )

    participants_tsv = ds_root / "participants.tsv"
    if not participants_tsv.exists():
        reason = f"participants.tsv missing: {participants_tsv}"
        _write_text(stop_reason, f"# STOP_REASON ds003490\n\n{reason}\n")
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="pd clinical",
            outputs=[stop_reason],
            error=reason,
        )

    # 1) Decode fail-closed mapping.
    event_files = sorted(list(ds_root.rglob("*_events.tsv")) + list(ds_root.rglob("*_events.tsv.gz")))
    if not event_files:
        reason = "no *_events.tsv files found"
        _write_text(stop_reason, f"# STOP_REASON ds003490\n\n{reason}\n")
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="pd mapping decode",
            outputs=[stop_reason],
            error=reason,
        )

    mapping_decode_dir = out_dir / "mapping_decode"
    mapping_decode_dir.mkdir(parents=True, exist_ok=True)

    val_counts: Dict[str, int] = {}
    missing_semantics: List[str] = []
    for fp in event_files[: min(80, len(event_files))]:
        try:
            df = pd.read_csv(fp, sep="\t")
        except Exception:
            continue
        cols = {str(c) for c in df.columns}
        if "onset" not in cols or "duration" not in cols:
            missing_semantics.append(str(fp))
            continue
        if "trial_type" not in cols:
            continue
        vc = df["trial_type"].dropna().astype(str).value_counts().to_dict()
        for k, v in vc.items():
            val_counts[k] = val_counts.get(k, 0) + int(v)

    if missing_semantics:
        reason = "BIDS events semantics ambiguous: missing onset/duration in one or more events files"
        _write_text(
            stop_reason,
            "\n".join(
                [
                    "# STOP_REASON ds003490",
                    "",
                    reason,
                    "",
                    "## Files",
                    *[f"- {x}" for x in missing_semantics[:20]],
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
            command="pd mapping decode",
            outputs=[stop_reason],
            error=reason,
        )

    # Explicit target/standard/novel decode.
    target_vals = sorted([k for k in val_counts if re.search(r"\btarget\b", k, flags=re.IGNORECASE)])
    standard_vals = sorted([k for k in val_counts if re.search(r"\bstandard\b", k, flags=re.IGNORECASE)])
    novel_vals = sorted([k for k in val_counts if re.search(r"\bnovel\b", k, flags=re.IGNORECASE)])

    if not target_vals or not standard_vals:
        reason = (
            "Could not decode required oddball classes from events trial_type; "
            f"target={target_vals} standard={standard_vals}"
        )
        _write_text(
            stop_reason,
            "\n".join(
                [
                    "# STOP_REASON ds003490",
                    "",
                    reason,
                    "",
                    "## trial_type values observed",
                    "```json",
                    json.dumps(val_counts, indent=2),
                    "```",
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
            command="pd mapping decode",
            outputs=[stop_reason],
            error=reason,
        )

    # Build explicit auditable event map for extraction.
    oddball_values = target_vals + standard_vals + novel_vals
    load_value_map: Dict[str, float] = {}
    for v in standard_vals:
        load_value_map[v] = 0.0
    for v in target_vals:
        load_value_map[v] = 1.0
    for v in novel_vals:
        load_value_map[v] = 2.0

    event_filter = f"trial_type.isin({json.dumps(oddball_values)})"
    pd_event_map = {
        "defaults": _read_json_if_exists(ctx.lawc_event_map) or {},
        "datasets": {
            "ds003490": {
                "event_filter": event_filter,
                "load_column": "trial_type",
                "load_value_map": load_value_map,
                "load_sign": 1.0,
                "rt_column": "response_time",
            }
        },
    }

    # Normalize defaults if source file is YAML.
    try:
        base_yaml = yaml.safe_load(ctx.lawc_event_map.read_text(encoding="utf-8"))
        if isinstance(base_yaml, dict):
            pd_event_map["defaults"] = base_yaml.get("defaults", {})
    except Exception:
        pass

    pd_event_map_path = mapping_decode_dir / "pd_event_map.yaml"
    pd_event_map_path.write_text(yaml.safe_dump(pd_event_map, sort_keys=False), encoding="utf-8")

    cand_rows = []
    for cls_name, values in [("target", target_vals), ("standard", standard_vals), ("novel", novel_vals)]:
        for v in values:
            cand_rows.append({"class": cls_name, "trial_type": v, "count": int(val_counts.get(v, 0))})
    cand_csv = mapping_decode_dir / "CANDIDATE_TABLE.csv"
    pd.DataFrame(cand_rows).to_csv(cand_csv, index=False)

    decode_summary = {
        "dataset_id": "ds003490",
        "status": "PASS",
        "reason": "",
        "n_event_files": int(len(event_files)),
        "target_values": target_vals,
        "standard_values": standard_vals,
        "novel_values": novel_vals,
        "event_filter": event_filter,
        "load_value_map": load_value_map,
        "candidate_table": str(cand_csv),
        "event_map": str(pd_event_map_path),
    }
    _write_json(mapping_decode_dir / "mapping_decode_summary.json", decode_summary)

    # 2) Preprocess and primary extraction (locked window 0.35-0.60).
    runtime_cfg = yaml.safe_load(ctx.config.read_text(encoding="utf-8"))
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}
    runtime_cfg.setdefault("events", {})
    runtime_cfg["events"].setdefault("selector", {})
    runtime_cfg["events"]["selector"]["column"] = "trial_type"
    runtime_cfg["events"]["selector"]["include"] = [
        "Target Tone",
        "Standard Tone",
        "Novel Tone",
        "target",
        "standard",
        "novel",
    ]

    cfg_primary = out_dir / "config_pd_primary.yaml"
    cfg_primary.write_text(yaml.safe_dump(runtime_cfg, sort_keys=False), encoding="utf-8")

    deriv_root = out_dir / "derivatives" / "ds003490"
    features_root_primary = out_dir / "features_primary"

    cmd_pre = [
        sys.executable,
        "01_preprocess_CPU.py",
        "--bids_root",
        str(ds_root),
        "--deriv_root",
        str(deriv_root),
        "--config",
        str(cfg_primary),
        "--workers",
        str(max(1, min(10, ctx.cpu_workers // 2))),
        "--mne_n_jobs",
        "1",
    ]
    rc_pre = _run_cmd(cmd_pre, cwd=REPO_ROOT, log_path=log_path, env=ctx.runtime_env)
    if rc_pre != 0:
        reason = f"preprocess failed rc={rc_pre}"
        _write_text(stop_reason, f"# STOP_REASON ds003490\n\n{reason}\n")
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command=" ".join(cmd_pre),
            outputs=[stop_reason],
            error=reason,
        )

    cmd_ext = [
        sys.executable,
        "02_extract_features_CPU.py",
        "--bids_root",
        str(ds_root),
        "--deriv_root",
        str(deriv_root),
        "--features_root",
        str(features_root_primary),
        "--config",
        str(cfg_primary),
        "--cohort",
        "clinical",
        "--dataset_id",
        "ds003490",
        "--lawc_event_map",
        str(pd_event_map_path),
        "--workers",
        str(max(1, min(10, ctx.cpu_workers // 2))),
    ]
    rc_ext = _run_cmd(cmd_ext, cwd=REPO_ROOT, log_path=log_path, env=ctx.runtime_env)
    if rc_ext != 0:
        reason = f"feature extraction failed rc={rc_ext}"
        _write_text(stop_reason, f"# STOP_REASON ds003490\n\n{reason}\n")
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command=" ".join(cmd_ext),
            outputs=[stop_reason],
            error=reason,
        )

    # Optional exploratory sensitivity window [0.30, 0.60].
    features_root_sens = out_dir / "features_sensitivity_030_060"
    ran_sensitivity = False
    if _wall_guard(ctx, reserve_sec=1800):
        cfg_sens_dict = yaml.safe_load(cfg_primary.read_text(encoding="utf-8"))
        if not isinstance(cfg_sens_dict, dict):
            cfg_sens_dict = runtime_cfg
        cfg_sens_dict.setdefault("eeg", {})
        cfg_sens_dict["eeg"].setdefault("p3b", {})
        cfg_sens_dict["eeg"]["p3b"]["amp_window_s"] = [0.30, 0.60]
        cfg_sens = out_dir / "config_pd_sensitivity_030_060.yaml"
        cfg_sens.write_text(yaml.safe_dump(cfg_sens_dict, sort_keys=False), encoding="utf-8")

        cmd_ext_sens = [
            sys.executable,
            "02_extract_features_CPU.py",
            "--bids_root",
            str(ds_root),
            "--deriv_root",
            str(deriv_root),
            "--features_root",
            str(features_root_sens),
            "--config",
            str(cfg_sens),
            "--cohort",
            "clinical",
            "--dataset_id",
            "ds003490",
            "--lawc_event_map",
            str(pd_event_map_path),
            "--workers",
            str(max(1, min(10, ctx.cpu_workers // 2))),
        ]
        rc_sens = _run_cmd(cmd_ext_sens, cwd=REPO_ROOT, log_path=log_path, env=ctx.runtime_env, allow_fail=True)
        ran_sensitivity = rc_sens == 0

    # 3) Aggregate subject/session features from H5.
    part = pd.read_csv(participants_tsv, sep="\t")
    part = part.rename(columns={c: str(c) for c in part.columns})
    part["subject_id"] = part["participant_id"].map(_safe_subject)

    group_map = {}
    med_map: Dict[Tuple[str, str], str] = {}
    age_map: Dict[str, float] = {}
    sex_map: Dict[str, str] = {}
    for _, r in part.iterrows():
        sid = str(r.get("subject_id", "")).strip()
        if not sid:
            continue
        grp = str(r.get("Group", "")).strip().upper()
        if grp == "CTL":
            group_map[sid] = "control"
        elif grp == "PD":
            group_map[sid] = "pd"
        else:
            group_map[sid] = "unknown"

        s1 = str(r.get("sess1_Med", "")).strip().upper()
        s2 = str(r.get("sess2_Med", "")).strip().upper()
        if s1 in {"ON", "OFF"}:
            med_map[(sid, "01")] = s1
        if s2 in {"ON", "OFF"}:
            med_map[(sid, "02")] = s2

        age_map[sid] = float(pd.to_numeric(pd.Series([r.get("age")]), errors="coerce").iloc[0])
        sex_map[sid] = str(r.get("sex", ""))

    def _extract_trial_rows(features_root: Path, window_label: str) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        ds_dir = features_root / "ds003490"
        for fp in sorted(ds_dir.rglob("*.h5")) if ds_dir.exists() else []:
            try:
                with h5py.File(fp, "r") as h:
                    amp = np.asarray(h["p3b_amp"], dtype=float) * 1e6
                    lat = np.asarray(h["p3b_lat"], dtype=float) if "p3b_lat" in h else np.full(len(amp), np.nan)
                    load = np.asarray(h["memory_load"], dtype=float)
                    rt = np.asarray(h["rt"], dtype=float) if "rt" in h else np.full(len(amp), np.nan)
                    acc = np.asarray(h["accuracy"], dtype=float) if "accuracy" in h else np.full(len(amp), np.nan)
                    ch = np.asarray(h["p3b_channel"]).astype(str) if "p3b_channel" in h else np.asarray(["unknown"] * len(amp))

                    sid = str(h.attrs.get("bids_subject", "")).strip()
                    ses = str(h.attrs.get("bids_session", "")).strip() or "01"
                    if ses.lower().startswith("ses-"):
                        ses = ses.split("-", 1)[1]
                    if sid.lower().startswith("sub-"):
                        sid = sid.split("-", 1)[1]
            except Exception:
                continue

            for i in range(len(amp)):
                lv = float(load[i]) if i < len(load) else float("nan")
                cls = "unknown"
                if np.isfinite(lv):
                    if int(round(lv)) == 1:
                        cls = "target"
                    elif int(round(lv)) == 0:
                        cls = "standard"
                    elif int(round(lv)) == 2:
                        cls = "novel"
                rows.append(
                    {
                        "window": window_label,
                        "subject_id": sid,
                        "session": ses,
                        "class_label": cls,
                        "memory_load": lv,
                        "p3_amp_uV": float(amp[i]),
                        "p3_lat_s": float(lat[i]) if i < len(lat) else float("nan"),
                        "rt_s": float(rt[i]) if i < len(rt) else float("nan"),
                        "accuracy": float(acc[i]) if i < len(acc) else float("nan"),
                        "p3_channel": str(ch[i]) if i < len(ch) else "unknown",
                    }
                )
        return pd.DataFrame(rows)

    trial_primary = _extract_trial_rows(features_root_primary, "primary_0.35_0.60")
    if trial_primary.empty:
        reason = "no extracted trial rows for ds003490"
        _write_text(stop_reason, f"# STOP_REASON ds003490\n\n{reason}\n")
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="pd extraction",
            outputs=[stop_reason],
            error=reason,
        )

    trial_sens = _extract_trial_rows(features_root_sens, "exploratory_0.30_0.60") if ran_sensitivity else pd.DataFrame()

    trial_all = pd.concat([x for x in [trial_primary, trial_sens] if not x.empty], axis=0, ignore_index=True)
    trial_all.to_csv(out_dir / "trial_level_features.csv", index=False)

    subj_sess = (
        trial_primary[trial_primary["class_label"] == "target"]
        .groupby(["subject_id", "session"], as_index=False)
        .agg(
            n_target_trials=("p3_amp_uV", "size"),
            p3_amp_uV=("p3_amp_uV", "mean"),
            p3_lat_s=("p3_lat_s", "mean"),
            rt_s=("rt_s", "mean"),
            accuracy=("accuracy", "mean"),
            fallback_non_pz_trials=("p3_channel", lambda s: int((pd.Series(s).astype(str).str.upper() != "PZ").sum())),
        )
    )

    subj_sess["group"] = subj_sess["subject_id"].map(group_map).fillna("unknown")
    subj_sess["group_bin"] = subj_sess["group"].map({"control": 0.0, "pd": 1.0})
    subj_sess["med_state"] = [med_map.get((sid, ses), "") for sid, ses in zip(subj_sess["subject_id"], subj_sess["session"])]
    subj_sess["age"] = subj_sess["subject_id"].map(age_map)
    subj_sess["sex"] = subj_sess["subject_id"].map(sex_map)

    subj_sess.to_csv(out_dir / "subject_session_primary_features.csv", index=False)

    # Exploratory sensitivity summary (optional).
    if not trial_sens.empty:
        subj_sess_sens = (
            trial_sens[trial_sens["class_label"] == "target"]
            .groupby(["subject_id", "session"], as_index=False)
            .agg(
                n_target_trials=("p3_amp_uV", "size"),
                p3_amp_uV=("p3_amp_uV", "mean"),
            )
        )
        subj_sess_sens["group"] = subj_sess_sens["subject_id"].map(group_map).fillna("unknown")
        subj_sess_sens.to_csv(out_dir / "subject_session_exploratory_030_060.csv", index=False)

    # 4) Normative model trained on controls only.
    controls = subj_sess[(subj_sess["group"] == "control") & np.isfinite(pd.to_numeric(subj_sess["p3_amp_uV"], errors="coerce"))].copy()
    eval_df = subj_sess[np.isfinite(pd.to_numeric(subj_sess["p3_amp_uV"], errors="coerce"))].copy()

    if controls.empty:
        reason = "no control target sessions available for normative training"
        _write_text(stop_reason, f"# STOP_REASON ds003490\n\n{reason}\n")
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="pd normative",
            outputs=[stop_reason],
            error=reason,
        )

    import statsmodels.api as sm

    controls = controls.copy()
    controls["age"] = pd.to_numeric(controls["age"], errors="coerce")
    controls["sex_num"] = _encode_sex(controls["sex"])

    eval_df = eval_df.copy()
    eval_df["age"] = pd.to_numeric(eval_df["age"], errors="coerce")
    eval_df["sex_num"] = _encode_sex(eval_df["sex"])

    fit = controls[["p3_amp_uV", "age", "sex_num"]].dropna().copy()
    use_covariates = len(fit) >= 10

    if use_covariates:
        y = fit["p3_amp_uV"].astype(float)
        X = sm.add_constant(fit[["age", "sex_num"]].astype(float), has_constant="add")
        mdl = sm.RLM(y, X, M=sm.robust.norms.HuberT())
        res = mdl.fit()

        X_eval = sm.add_constant(eval_df[["age", "sex_num"]].astype(float), has_constant="add")
        pred = res.predict(X_eval)

        X_ctrl = sm.add_constant(controls[["age", "sex_num"]].astype(float), has_constant="add")
        pred_ctrl = res.predict(X_ctrl)
        resid_ctrl = pd.to_numeric(controls["p3_amp_uV"], errors="coerce").to_numpy(dtype=float) - np.asarray(pred_ctrl, dtype=float)
    else:
        mu = float(pd.to_numeric(controls["p3_amp_uV"], errors="coerce").mean())
        pred = np.full(len(eval_df), mu, dtype=float)
        resid_ctrl = pd.to_numeric(controls["p3_amp_uV"], errors="coerce").to_numpy(dtype=float) - mu

    resid_ctrl = resid_ctrl[np.isfinite(resid_ctrl)]
    if resid_ctrl.size == 0:
        sigma = 1.0
    else:
        mad = float(np.median(np.abs(resid_ctrl - np.median(resid_ctrl))))
        sigma = float(max(1e-6, 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.std(resid_ctrl)))
        if not np.isfinite(sigma) or sigma <= 1e-6:
            sigma = 1.0

    eval_df["pred_p3_amp_uV"] = np.asarray(pred, dtype=float)
    eval_df["deviation_z"] = (pd.to_numeric(eval_df["p3_amp_uV"], errors="coerce").to_numpy(dtype=float) - eval_df["pred_p3_amp_uV"].to_numpy(dtype=float)) / sigma

    eval_df.to_csv(out_dir / "normative_deviation_scores.csv", index=False)

    # 5) Endpoints
    endpoint_rows: List[Dict[str, Any]] = []

    # Primary AUC PD vs controls: one row per subject (prefer OFF if available).
    per_subject_rows: List[Dict[str, Any]] = []
    for sid, g in eval_df.groupby("subject_id"):
        gg = g.copy()
        group = str(gg["group"].iloc[0])
        if group == "pd":
            off = gg[gg["med_state"].astype(str).str.upper() == "OFF"]
            on = gg[gg["med_state"].astype(str).str.upper() == "ON"]
            pick = off.iloc[0] if len(off) else (on.iloc[0] if len(on) else gg.iloc[0])
            med_pick = str(pick.get("med_state", ""))
        else:
            pick = gg.iloc[0]
            med_pick = "CONTROL"
        per_subject_rows.append(
            {
                "subject_id": sid,
                "group": group,
                "group_bin": 1 if group == "pd" else 0,
                "med_pick": med_pick,
                "deviation_z": float(pick.get("deviation_z", np.nan)),
                "age": float(pd.to_numeric(pd.Series([pick.get("age")]), errors="coerce").iloc[0]),
                "sex": str(pick.get("sex", "")),
            }
        )

    per_subject_df = pd.DataFrame(per_subject_rows)
    per_subject_df.to_csv(out_dir / "subject_primary_endpoint_rows.csv", index=False)

    auc_df = per_subject_df[(per_subject_df["group"].isin(["pd", "control"])) & np.isfinite(pd.to_numeric(per_subject_df["deviation_z"], errors="coerce"))].copy()
    if not auc_df.empty and auc_df["group_bin"].nunique() == 2:
        y = auc_df["group_bin"].to_numpy(dtype=int)
        s = auc_df["deviation_z"].to_numpy(dtype=float)
        auc, ci = _bootstrap_auc(y, s, n_boot=3000, seed=41)
        p_auc = _perm_p_auc(y, s, n_perm=max(5000, int(ctx.pd_n_perm)), seed=52)
        endpoint_rows.append(
            {
                "endpoint": "AUC_PD_vs_Control_primary_off_preferred",
                "type": "auc",
                "n": int(len(auc_df)),
                "estimate": float(auc),
                "ci95_lo": float(ci[0]),
                "ci95_hi": float(ci[1]),
                "perm_p": float(p_auc),
            }
        )

    beta, n_fit = _robust_group_beta(per_subject_df, "deviation_z", "group_bin")
    p_beta = _perm_p_group_beta(per_subject_df, "deviation_z", "group_bin", n_perm=max(5000, int(ctx.pd_n_perm)), seed=61)
    endpoint_rows.append(
        {
            "endpoint": "GroupEffect_PD_vs_Control_deviation",
            "type": "robust_beta",
            "n": int(n_fit),
            "estimate": float(beta),
            "ci95_lo": float("nan"),
            "ci95_hi": float("nan"),
            "perm_p": float(p_beta),
        }
    )

    pairs = []
    for sid, g in eval_df[eval_df["group"] == "pd"].groupby("subject_id"):
        gg = g.copy()
        off = gg[gg["med_state"].astype(str).str.upper() == "OFF"]
        on = gg[gg["med_state"].astype(str).str.upper() == "ON"]
        if len(off) and len(on):
            pairs.append({"subject_id": sid, "off": float(off.iloc[0]["deviation_z"]), "on": float(on.iloc[0]["deviation_z"])})

    if pairs:
        pair_df = pd.DataFrame(pairs)
        diff = pair_df["on"].to_numpy(dtype=float) - pair_df["off"].to_numpy(dtype=float)
        obs = float(np.mean(diff))
        rng = np.random.default_rng(71)
        null = np.full(max(5000, int(ctx.pd_n_perm)), np.nan, dtype=float)
        for i in range(len(null)):
            signs = rng.choice([-1.0, 1.0], size=len(diff))
            null[i] = float(np.mean(diff * signs))
        p_pair = float((1.0 + np.sum(np.abs(null) >= abs(obs))) / (1.0 + len(null)))
        endpoint_rows.append(
            {
                "endpoint": "Paired_PD_ON_minus_OFF_deviation",
                "type": "paired_mean_diff",
                "n": int(len(pair_df)),
                "estimate": float(obs),
                "ci95_lo": float(np.quantile(null, 0.025)),
                "ci95_hi": float(np.quantile(null, 0.975)),
                "perm_p": float(p_pair),
            }
        )
        pair_df.to_csv(out_dir / "paired_on_off_rows.csv", index=False)

    end_df = pd.DataFrame(endpoint_rows)
    if not end_df.empty:
        pvals = pd.to_numeric(end_df["perm_p"], errors="coerce").fillna(1.0).to_numpy(dtype=float).tolist()
        end_df["perm_q"] = bh_fdr([float(x) for x in pvals])
    else:
        end_df["perm_q"] = []
    end_df.to_csv(out_dir / "pd_endpoints.csv", index=False)

    # 6) Figures
    fig1 = out_dir / "FIG_pd_deviation_by_group_medstate.png"
    fig2 = out_dir / "FIG_pd_on_off_paired.png"
    fig3 = out_dir / "FIG_pd_primary_auc_roc.png"

    try:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        plot_rows = []
        for _, r in eval_df.iterrows():
            g = str(r.get("group", "unknown"))
            m = str(r.get("med_state", "")).upper()
            if g == "control":
                lab = "Control"
            elif g == "pd" and m == "OFF":
                lab = "PD OFF"
            elif g == "pd" and m == "ON":
                lab = "PD ON"
            else:
                lab = "PD/UNK"
            plot_rows.append({"label": lab, "z": float(r.get("deviation_z", np.nan))})
        pdf = pd.DataFrame(plot_rows)
        order = ["Control", "PD OFF", "PD ON", "PD/UNK"]
        for i, lab in enumerate(order, start=1):
            vals = pd.to_numeric(pdf.loc[pdf["label"] == lab, "z"], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            x = np.full(vals.shape, i, dtype=float)
            jitter = np.linspace(-0.15, 0.15, num=len(vals), dtype=float) if len(vals) > 1 else np.asarray([0.0])
            ax.scatter(x + jitter, vals, alpha=0.55, s=14)
            ax.hlines(float(np.median(vals)), i - 0.2, i + 0.2, color="#b22222", linewidth=2.0)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax.set_xticks(range(1, len(order) + 1), order)
        ax.set_ylabel("Deviation z-score")
        ax.set_title("PD oddball target deviation (primary locked window)")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(fig1, dpi=160)
        plt.close(fig)
    except Exception:
        pass

    if (out_dir / "paired_on_off_rows.csv").exists():
        try:
            pair_df = pd.read_csv(out_dir / "paired_on_off_rows.csv")
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            for _, r in pair_df.iterrows():
                ax.plot([0, 1], [float(r["off"]), float(r["on"])], color="#2a607f", alpha=0.45, linewidth=1.1)
            ax.set_xticks([0, 1], ["OFF", "ON"])
            ax.set_ylabel("Deviation z-score")
            ax.set_title("Within-subject PD ON/OFF deviation")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(fig2, dpi=160)
            plt.close(fig)
        except Exception:
            pass

    try:
        from sklearn.metrics import roc_curve, auc as sk_auc

        if not auc_df.empty and auc_df["group_bin"].nunique() == 2:
            y = auc_df["group_bin"].to_numpy(dtype=int)
            s = auc_df["deviation_z"].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y, s)
            a = float(sk_auc(fpr, tpr))
            fig, ax = plt.subplots(figsize=(5.2, 5.2))
            ax.plot(fpr, tpr, color="#224b8f", linewidth=2.0, label=f"AUC={a:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("PD vs Control ROC (primary)")
            ax.legend(frameon=False)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(fig3, dpi=160)
            plt.close(fig)
    except Exception:
        pass

    include_exclude = {
        "n_event_files": int(len(event_files)),
        "n_trial_rows_primary": int(len(trial_primary)),
        "n_subject_session_target_rows": int(len(subj_sess)),
        "n_controls_target_rows": int((subj_sess["group"] == "control").sum()),
        "n_pd_target_rows": int((subj_sess["group"] == "pd").sum()),
        "n_paired_pd_on_off": int(len(pairs)),
        "fallback_non_pz_trials_total": int(pd.to_numeric(subj_sess["fallback_non_pz_trials"], errors="coerce").fillna(0).sum()),
        "ran_sensitivity_window_030_060": bool(ran_sensitivity),
    }
    _write_json(out_dir / "inclusion_exclusion_summary.json", include_exclude)

    status = "PASS" if not end_df.empty else "SKIP"
    error = "" if status == "PASS" else "no PD endpoints computed"

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=0 if status != "FAIL" else 1,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="pd oddball decode + extract + normative + endpoints",
        outputs=[
            mapping_decode_dir / "mapping_decode_summary.json",
            cand_csv,
            pd_event_map_path,
            out_dir / "trial_level_features.csv",
            out_dir / "subject_session_primary_features.csv",
            out_dir / "normative_deviation_scores.csv",
            out_dir / "pd_endpoints.csv",
            out_dir / "inclusion_exclusion_summary.json",
            fig1,
            fig2,
            fig3,
        ],
        error=error,
    )


def _stage_clinical_dementia(ctx: RunContext) -> Dict[str, Any]:
    stage = "clinical_Dementia_ds004504"
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
            p = _perm_p_auc(y, s, n_perm=max(3000, int(ctx.dementia_n_perm)), seed=300 + seed_off)
            return {
                "endpoint": label,
                "feature": feat,
                "type": "auc",
                "n": int(len(sub)),
                "estimate": float(auc),
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
            null = np.full(max(3000, int(ctx.dementia_n_perm)), np.nan, dtype=float)
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
                    "ci95_lo": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
                    "ci95_hi": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
                    "perm_p": float(p_mmse),
                }
            )

    end_df = pd.DataFrame(endpoint_rows)
    if not end_df.empty:
        pvals = pd.to_numeric(end_df["perm_p"], errors="coerce").fillna(1.0).to_numpy(dtype=float).tolist()
        end_df["perm_q"] = bh_fdr([float(x) for x in pvals])
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
            vals = pd.to_numeric(comp["estimate"], errors="coerce").to_numpy(dtype=float)
            ax.bar(np.arange(len(vals)), vals, color=["#224b8f", "#2a9d8f", "#e76f51"][: len(vals)])
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks(np.arange(len(vals)), labs, rotation=20, ha="right")
            ax.set_ylabel("AUC")
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
    report = ctx.audit_dir / "NN_FINAL_PUSH_REPORT.md"

    stage_rows = [
        f"| {r['stage']} | {r['status']} | {r['returncode']} | {r['elapsed_sec']:.1f} | {Path(r['log']).name} | {Path(r['summary']).name} |"
        for r in ctx.stage_records
    ]

    ds_hashes = _read_json_if_exists(ctx.audit_dir / "dataset_hashes.json") or {}
    ds_rows: List[str] = []
    for d in ds_hashes.get("datasets", []):
        ds_rows.append(
            f"| {d.get('dataset_id')} | {d.get('checked_out_commit')} | {d.get('n_event_files')} | {d.get('n_eeg_files')} | {d.get('status')} |"
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

    dem_rows = ["| <none> | <none> | <none> | <none> | <none> | <none> |"]
    dem_end = ctx.pack_dementia / "dementia_endpoints.csv"
    if dem_end.exists():
        try:
            d = pd.read_csv(dem_end)
            if not d.empty:
                dem_rows = []
                for _, r in d.iterrows():
                    dem_rows.append(
                        f"| {r.get('endpoint')} | {r.get('feature')} | {r.get('n')} | {r.get('estimate')} | {r.get('perm_p')} | {r.get('perm_q')} |"
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

    lines = [
        "# NN_FINAL_PUSH REPORT",
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
            "",
            "## Clinical Dementia endpoints",
            "| Endpoint | Feature | N | Estimate | Perm p | Perm q |",
            "|---|---|---:|---:|---:|---:|",
            *dem_rows,
            "",
            "## Figure paths",
            f"- `{ctx.pack_core / 'effect_sizes' / 'FIG_slopes_uv_per_load.png'}`",
            f"- `{ctx.pack_core / 'effect_sizes' / 'FIG_delta_uv_high_vs_low.png'}`",
            f"- `{ctx.pack_core / 'effect_sizes' / 'FIG_waveforms_by_load.png'}`",
            f"- `{ctx.pack_mechanism / 'FIG_load_vs_pupil.png'}`",
            f"- `{ctx.pack_mechanism / 'FIG_pupil_vs_p3_partial.png'}`",
            f"- `{ctx.pack_mechanism / 'FIG_mediation_ab.png'}`",
            f"- `{ctx.pack_mechanism / 'FIG_mechanism_summary.png'}`",
            f"- `{ctx.pack_pd / 'FIG_pd_deviation_by_group_medstate.png'}`",
            f"- `{ctx.pack_pd / 'FIG_pd_on_off_paired.png'}`",
            f"- `{ctx.pack_pd / 'FIG_pd_primary_auc_roc.png'}`",
            f"- `{ctx.pack_dementia / 'FIG_dementia_composite_by_group.png'}`",
            f"- `{ctx.pack_dementia / 'FIG_dementia_auc_composite.png'}`",
            f"- `{ctx.pack_dementia / 'FIG_dementia_mmse_vs_composite.png'}`",
            "",
            "## Provenance",
            f"- Repo fingerprint: `{ctx.audit_dir / 'repo_fingerprint.json'}`",
            f"- Dataset hashes: `{ctx.audit_dir / 'dataset_hashes.json'}`",
            f"- GPU monitor: `{ctx.audit_dir / 'nvidia_smi_1hz.csv'}`",
            f"- Config default: `{ctx.config}`",
            f"- Law-C event map: `{ctx.lawc_event_map}`",
            f"- Mechanism event map: `{ctx.mechanism_event_map}`",
            f"- Final bundle: `{ctx.outzip_dir / 'NN_SUBMISSION_PACKET.zip'}`",
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

    zpath = ctx.outzip_dir / "NN_SUBMISSION_PACKET.zip"
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

    ap.add_argument("--wall_hours", type=float, default=10.0)
    ap.add_argument("--lawc_n_perm", type=int, default=50000)
    ap.add_argument("--mechanism_n_perm", type=int, default=20000)
    ap.add_argument("--pd_n_perm", type=int, default=20000)
    ap.add_argument("--dementia_n_perm", type=int, default=10000)

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
    "_stage_mechanism_deep": "mechanism_deep",
    "_stage_clinical_pd": "clinical_PD_ds003490",
    "_stage_clinical_dementia": "clinical_Dementia_ds004504",
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
        out_root = Path("/filesystemHcog/runs") / f"{ts}_NN_FINAL_PUSH"

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
        _stage_mechanism_deep,
        _stage_clinical_pd,
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
            if fn.__name__ in {"_stage_clinical_pd", "_stage_clinical_dementia"} and not _wall_guard(ctx, reserve_sec=1800):
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

    report_path = ctx.audit_dir / "NN_FINAL_PUSH_REPORT.md"
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
    print(f"BUNDLE={ctx.outzip_dir / 'NN_SUBMISSION_PACKET.zip'}", flush=True)

    return 0 if run_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
