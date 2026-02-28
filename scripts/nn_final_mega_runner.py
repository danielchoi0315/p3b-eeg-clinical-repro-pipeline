#!/usr/bin/env python3
"""NN_FINAL_MEGA end-to-end orchestrator.

Stages
1) preflight
2) compile_gate
3) stage_datasets
4) decode_mapping_all
5) extract_features_all
6) core_lawc_ultradeep
7) mechanism_deep
8) normative_lodo_manyseed
9) clinical_translation
10) final_bundle

Design locks
- Fail-closed everywhere: ambiguous mapping -> STOP_REASON + SKIP dataset.
- No silent fallbacks.
- Checkpointed stage summaries; resume-safe.
- Walltime-aware: bundle partial outputs with PARTIAL_PASS when needed.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shlex
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

from common.hardware import (  # noqa: E402
    apply_cpu_thread_env,
    auto_tune_gpu_parallel_procs,
    configure_torch_backends,
    detect_hardware_info,
    start_gpu_util_logger,
    summarize_gpu_util_csv,
)
from common.lawc_audit import (  # noqa: E402
    bh_fdr,
    collect_lawc_trials_from_features,
)

try:
    from aggregate_results import _compute_repo_fingerprint
except Exception:
    _compute_repo_fingerprint = None


STAGE_ORDER: List[str] = [
    "preflight",
    "compile_gate",
    "stage_datasets",
    "decode_mapping_all",
    "extract_features_all",
    "core_lawc_ultradeep",
    "mechanism_deep",
    "normative_lodo_manyseed",
    "clinical_translation",
    "final_bundle",
]

LOCKED_LAWC_CANONICAL = ["ds003655", "ds004117", "ds005095"]

FEATURE_REUSE_ROOTS: Dict[str, Path] = {
    "ds003655": Path("/filesystemHcog/features_cache_FIX2_20260222_061927/ds003655"),
    "ds004117": Path("/filesystemHcog/features_cache_FIX2_20260222_061927/ds004117"),
    "ds005095": Path("/filesystemHcog/features_cache_FIX2_20260222_061927/ds005095"),
    "ds004796": Path("/filesystemHcog/features_cache_PEARL_SOLID2_20260222/ds004796"),
}


@dataclass
class DatasetSpec:
    dataset_id: str
    git_url: str
    pinned_hash: Optional[str]


@dataclass
class RunContext:
    out_root: Path
    audit_dir: Path
    outzip_dir: Path

    pack_core: Path
    pack_mechanism: Path
    pack_normative: Path
    pack_clinical: Path
    pack_mapping: Path
    pack_features: Path

    data_root: Path
    features_root: Path

    config: Path
    mega_config: Path
    datasets_config: Path
    lawc_event_map: Path
    mechanism_event_map: Path
    pearl_event_map: Path

    wall_hours: float
    resume: bool

    start_ts: float
    deadline_ts: float

    stage_records: List[Dict[str, Any]]
    stage_status: Dict[str, str]
    stage_outputs: Dict[str, Dict[str, Any]]

    monitor_proc: Optional[subprocess.Popen]
    monitor_handle: Optional[Any]
    nvml_logger: Optional[Any]

    runtime_env: Dict[str, str]

    cfg_mega: Dict[str, Any]
    cfg_datasets: Dict[str, Any]

    gpu_parallel_procs: int
    cpu_workers: int

    partial_reasons: List[str]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _json_sanitize(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, (np.generic,)):
        return _json_sanitize(obj.item())
    if isinstance(obj, np.ndarray):
        return [_json_sanitize(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _json_sanitize(payload)
    path.write_text(json.dumps(clean, indent=2, allow_nan=False), encoding="utf-8")


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tail(path: Path, n: int = 160) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max(1, int(n)) :])


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _parse_seed_spec(spec: str) -> List[int]:
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


def _parse_bool(raw: Any) -> bool:
    t = str(raw).strip().lower()
    return t in {"1", "true", "yes", "y", "on"}


def _stable_int_from_str(text: str) -> int:
    h = hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:12]
    return int(h, 16)


def _seconds_left(ctx: RunContext) -> float:
    return float(ctx.deadline_ts - time.time())


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
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            check=False,
        )
    rc = int(p.returncode)
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
    rec: Dict[str, Any] = {
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
    ctx.stage_outputs[stage] = rec
    return rec


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


def _print_stage_table(ctx: RunContext) -> None:
    print("PASS_PARTIAL_SKIP_TABLE_BEGIN", flush=True)
    print(_status_table(ctx), flush=True)
    print("PASS_PARTIAL_SKIP_TABLE_END", flush=True)


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


def _count_event_files(dataset_root: Path) -> int:
    return int(len(list(dataset_root.rglob("*_events.tsv"))) + len(list(dataset_root.rglob("*_events.tsv.gz"))))


_EEG_PAT = re.compile(r"_eeg\.(edf|bdf|set|vhdr|eeg|fif|gdf|cnt|fdt)(\.gz)?$", re.IGNORECASE)


def _count_eeg_payload_files(dataset_root: Path) -> int:
    n = 0
    for p in dataset_root.rglob("*"):
        if p.is_file() and _EEG_PAT.search(p.name):
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


def _git_head(path: Path) -> Optional[str]:
    try:
        p = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if p.returncode == 0:
            return p.stdout.strip()
    except Exception:
        pass
    return None


def _remote_head(url: str) -> Optional[str]:
    try:
        p = subprocess.run(["git", "ls-remote", url, "HEAD"], capture_output=True, text=True, check=False)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip().split()[0]
    except Exception:
        pass
    return None


def _write_stop_reason(path: Path, title: str, reason: str, diagnostics: Optional[Dict[str, Any]] = None) -> None:
    lines = [
        f"# STOP_REASON {title}",
        "",
        "## Why skipped",
        reason,
    ]
    if diagnostics:
        lines.extend([
            "",
            "## Diagnostics",
            "```json",
            json.dumps(diagnostics, indent=2),
            "```",
        ])
    _write_text(path, "\n".join(lines) + "\n")


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


def _start_nvidia_smi_monitor(ctx: RunContext) -> None:
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
    ctx.monitor_proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=handle, stderr=handle, text=True)


def _stop_monitors(ctx: RunContext) -> None:
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

    if ctx.nvml_logger is not None:
        try:
            ctx.nvml_logger.stop()
        except Exception:
            pass
        ctx.nvml_logger = None


def _load_resume_record(ctx: RunContext, stage: str) -> Optional[Dict[str, Any]]:
    status_path = ctx.audit_dir / f"{stage}.status"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"
    if not status_path.exists() or not summary_path.exists():
        return None
    st = status_path.read_text(encoding="utf-8", errors="ignore").strip().upper()
    if st not in {"PASS", "SKIP"}:
        return None
    rec = _read_json_if_exists(summary_path) or {}
    rec["stage"] = stage
    rec["status"] = st
    rec.setdefault("returncode", 0)
    rec.setdefault("elapsed_sec", 0.0)
    rec.setdefault("log", str(ctx.audit_dir / f"{stage}.log"))
    rec.setdefault("summary", str(summary_path))
    rec.setdefault("command", "resume-skip")
    rec.setdefault("outputs", [])
    rec.setdefault("error", "")
    return rec


def _cpu_physical_cores() -> int:
    try:
        out = subprocess.check_output(["bash", "-lc", "lscpu -p=Core | grep -v '^#' | sort -u | wc -l"], text=True)
        n = int(out.strip())
        if n > 0:
            return n
    except Exception:
        pass
    return int(os.cpu_count() or 32)


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
        for p in [ctx.pack_core, ctx.pack_mechanism, ctx.pack_normative, ctx.pack_clinical, ctx.pack_mapping, ctx.pack_features, ctx.features_root]:
            p.mkdir(parents=True, exist_ok=True)

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

        # Thread/env safety
        phys = _cpu_physical_cores()
        cpu_default = int((ctx.cfg_mega.get("analysis", {}) or {}).get("cpu_workers_default", phys))
        if cpu_default <= 0:
            cpu_default = int(phys)
        cpu_cap = bool((ctx.cfg_mega.get("analysis", {}) or {}).get("cpu_workers_cap_physical", True))
        cpu_selected = int(min(cpu_default, phys)) if cpu_cap else int(cpu_default)
        if ctx.cpu_workers <= 0:
            ctx.cpu_workers = max(1, cpu_selected)
        if cpu_cap:
            ctx.cpu_workers = int(max(1, min(ctx.cpu_workers, phys)))

        apply_cpu_thread_env(threads=max(1, ctx.cpu_workers), allow_override=True)

        # GPU proc auto-tuner
        ana = ctx.cfg_mega.get("analysis", {}) or {}
        tune = auto_tune_gpu_parallel_procs(
            min_procs=int(ana.get("gpu_parallel_procs_min", 8)),
            max_procs=int(ana.get("gpu_parallel_procs_max", 12)),
            headroom_frac=0.15,
        )
        if ctx.gpu_parallel_procs <= 0:
            ctx.gpu_parallel_procs = int(tune.get("selected_procs", 8))
        _write_json(ctx.audit_dir / "gpu_proc_tuner.json", tune)
        outputs.append(ctx.audit_dir / "gpu_proc_tuner.json")

        _start_nvidia_smi_monitor(ctx)
        try:
            ctx.nvml_logger = start_gpu_util_logger(csv_path=ctx.out_root / "gpu_util.csv", tag="NN_FINAL_MEGA")
        except Exception:
            ctx.nvml_logger = None

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
        _stop_monitors(ctx)
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


def _torch_compile_bf16_sanity() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "cuda_available": False,
        "gpu_name": "",
        "bf16_ok": False,
        "compile_ok": False,
        "compile_reason": "",
    }
    try:
        import torch

        info = detect_hardware_info()
        out["cuda_available"] = bool(info.cuda_available)
        out["gpu_name"] = str(info.gpu_name)

        configure_torch_backends(enable_tf32=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            out["compile_reason"] = "cuda-unavailable"
            return out

        x = torch.randn((1024, 1024), device=device, dtype=torch.float32)
        w = torch.randn((1024, 1024), device=device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = x @ w
        out["bf16_ok"] = bool(torch.isfinite(y).all().item())

        class Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(64, 128),
                    torch.nn.GELU(),
                    torch.nn.Linear(128, 1),
                )

            def forward(self, z: "torch.Tensor") -> "torch.Tensor":
                return self.net(z)

        m = Tiny().to(device)
        try:
            m2 = torch.compile(m, backend="inductor")
            z = torch.randn((4096, 64), device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = m2(z)
            out["compile_ok"] = True
            out["compile_reason"] = "inductor-ok"
        except Exception as exc:
            out["compile_ok"] = False
            out["compile_reason"] = f"fallback-eager:{exc}"

        return out
    except Exception as exc:
        out["compile_reason"] = f"sanity-error:{exc}"
        return out


def _stage_compile_gate(ctx: RunContext) -> Dict[str, Any]:
    stage = "compile_gate"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    cmd = ["bash", "-lc", "find . -name '*.py' -print0 | xargs -0 python -m py_compile"]
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, env=ctx.runtime_env, allow_fail=True)
    if rc != 0:
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="py_compile all .py",
            error="py_compile failed",
        )

    sanity = _torch_compile_bf16_sanity()
    _write_json(ctx.audit_dir / "compile_gate_torch_sanity.json", sanity)

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="py_compile + torch.compile/bf16 sanity",
        outputs=[ctx.audit_dir / "compile_gate_torch_sanity.json"],
        extra={"torch_sanity": sanity},
    )


def _clone_with_datalad(ds_root: Path, url: str, log_path: Path, env: Dict[str, str]) -> Tuple[int, str]:
    if shutil.which("datalad") is None:
        return 1, "datalad unavailable"

    if not (ds_root / ".git").exists():
        rc = _run_cmd(["datalad", "clone", url, str(ds_root)], cwd=REPO_ROOT, log_path=log_path, env=env, allow_fail=True)
        if rc != 0:
            return rc, "datalad clone failed"

    rc_n = _run_cmd(["datalad", "get", "-n", "."], cwd=ds_root, log_path=log_path, env=env, allow_fail=True)
    if rc_n != 0:
        return rc_n, "datalad get -n failed"
    return 0, "datalad"


def _clone_with_git(ds_root: Path, url: str, log_path: Path, env: Dict[str, str]) -> Tuple[int, str]:
    if not (ds_root / ".git").exists():
        rc = _run_cmd(["git", "clone", url, str(ds_root)], cwd=REPO_ROOT, log_path=log_path, env=env, allow_fail=True)
        if rc != 0:
            return rc, "git clone failed"
    return 0, "git"


def _annex_get_filtered(
    *,
    ds_root: Path,
    dataset_id: str,
    require_events: bool,
    jobs: int,
    log_path: Path,
    env: Dict[str, str],
) -> Tuple[int, str, int]:
    if shutil.which("git-annex") is None:
        return 0, "git-annex unavailable (skipped)", 0

    p = subprocess.run(["git", "annex", "find", "--not", "--in=here"], cwd=str(ds_root), capture_output=True, text=True, env=env, check=False)
    if p.returncode != 0:
        return int(p.returncode), "git annex find failed", 0

    missing = [x.strip() for x in str(p.stdout).splitlines() if x.strip()]
    if not missing:
        return 0, "nothing-missing", 0

    eeg_need_re = re.compile(
        r"(^|/)sub-[^/]+(/ses-[^/]+)?/eeg/.*_eeg\.(edf|bdf|set|vhdr|eeg|fif|gdf|cnt|fdt|vmrk)(\.gz)?$",
        flags=re.IGNORECASE,
    )

    keep: List[str] = []
    for rel in missing:
        low = rel.lower()
        if low in {"dataset_description.json", "participants.tsv", "participants.json", "README".lower()}:
            keep.append(rel)
            continue
        if eeg_need_re.search(rel):
            keep.append(rel)
            continue
        if require_events and (low.endswith("_events.tsv") or low.endswith("_events.tsv.gz")):
            keep.append(rel)
            continue

    keep = sorted(set(keep))
    if not keep:
        return 0, "no-target-missing", 0

    batch = 32
    got = 0
    for i in range(0, len(keep), batch):
        chunk = keep[i : i + batch]
        rc = 1
        cmd_variants = [
            ["git", "annex", "get", "--from", "OpenNeuro", "-J", str(max(1, jobs)), "--", *chunk],
            ["git", "annex", "get", "--from", "s3-PUBLIC", "-J", str(max(1, jobs)), "--", *chunk],
            ["git", "annex", "get", "-J", str(max(1, jobs)), "--", *chunk],
        ]
        for cmd in cmd_variants:
            rc = _run_cmd(cmd, cwd=ds_root, log_path=log_path, env=env, allow_fail=True)
            if rc == 0:
                break
        if rc != 0:
            return rc, "git annex get failed", got
        got += len(chunk)

    return 0, "git-annex", got


def _dataset_ready(dataset_id: str, ds_root: Path, require_events: bool) -> bool:
    if not ds_root.exists() or not (ds_root / "participants.tsv").exists():
        return False
    eeg_n = _count_eeg_payload_files(ds_root)
    if eeg_n <= 0:
        return False
    if require_events:
        if _count_event_files(ds_root) <= 0:
            return False
    return True


def _stage_stage_datasets(ctx: RunContext) -> Dict[str, Any]:
    stage = "stage_datasets"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    out_hashes = ctx.audit_dir / "dataset_hashes.json"
    manifest_dir = ctx.audit_dir / "sha256_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    ds_cfg = ctx.cfg_datasets or {}
    ds_rows = ds_cfg.get("datasets", [])

    groups = (ctx.cfg_mega.get("dataset_groups", {}) or {})
    clinical_rest = set(groups.get("clinical_rest", []))
    core_sternberg = set(groups.get("core_sternberg", []))
    mechanism = set(groups.get("mechanism", []))

    mandatory = set(core_sternberg) | set(mechanism) | {"ds004504"}

    results: List[Dict[str, Any]] = []
    hard_failures: List[str] = []

    ctx.data_root.mkdir(parents=True, exist_ok=True)

    for row in ds_rows:
        dataset_id = str(row.get("id", "")).strip()
        url = str(row.get("git_url", "")).strip()
        pinned = row.get("pinned_hash", None)
        pinned_s = str(pinned).strip() if pinned is not None else ""
        if pinned_s.lower() in {"", "none", "null", "nan"}:
            pinned_s = ""

        ds_root = ctx.data_root / dataset_id
        require_events = dataset_id not in clinical_rest

        method = "existing"
        fallback_used = False
        status = "PASS"
        reason = ""

        try:
            if not ds_root.exists() or not (ds_root / ".git").exists():
                rc, method = _clone_with_datalad(ds_root, url, log_path, ctx.runtime_env)
                if rc != 0:
                    fallback_used = True
                    rc, method = _clone_with_git(ds_root, url, log_path, ctx.runtime_env)
                    if rc != 0:
                        if ds_root.exists() and _dataset_ready(dataset_id, ds_root, require_events=require_events):
                            method = "existing_non_git"
                        else:
                            raise RuntimeError(f"clone failed ({method})")

            if pinned_s:
                rc = _run_cmd(["git", "checkout", pinned_s], cwd=ds_root, log_path=log_path, env=ctx.runtime_env, allow_fail=True)
                if rc != 0:
                    raise RuntimeError(f"git checkout failed for pinned_hash={pinned_s}")

            if _dataset_ready(dataset_id, ds_root, require_events=require_events):
                method = f"{method}+already-ready"
            else:
                rc_annex, annex_method, got_n = _annex_get_filtered(
                    ds_root=ds_root,
                    dataset_id=dataset_id,
                    require_events=require_events,
                    jobs=max(1, min(64, ctx.cpu_workers)),
                    log_path=log_path,
                    env=ctx.runtime_env,
                )
                method = f"{method}+{annex_method}"
                if rc_annex != 0:
                    raise RuntimeError("annex retrieval failed")

                ready = _dataset_ready(dataset_id, ds_root, require_events=require_events)
                if not ready:
                    raise RuntimeError("dataset not ready after staging")

        except Exception as exc:
            status = "SKIP"
            reason = str(exc)
            stop = ctx.audit_dir / "STOP_REASONS" / f"STOP_REASON_stage_datasets_{dataset_id}.md"
            _write_stop_reason(
                stop,
                f"{stage}:{dataset_id}",
                reason,
                diagnostics={
                    "dataset_id": dataset_id,
                    "url": url,
                    "pinned_hash": pinned_s or None,
                    "require_events": bool(require_events),
                },
            )
            if dataset_id in mandatory:
                hard_failures.append(f"{dataset_id}: {reason}")

        event_n = _count_event_files(ds_root) if ds_root.exists() else 0
        eeg_n = _count_eeg_payload_files(ds_root) if ds_root.exists() else 0
        broken_links = _count_eeg_broken_symlinks(ds_root) if ds_root.exists() else 0

        git_exists = (ds_root / ".git").exists()
        git_head = _git_head(ds_root) if git_exists else None
        manifest_path: Optional[Path] = None
        if git_exists and not git_head:
            status = "SKIP"
            reason = "git exists but HEAD could not be resolved"
            if dataset_id in mandatory:
                hard_failures.append(f"{dataset_id}: {reason}")
        if not git_exists and ds_root.exists():
            manifest_path = manifest_dir / f"{dataset_id}_sha256_manifest.json"
            _sha256_manifest(ds_root, manifest_path)

        results.append(
            {
                "dataset_id": dataset_id,
                "git_url": url,
                "path": str(ds_root),
                "pinned_hash": pinned_s or None,
                "git_head": git_head,
                "sha256_manifest": str(manifest_path) if manifest_path else None,
                "remote_head_commit": _remote_head(url),
                "status": status,
                "reason": reason,
                "staging_method": method,
                "fallback_used": bool(fallback_used),
                "n_event_files": int(event_n),
                "n_eeg_files": int(eeg_n),
                "n_broken_eeg_symlinks": int(broken_links),
            }
        )

    stage_status = "FAIL" if hard_failures else "PASS"
    error = "" if stage_status == "PASS" else " ; ".join(sorted(set(hard_failures)))

    payload = {
        "timestamp_utc": _iso_now(),
        "data_root": str(ctx.data_root),
        "datasets": results,
        "status": stage_status,
        "error": error,
    }
    _write_json(out_hashes, payload)

    return _record_stage(
        ctx,
        stage=stage,
        status=stage_status,
        rc=0 if stage_status == "PASS" else 1,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="dataset staging (datalad/git-annex fail-closed)",
        outputs=[out_hashes],
        error=error,
        extra={"dataset_results": results},
    )


def _load_predefined_mapping(ctx: RunContext, dataset_id: str) -> Optional[Dict[str, Any]]:
    out: Optional[Dict[str, Any]] = None

    for path in [ctx.lawc_event_map, ctx.pearl_event_map, ctx.mechanism_event_map]:
        if not path.exists():
            continue
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            d = (cfg.get("datasets") or {}) if isinstance(cfg, dict) else {}
            if dataset_id in d:
                out = dict(d.get(dataset_id) or {})
                break
        except Exception:
            continue

    return out


def _decode_mapping_generic(dataset_id: str, ds_root: Path) -> Tuple[str, str, Optional[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    events = sorted(list(ds_root.rglob("*_events.tsv")) + list(ds_root.rglob("*_events.tsv.gz")))
    if not events:
        return "SKIP", "no events.tsv found", None, pd.DataFrame(), {}

    subj_to_files: Dict[str, List[Path]] = {}
    for fp in events:
        m = re.search(r"sub-([A-Za-z0-9]+)", str(fp))
        if not m:
            continue
        subj_to_files.setdefault(m.group(1), []).append(fp)

    subjects = sorted(subj_to_files.keys())
    if len(subjects) < 10:
        return "SKIP", f"too few subjects with events ({len(subjects)})", None, pd.DataFrame(), {}

    sampled = subjects[: min(len(subjects), 30)]

    candidate_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    cand_cols = ["trial_type", "event_type", "task_role", "condition", "stim_type", "label"]
    keywords = "probe|memory|sternberg|target|workload|arith"

    for ev_col in cand_cols:
        # Evaluate text-filter candidate
        event_filter = f"{ev_col}.str.contains('{keywords}', case=False, na=False, regex=True)"

        per_subj = []
        all_levels = set()
        finite_rates = []

        # load candidates
        direct_cols = ["memory_load", "set_size", "setsize", "load", "n_items", "memory_cond", "value"]

        chosen_load_col = ""
        chosen_load_regex = ""

        for sid in sampled:
            files = subj_to_files.get(sid, [])
            if not files:
                continue
            sdf_parts = []
            for fp in files[:4]:
                try:
                    sdf_parts.append(pd.read_csv(fp, sep="\t"))
                except Exception:
                    continue
            if not sdf_parts:
                continue
            sdf = pd.concat(sdf_parts, axis=0, ignore_index=True)
            if ev_col not in sdf.columns:
                continue
            try:
                sel = sdf.query(event_filter, engine="python").copy()
            except Exception:
                sel = sdf.iloc[0:0].copy()
            if sel.empty:
                continue

            # pick load extraction method for this subject
            load = np.full(len(sel), np.nan, dtype=float)
            local_col = ""
            local_regex = ""

            for c in direct_cols:
                if c in sel.columns:
                    vv = pd.to_numeric(sel[c], errors="coerce").to_numpy(dtype=float)
                    if np.isfinite(vv).mean() >= 0.7:
                        local_col = c
                        load = vv
                        break

            if not local_col:
                text = sel[ev_col].astype(str)
                ex = text.str.extract(r"(\d+)", expand=False)
                vv = pd.to_numeric(ex, errors="coerce").to_numpy(dtype=float)
                if np.isfinite(vv).mean() >= 0.7:
                    local_col = ev_col
                    local_regex = "(\\d+)"
                    load = vv

            if not local_col:
                continue

            lv = sorted(set(load[np.isfinite(load)].tolist()))
            finite_rate = float(np.isfinite(load).mean())
            per_subj.append({"subject_id": sid, "n_rows": int(len(sel)), "finite_rate": finite_rate, "n_levels": int(len(lv))})
            finite_rates.append(finite_rate)
            all_levels.update(lv)

            if not chosen_load_col:
                chosen_load_col = local_col
                chosen_load_regex = local_regex

        n_sub = len(per_subj)
        n_levels = len(all_levels)
        med_finite = float(np.median(finite_rates)) if finite_rates else float("nan")

        pass_gate = bool(n_sub >= 10 and np.isfinite(med_finite) and med_finite >= 0.8 and 2 <= n_levels <= 12)
        score = float((0 if not np.isfinite(med_finite) else med_finite) + min(1.0, n_sub / 30.0) + (min(n_levels, 12) / 12.0))

        row = {
            "dataset_id": dataset_id,
            "candidate_id": f"{ev_col}_keyword_filter",
            "status": "PASS" if pass_gate else "SKIP",
            "reason": "" if pass_gate else "failed gates",
            "event_filter": event_filter,
            "load_column": chosen_load_col,
            "load_regex": chosen_load_regex,
            "n_subjects_eval": int(n_sub),
            "n_load_levels": int(n_levels),
            "finite_rate_median": med_finite,
            "score": score,
        }
        candidate_rows.append(row)

        if pass_gate:
            cand_map: Dict[str, Any] = {
                "event_filter": event_filter,
                "load_column": chosen_load_col,
                "rt_strategy": "next_response_any",
            }
            if chosen_load_regex:
                cand_map["load_regex"] = chosen_load_regex
            # Response filter heuristic.
            cand_map["response_filter"] = "trial_type.str.contains('response|click', case=False, na=False, regex=True)"
            if best is None or float(score) > float(best["score"]):
                best = {"mapping": cand_map, "score": score, "candidate": row}

    cand_df = pd.DataFrame(candidate_rows)
    if best is None:
        return "SKIP", "no mapping candidate passed strict gates", None, cand_df, {
            "n_events_files": int(len(events)),
            "n_subjects_with_events": int(len(subjects)),
        }

    return "PASS", "", dict(best["mapping"]), cand_df, {
        "n_events_files": int(len(events)),
        "n_subjects_with_events": int(len(subjects)),
        "selected_candidate": dict(best["candidate"]),
    }


def _stage_decode_mapping_all(ctx: RunContext) -> Dict[str, Any]:
    stage = "decode_mapping_all"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    groups = (ctx.cfg_mega.get("dataset_groups", {}) or {})
    event_datasets = sorted(
        set(groups.get("core_sternberg", []))
        | set(groups.get("sternberg_generalization", []))
        | set(groups.get("workload_expansion", []))
        | set(groups.get("mechanism", []))
    )

    ds_hashes = _read_json_if_exists(ctx.audit_dir / "dataset_hashes.json") or {}
    staged_map = {str(r.get("dataset_id")): str(r.get("status", "")) for r in ds_hashes.get("datasets", [])}

    defaults = {}
    if ctx.lawc_event_map.exists():
        try:
            base = yaml.safe_load(ctx.lawc_event_map.read_text(encoding="utf-8")) or {}
            defaults = dict(base.get("defaults") or {}) if isinstance(base, dict) else {}
        except Exception:
            defaults = {}

    out_event_map = {"defaults": defaults, "datasets": {}}
    rows: List[Dict[str, Any]] = []
    outputs: List[Path] = []

    mandatory = set(groups.get("core_sternberg", []))
    hard_failures: List[str] = []

    for ds in event_datasets:
        ds_dir = ctx.data_root / ds
        ds_out = ctx.pack_mapping / ds
        ds_out.mkdir(parents=True, exist_ok=True)

        cand_csv = ds_out / "CANDIDATE_TABLE.csv"
        ds_summary = ds_out / "mapping_decode_summary.json"
        ds_map_yaml = ds_out / "event_map_autogen.yaml"
        stop = ds_out / "STOP_REASON.md"

        st_status = staged_map.get(ds, "")
        if st_status != "PASS":
            reason = f"dataset not staged PASS (status={st_status or 'missing'})"
            _write_stop_reason(stop, f"{stage}:{ds}", reason, diagnostics={"dataset_status": st_status})
            _write_json(ds_summary, {"dataset_id": ds, "status": "SKIP", "reason": reason, "stop_reason": str(stop)})
            rows.append({"dataset_id": ds, "status": "SKIP", "reason": reason, "selected_candidate_id": ""})
            if ds in mandatory:
                hard_failures.append(f"{ds}: {reason}")
            continue

        predefined = _load_predefined_mapping(ctx, ds)
        if predefined is not None:
            pd.DataFrame([{"dataset_id": ds, "candidate_id": "predefined_locked", "status": "PASS", "reason": "", "score": 1.0}]).to_csv(cand_csv, index=False)
            _write_json(ds_summary, {"dataset_id": ds, "status": "PASS", "reason": "", "selected_candidate_id": "predefined_locked", "mapping": predefined})
            _write_text(ds_map_yaml, yaml.safe_dump({"defaults": defaults, "datasets": {ds: predefined}}, sort_keys=False))
            out_event_map["datasets"][ds] = predefined
            rows.append({"dataset_id": ds, "status": "PASS", "reason": "", "selected_candidate_id": "predefined_locked", "mapping": predefined})
            outputs.extend([cand_csv, ds_summary, ds_map_yaml])
            continue

        status, reason, mapping, cand_df, diag = _decode_mapping_generic(ds, ds_dir)
        if cand_df.empty:
            pd.DataFrame(columns=["dataset_id", "candidate_id", "status", "reason", "event_filter", "load_column", "n_load_levels", "finite_rate_median", "score"]).to_csv(cand_csv, index=False)
        else:
            cand_df.to_csv(cand_csv, index=False)

        if status != "PASS" or mapping is None:
            _write_stop_reason(stop, f"{stage}:{ds}", reason, diagnostics=diag)
            _write_json(ds_summary, {"dataset_id": ds, "status": "SKIP", "reason": reason, "diagnostics": diag, "stop_reason": str(stop)})
            rows.append({"dataset_id": ds, "status": "SKIP", "reason": reason, "selected_candidate_id": ""})
            if ds in mandatory:
                hard_failures.append(f"{ds}: {reason}")
        else:
            _write_json(ds_summary, {"dataset_id": ds, "status": "PASS", "reason": "", "mapping": mapping, "diagnostics": diag})
            _write_text(ds_map_yaml, yaml.safe_dump({"defaults": defaults, "datasets": {ds: mapping}}, sort_keys=False))
            out_event_map["datasets"][ds] = mapping
            rows.append({"dataset_id": ds, "status": "PASS", "reason": "", "selected_candidate_id": "generic", "mapping": mapping})
            outputs.append(ds_map_yaml)

        outputs.extend([cand_csv, ds_summary])

    combined_event_map = ctx.audit_dir / "event_map_autogen_mega.yaml"
    _write_text(combined_event_map, yaml.safe_dump(out_event_map, sort_keys=False))
    outputs.append(combined_event_map)

    summary = {
        "status": "FAIL" if hard_failures else "PASS",
        "rows": rows,
        "combined_event_map": str(combined_event_map),
        "error": " ; ".join(hard_failures) if hard_failures else "",
    }
    _write_json(ctx.pack_mapping / "mapping_decode_summary.json", summary)
    outputs.append(ctx.pack_mapping / "mapping_decode_summary.json")

    return _record_stage(
        ctx,
        stage=stage,
        status="FAIL" if hard_failures else "PASS",
        rc=1 if hard_failures else 0,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="decode mapping for all event datasets",
        outputs=outputs,
        error=" ; ".join(hard_failures) if hard_failures else "",
        extra={"mapping_rows": rows, "combined_event_map": str(combined_event_map)},
    )


def _symlink_dataset_features(src_dataset_dir: Path, dst_features_root: Path, dataset_id: str) -> bool:
    if not src_dataset_dir.exists():
        return False
    if not any(src_dataset_dir.rglob("*.h5")):
        return False

    dst = dst_features_root / dataset_id
    if dst.exists() or dst.is_symlink():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src_dataset_dir, target_is_directory=True)
    except Exception:
        _copy_tree_merge(src_dataset_dir, dst)
    return True


def _h5_trial_count(dataset_dir: Path) -> int:
    n = 0
    for fp in dataset_dir.rglob("*.h5"):
        try:
            with h5py.File(fp, "r") as h:
                if "p3b_amp" in h:
                    n += int(np.asarray(h["p3b_amp"]).shape[0])
        except Exception:
            continue
    return int(n)


def _stage_extract_features_all(ctx: RunContext) -> Dict[str, Any]:
    stage = "extract_features_all"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    groups = (ctx.cfg_mega.get("dataset_groups", {}) or {})
    event_datasets = sorted(
        set(groups.get("core_sternberg", []))
        | set(groups.get("sternberg_generalization", []))
        | set(groups.get("workload_expansion", []))
        | set(groups.get("mechanism", []))
    )
    rest_datasets = set(groups.get("clinical_rest", []))

    map_summary = _read_json_if_exists(ctx.pack_mapping / "mapping_decode_summary.json") or {}
    map_rows = {str(r.get("dataset_id")): str(r.get("status", "")) for r in map_summary.get("rows", [])}
    combined_event_map = Path(str(map_summary.get("combined_event_map", ctx.audit_dir / "event_map_autogen_mega.yaml")))

    ds_hashes = _read_json_if_exists(ctx.audit_dir / "dataset_hashes.json") or {}
    staged_map = {str(r.get("dataset_id")): str(r.get("status", "")) for r in ds_hashes.get("datasets", [])}
    eeg_counts = {str(r.get("dataset_id")): int(r.get("n_eeg_files", 0) or 0) for r in ds_hashes.get("datasets", [])}

    mandatory = set(groups.get("core_sternberg", [])) | set(groups.get("mechanism", []))
    hard_failures: List[str] = []

    rows: List[Dict[str, Any]] = []
    outputs: List[Path] = []

    ctx.features_root.mkdir(parents=True, exist_ok=True)

    for ds in event_datasets:
        row: Dict[str, Any] = {
            "dataset_id": ds,
            "status": "SKIP",
            "reason": "",
            "method": "",
            "n_h5": 0,
            "n_trials": 0,
        }

        if staged_map.get(ds) != "PASS":
            row["reason"] = f"dataset staging status={staged_map.get(ds, 'missing')}"
            stop = ctx.pack_features / ds / "STOP_REASON.md"
            _write_stop_reason(stop, f"{stage}:{ds}", row["reason"])
            if ds in mandatory:
                hard_failures.append(f"{ds}: {row['reason']}")
            rows.append(row)
            continue

        if ds not in groups.get("mechanism", []) and map_rows.get(ds) != "PASS":
            row["reason"] = f"mapping status={map_rows.get(ds, 'missing')}"
            stop = ctx.pack_features / ds / "STOP_REASON.md"
            _write_stop_reason(stop, f"{stage}:{ds}", row["reason"])
            if ds in mandatory:
                hard_failures.append(f"{ds}: {row['reason']}")
            rows.append(row)
            continue

        reused = False
        src = FEATURE_REUSE_ROOTS.get(ds)
        if src is not None:
            reused = _symlink_dataset_features(src, ctx.features_root, ds)

        if reused:
            ds_feat = ctx.features_root / ds
            row["status"] = "PASS"
            row["method"] = "reuse_symlink"
            row["n_h5"] = int(len(list(ds_feat.rglob("*.h5"))))
            row["n_trials"] = _h5_trial_count(ds_feat)
            rows.append(row)
            continue

        # No walltime/runtime-budget skip gate: always attempt extraction.

        # Attempt full preprocess + extraction.
        ds_root = ctx.data_root / ds
        deriv_root = ctx.pack_features / ds / "derivatives"
        deriv_root.mkdir(parents=True, exist_ok=True)

        cpu_run_workers = max(1, min(64, ctx.cpu_workers))
        cpu_run_threads = max(1, ctx.cpu_workers // cpu_run_workers)

        rc1 = _run_cmd(
            [
                sys.executable,
                "01_preprocess_CPU.py",
                "--bids_root",
                str(ds_root),
                "--deriv_root",
                str(deriv_root),
                "--config",
                str(ctx.config),
                "--workers",
                str(cpu_run_workers),
                "--per_run_threads",
                str(cpu_run_threads),
            ],
            cwd=REPO_ROOT,
            log_path=log_path,
            env=ctx.runtime_env,
            allow_fail=True,
        )

        rc2 = 1
        if rc1 == 0:
            cohort = "mechanism" if ds in groups.get("mechanism", []) else "healthy"
            rc2 = _run_cmd(
                [
                    sys.executable,
                    "02_extract_features_CPU.py",
                    "--bids_root",
                    str(ds_root),
                    "--deriv_root",
                    str(deriv_root),
                    "--features_root",
                    str(ctx.features_root),
                    "--dataset_id",
                    ds,
                    "--cohort",
                    cohort,
                    "--lawc_event_map",
                    str(combined_event_map),
                    "--config",
                    str(ctx.config),
                    "--workers",
                    str(cpu_run_workers),
                    "--per_run_threads",
                    str(cpu_run_threads),
                ],
                cwd=REPO_ROOT,
                log_path=log_path,
                env=ctx.runtime_env,
                allow_fail=True,
            )

        if rc1 != 0 or rc2 != 0:
            row["status"] = "SKIP"
            row["reason"] = f"feature extraction failed rc_pre={rc1} rc_extract={rc2}"
            stop = ctx.pack_features / ds / "STOP_REASON.md"
            _write_stop_reason(stop, f"{stage}:{ds}", row["reason"])
            if ds in mandatory:
                hard_failures.append(f"{ds}: {row['reason']}")
            rows.append(row)
            continue

        ds_feat = ctx.features_root / ds
        row["status"] = "PASS"
        row["method"] = "preprocess+extract"
        row["n_h5"] = int(len(list(ds_feat.rglob("*.h5"))))
        row["n_trials"] = _h5_trial_count(ds_feat)
        if row["n_h5"] <= 0:
            row["status"] = "SKIP"
            row["reason"] = "no HDF5 outputs after extraction"
            if ds in mandatory:
                hard_failures.append(f"{ds}: {row['reason']}")
        rows.append(row)

    # Resting datasets are handled in clinical stage; we still record as intentional skip here.
    for ds in sorted(rest_datasets):
        rows.append(
            {
                "dataset_id": ds,
                "status": "SKIP",
                "reason": "resting dataset handled in clinical_translation stage",
                "method": "resting_path",
                "n_h5": 0,
                "n_trials": 0,
            }
        )

    feat_summary_csv = ctx.pack_features / "features_summary_all.csv"
    feat_summary_json = ctx.pack_features / "features_summary_all.json"
    pd.DataFrame(rows).to_csv(feat_summary_csv, index=False)
    _write_json(feat_summary_json, {"rows": rows})
    outputs.extend([feat_summary_csv, feat_summary_json])

    status = "FAIL" if hard_failures else "PASS"
    error = " ; ".join(hard_failures) if hard_failures else ""

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=1 if status == "FAIL" else 0,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="reuse/extract features for all datasets",
        outputs=outputs,
        error=error,
        extra={"feature_rows": rows},
    )


def _random_effects_meta(rows: pd.DataFrame) -> Dict[str, Any]:
    if rows.empty:
        return {"k": 0, "mu_random": float("nan"), "tau2": float("nan"), "I2": float("nan")}

    eff = pd.to_numeric(rows["median_rho"], errors="coerce").to_numpy(dtype=float)
    n = pd.to_numeric(rows.get("n_subjects_used", rows.get("n_subjects", np.nan)), errors="coerce").to_numpy(dtype=float)
    var = 1.0 / np.maximum(n - 3.0, 1.0)

    m = np.isfinite(eff) & np.isfinite(var) & (var > 0)
    eff = eff[m]
    var = var[m]
    if eff.size == 0:
        return {"k": 0, "mu_random": float("nan"), "tau2": float("nan"), "I2": float("nan")}

    w = 1.0 / var
    mu_fixed = float(np.sum(w * eff) / np.sum(w))
    q = float(np.sum(w * (eff - mu_fixed) ** 2))
    k = int(len(eff))
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = float(max(0.0, (q - (k - 1)) / max(c, 1e-12)))
    w_star = 1.0 / (var + tau2)
    mu_random = float(np.sum(w_star * eff) / np.sum(w_star))
    i2 = float(max(0.0, ((q - (k - 1)) / max(q, 1e-12)) * 100.0))

    return {
        "k": int(k),
        "mu_fixed": mu_fixed,
        "mu_random": mu_random,
        "tau2": tau2,
        "I2": i2,
        "Q": q,
    }


def _stage_core_lawc_ultradeep(ctx: RunContext) -> Dict[str, Any]:
    stage = "core_lawc_ultradeep"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    out_dir = ctx.pack_core
    out_dir.mkdir(parents=True, exist_ok=True)
    lawc_root = out_dir / "lawc_ultradeep"
    effects_root = out_dir / "effect_sizes"

    groups = (ctx.cfg_mega.get("dataset_groups", {}) or {})
    lawc_pool = sorted(set(groups.get("core_sternberg", [])) | set(groups.get("sternberg_generalization", [])) | set(groups.get("workload_expansion", [])))

    candidates: List[str] = []
    n_trials_map: Dict[str, int] = {}
    for ds in lawc_pool:
        df = pd.DataFrame()
        ds_feat = ctx.features_root / ds
        search_roots: List[Path] = [ctx.features_root]
        if ds_feat.exists() or ds_feat.is_symlink():
            try:
                search_roots = [ds_feat.resolve() if ds_feat.is_symlink() else ds_feat]
            except Exception:
                search_roots = [ds_feat]
        for root in search_roots:
            df = collect_lawc_trials_from_features(root, ds)
            if len(df) > 0:
                break
        ntr = int(len(df))
        if ntr > 0:
            candidates.append(ds)
            n_trials_map[ds] = ntr

    if not candidates:
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="lawc ultradeep",
            error="no candidate datasets with extracted features",
        )

    ana = ctx.cfg_mega.get("analysis", {}) or {}
    nperm_cfg = int(ana.get("lawc_n_perm", 100000))
    nperm_min = int(ana.get("lawc_n_perm_min", 20000))
    factor = int(ana.get("lawc_n_perm_adaptive_factor", 10))

    cand_perm = [max(nperm_min, min(nperm_cfg, factor * int(n_trials_map.get(ds, 0)))) for ds in candidates]
    n_perm = int(max(cand_perm)) if cand_perm else int(nperm_min)
    n_perm = max(nperm_min, min(nperm_cfg, n_perm))

    # Materialize candidate feature directories as real directories (not symlink dirs)
    # so downstream recursive file scans cannot miss HDF5 files.
    lawc_features_root = out_dir / "_lawc_features_materialized"
    lawc_features_root.mkdir(parents=True, exist_ok=True)
    for ds in candidates:
        src = ctx.features_root / ds
        if src.is_symlink():
            src = src.resolve()
        if not src.exists():
            continue
        dst = lawc_features_root / ds
        if dst.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        rc_mat = _run_cmd(
            ["bash", "-lc", f"cp -al {shlex.quote(str(src))}/. {shlex.quote(str(dst))}/"],
            cwd=REPO_ROOT,
            log_path=log_path,
            env=ctx.runtime_env,
            allow_fail=True,
        )
        if rc_mat != 0:
            _copy_tree_merge(src, dst)

    event_map_path = ctx.audit_dir / "event_map_autogen_mega.yaml"
    if not event_map_path.exists():
        event_map_path = ctx.lawc_event_map

    # Locked Law-C gate must remain canonical and fail-closed on those datasets only.
    run_datasets = [ds for ds in candidates if ds in LOCKED_LAWC_CANONICAL]
    if not run_datasets:
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="select canonical Law-C datasets",
            error="none of canonical locked Law-C datasets have extracted features",
            extra={"lawc_candidates": candidates},
        )

    lawc_json = lawc_root / "lawc_audit" / "locked_test_results.json"
    lawc_csv = lawc_root / "lawc_audit" / "locked_test_results.csv"
    neg_csv = lawc_root / "lawc_audit" / "negative_controls.csv"

    rc_lawc = 0
    used_cached_lawc = False
    if lawc_json.exists() and lawc_csv.exists() and neg_csv.exists():
        try:
            cached_rows = pd.read_csv(lawc_csv)
            cached_pass = {
                str(r["dataset_id"]): bool(r.get("pass_all", False))
                for _, r in cached_rows.iterrows()
            } if not cached_rows.empty else {}
            missing = [ds for ds in run_datasets if ds not in cached_pass]
            failed = [ds for ds in run_datasets if not cached_pass.get(ds, False)]
            if not missing and not failed:
                used_cached_lawc = True
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"[{_iso_now()}] INFO: reusing cached Law-C outputs for canonical datasets: "
                        + ",".join(run_datasets)
                        + "\n"
                    )
            else:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"[{_iso_now()}] INFO: cached Law-C outputs invalid for canonical gate "
                        f"(missing={missing}, failed={failed}); rerunning.\n"
                    )
        except Exception as exc:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"[{_iso_now()}] INFO: failed to validate cached Law-C outputs ({exc}); rerunning.\n")

    if not used_cached_lawc:
        rc_lawc = _run_cmd(
            [
                sys.executable,
                "05_audit_lawc.py",
                "--features_root",
                str(lawc_features_root),
                "--out_root",
                str(lawc_root),
                "--event_map",
                str(event_map_path),
                "--datasets",
                ",".join(run_datasets),
                "--n_perm",
                str(int(n_perm)),
                "--workers",
                str(max(1, min(ctx.cpu_workers, len(run_datasets)))),
            ],
            cwd=REPO_ROOT,
            log_path=log_path,
            env=ctx.runtime_env,
            allow_fail=True,
        )

    rc_eff = _run_cmd(
        [
            sys.executable,
            "scripts/effect_size_pack.py",
            "--features_root",
            str(ctx.features_root),
            "--datasets",
            ",".join(candidates),
            "--out_dir",
            str(effects_root),
            "--n_boot",
            "5000",
            "--seed",
            "123",
        ],
        cwd=REPO_ROOT,
        log_path=log_path,
        env=ctx.runtime_env,
        allow_fail=True,
    )

    effect_csv = effects_root / "effect_size_summary.csv"

    if rc_lawc != 0 or not lawc_json.exists() or not lawc_csv.exists() or not neg_csv.exists():
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="run 05_audit_lawc.py",
            outputs=[lawc_json, lawc_csv, neg_csv],
            error=f"lawc command failed rc={rc_lawc}",
        )

    lawc_rows = pd.read_csv(lawc_csv)
    pass_all_map = {str(r["dataset_id"]): bool(r.get("pass_all", False)) for _, r in lawc_rows.iterrows()} if not lawc_rows.empty else {}

    hard_fail = []
    for ds in LOCKED_LAWC_CANONICAL:
        if ds in candidates and not pass_all_map.get(ds, False):
            hard_fail.append(f"{ds}: fail-closed gate (primary+controls)")

    meta = _random_effects_meta(lawc_rows[lawc_rows["pass_all"] == True] if "pass_all" in lawc_rows.columns else lawc_rows)
    meta_path = lawc_root / "lawc_audit" / "meta_random_effects.json"
    _write_json(meta_path, meta)

    outputs = [lawc_json, lawc_csv, neg_csv, meta_path]
    if effect_csv.exists():
        outputs.append(effect_csv)

    status = "FAIL" if hard_fail else "PASS"
    err = " ; ".join(hard_fail) if hard_fail else ""
    if rc_eff != 0 and status == "PASS":
        # Effect-size pack is required for final figures, but not for locked primary gate.
        status = "SKIP"
        err = f"effect_size_pack failed rc={rc_eff}"
        ctx.partial_reasons.append("core effect-size pack missing")

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=1 if status == "FAIL" else 0,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="Law-C ultradeep + effect sizes + meta-analysis",
        outputs=outputs,
        error=err,
        extra={
            "lawc_candidates": candidates,
            "lawc_run_datasets": run_datasets,
            "lawc_n_perm": int(n_perm),
            "meta_random_effects": meta,
        },
    )


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
        "- `a`: effect of load on pupil proxy (pupil AUC per load increment).",
        "- `b`: effect of pupil proxy on P3 amplitude controlling load (uV per pupil AUC).",
        "- `c_prime`: direct effect of load on P3 controlling pupil (uV per load increment).",
        "- `ab`: indirect mediation term from the same fitted mechanism model.",
        "",
        "## Arithmetic Sanity",
        f"- Observed `a`: {a}",
        f"- Observed `b`: {b}",
        f"- Reported `ab`: {ab}",
        f"- Product `a*b`: {ab_prod}",
        f"- Sign match (`ab` vs `a*b`): {sign_match}",
        f"- Ratio `ab/(a*b)`: {ratio}",
        "",
        "## CI / P-value basis",
        "- CI values are from `aggregate_mechanism.json` (`observed_ci95`) and correspond to the same model output as p-values.",
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


def _stage_mechanism_deep(ctx: RunContext) -> Dict[str, Any]:
    stage = "mechanism_deep"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    out_dir = ctx.pack_mechanism
    out_dir.mkdir(parents=True, exist_ok=True)

    features_ds = ctx.features_root / "ds003838"
    if not features_ds.exists() or not any(features_ds.rglob("*.h5")):
        reason = "missing ds003838 features"
        _write_stop_reason(out_dir / "STOP_REASON.md", stage, reason)
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="mechanism deep",
            outputs=[out_dir / "STOP_REASON.md"],
            error=reason,
        )

    ana = ctx.cfg_mega.get("analysis", {}) or {}
    seed_target = str(ana.get("mechanism_seeds", "0-199"))
    seed_min = str(ana.get("mechanism_min_seeds", "0-49"))

    seed_spec = seed_target
    if not _parse_seed_spec(seed_spec):
        seed_spec = seed_min

    n_perm = int(max(1000, int(ana.get("mechanism_n_perm", 10000))))

    cmd = [
        sys.executable,
        "scripts/mechanism_deep.py",
        "--features_root",
        str(ctx.features_root),
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
        str(n_perm),
        "--min_trials",
        "20",
    ]
    if ctx.resume:
        cmd.extend(["--resume"])

    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, env=ctx.runtime_env, allow_fail=True)

    req = [
        out_dir / "Table_mechanism_effects.csv",
        out_dir / "aggregate_mechanism.json",
        out_dir / "FIG_load_vs_pupil.png",
        out_dir / "FIG_pupil_vs_p3_partial.png",
        out_dir / "FIG_mediation_ab.png",
        out_dir / "FIG_mechanism_summary.png",
    ]

    ok_fix, fix_err = _mechanism_auditfix(out_dir)
    missing = [str(p) for p in req if not p.exists()]

    status = "PASS" if rc == 0 and ok_fix and not missing else "FAIL"
    error = "" if status == "PASS" else f"mechanism failed rc={rc} fix={ok_fix} fix_err={fix_err} missing={missing}"

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=0 if status == "PASS" else 1,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="mechanism_deep.py + auditfix",
        outputs=req + [out_dir / "MECHANISM_SANITY.md", out_dir / "mechanism_negative_controls_summary.json"],
        error=error,
        extra={"seeds": seed_spec, "n_perm": int(n_perm)},
    )


def _collect_normative_trials(features_root: Path, dataset_id: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    droot = features_root / dataset_id
    if not droot.exists():
        return pd.DataFrame()

    for fp in sorted(droot.rglob("*.h5")):
        try:
            with h5py.File(fp, "r") as h:
                if "p3b_amp" not in h or "memory_load" not in h:
                    continue
                n = int(np.asarray(h["p3b_amp"]).shape[0])
                if n <= 0:
                    continue
                p3 = np.asarray(h["p3b_amp"], dtype=float)
                load = np.asarray(h["memory_load"], dtype=float)
                trial_order = np.asarray(h["trial_order"], dtype=float) if "trial_order" in h else np.arange(1, n + 1, dtype=float)
                age = np.asarray(h["age"], dtype=float) if "age" in h else np.full(n, np.nan, dtype=float)
                skey = np.asarray(h["subject_key"]).astype(str) if "subject_key" in h else np.asarray([h.attrs.get("subject_key", "")] * n).astype(str)
                rows.append(
                    pd.DataFrame(
                        {
                            "dataset_id": [dataset_id] * n,
                            "subject_key": skey,
                            "p3b_amp": p3,
                            "memory_load": load,
                            "trial_order": trial_order,
                            "age": age,
                        }
                    )
                )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, axis=0, ignore_index=True)
    out = out[np.isfinite(out["p3b_amp"]) & np.isfinite(out["memory_load"])].copy()
    return out


def _gpu_stage_util_summary(gpu_csv: Path, start_ts: float, end_ts: float) -> Dict[str, Any]:
    out = {
        "stage_util_gpu_mean": float("nan"),
        "stage_util_gpu_median": float("nan"),
        "pre_util_gpu_mean": float("nan"),
        "util_rise_confirmed": False,
    }
    if not gpu_csv.exists():
        return out
    try:
        df = pd.read_csv(gpu_csv)
    except Exception:
        return out
    if df.empty or "unix_time" not in df.columns or "util_gpu_pct" not in df.columns:
        return out

    t = pd.to_numeric(df["unix_time"], errors="coerce").to_numpy(dtype=float)
    u = pd.to_numeric(df["util_gpu_pct"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(t) & np.isfinite(u)
    t = t[m]
    u = u[m]
    if t.size == 0:
        return out

    m_stage = (t >= float(start_ts)) & (t <= float(end_ts))
    m_pre = (t >= float(start_ts - 60.0)) & (t < float(start_ts))

    if int(m_stage.sum()) > 0:
        out["stage_util_gpu_mean"] = float(np.mean(u[m_stage]))
        out["stage_util_gpu_median"] = float(np.median(u[m_stage]))
    if int(m_pre.sum()) > 0:
        out["pre_util_gpu_mean"] = float(np.mean(u[m_pre]))

    if np.isfinite(out["stage_util_gpu_mean"]) and np.isfinite(out["pre_util_gpu_mean"]):
        out["util_rise_confirmed"] = bool(out["stage_util_gpu_mean"] > out["pre_util_gpu_mean"] + 1.0)

    return out


def _stage_normative_lodo_manyseed(ctx: RunContext) -> Dict[str, Any]:
    stage = "normative_lodo_manyseed"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    out_dir = ctx.pack_normative
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = (ctx.cfg_mega.get("dataset_groups", {}) or {})
    lodo_pool = sorted(set(groups.get("core_sternberg", [])) | set(groups.get("sternberg_generalization", [])) | set(groups.get("workload_expansion", [])))

    dfs: List[pd.DataFrame] = []
    for ds in lodo_pool:
        d = _collect_normative_trials(ctx.features_root, ds)
        if not d.empty:
            dfs.append(d)

    if len(dfs) < 2:
        reason = "need >=2 datasets with extracted trials for LODO"
        stop = out_dir / "STOP_REASON.md"
        _write_stop_reason(stop, stage, reason, diagnostics={"datasets_with_trials": [str(d["dataset_id"].iloc[0]) for d in dfs]})
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="normative lodo",
            outputs=[stop],
            error=reason,
        )

    all_df = pd.concat(dfs, axis=0, ignore_index=True)
    all_df = all_df[np.isfinite(pd.to_numeric(all_df["p3b_amp"], errors="coerce")) & np.isfinite(pd.to_numeric(all_df["memory_load"], errors="coerce"))].copy()
    if all_df.empty:
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="normative lodo",
            error="empty trial table after filtering",
        )

    ana = ctx.cfg_mega.get("analysis", {}) or {}
    seed_target = _parse_seed_spec(str(ana.get("normative_seeds", "0-499")))
    seed_min = _parse_seed_spec(str(ana.get("normative_min_seeds", "0-199")))

    seeds = seed_target
    if len(seeds) < len(seed_min):
        seeds = seed_min

    use_cuda = False
    compile_ok = False
    compile_reason = ""
    preload_gpu = False

    try:
        import torch

        configure_torch_backends(enable_tf32=True)
        use_cuda = bool(torch.cuda.is_available())
        device = torch.device("cuda" if use_cuda else "cpu")

        folds = sorted(all_df["dataset_id"].astype(str).unique().tolist())
        if len(folds) < 2:
            raise RuntimeError("need >=2 LODO folds")

        # Prepare arrays.
        all_df["age"] = pd.to_numeric(all_df["age"], errors="coerce")
        all_df["trial_order"] = pd.to_numeric(all_df["trial_order"], errors="coerce")
        all_df["memory_load"] = pd.to_numeric(all_df["memory_load"], errors="coerce")
        all_df["p3b_amp"] = pd.to_numeric(all_df["p3b_amp"], errors="coerce")

        # Impute global medians for stable tensor shapes.
        for col in ["age", "trial_order", "memory_load"]:
            med = float(np.nanmedian(all_df[col].to_numpy(dtype=float))) if np.isfinite(all_df[col]).any() else 0.0
            all_df[col] = all_df[col].fillna(med)

        X_np = all_df[["memory_load", "age", "trial_order"]].to_numpy(dtype=np.float32)
        y_np = all_df["p3b_amp"].to_numpy(dtype=np.float32)
        ds_np = all_df["dataset_id"].astype(str).to_numpy(dtype=object)

        if use_cuda:
            free_b, _ = torch.cuda.mem_get_info()
            need_b = int(X_np.nbytes + y_np.nbytes)
            preload_gpu = bool(need_b < 0.6 * float(free_b))

        X_base = torch.from_numpy(X_np)
        y_base = torch.from_numpy(y_np)
        if preload_gpu and use_cuda:
            X_base = X_base.to(device, non_blocking=True)
            y_base = y_base.to(device, non_blocking=True)

        def pred_fn(xx: "torch.Tensor", bb: "torch.Tensor") -> "torch.Tensor":
            return xx @ bb

        pred = pred_fn
        if use_cuda and bool((ctx.cfg_mega.get("runtime", {}).get("defaults", {}) or {}).get("try_torch_compile", True)):
            try:
                pred = torch.compile(pred_fn, backend="inductor")
                compile_ok = True
                compile_reason = "inductor-ok"
            except Exception as exc:
                compile_ok = False
                compile_reason = f"fallback-eager:{exc}"

        rows: List[Dict[str, Any]] = []

        lam = 1e-3
        for fold in folds:
            test_idx = np.where(ds_np == fold)[0]
            train_idx = np.where(ds_np != fold)[0]
            if len(test_idx) < 20 or len(train_idx) < 100:
                continue

            if preload_gpu and use_cuda:
                X_train = X_base[torch.as_tensor(train_idx, device=device)]
                y_train = y_base[torch.as_tensor(train_idx, device=device)]
                X_test = X_base[torch.as_tensor(test_idx, device=device)]
                y_test = y_base[torch.as_tensor(test_idx, device=device)]
            else:
                X_train = X_base[train_idx].to(device)
                y_train = y_base[train_idx].to(device)
                X_test = X_base[test_idx].to(device)
                y_test = y_base[test_idx].to(device)

            # Standardize from train only.
            mu = X_train.mean(dim=0, keepdim=True)
            sd = X_train.std(dim=0, keepdim=True)
            sd = torch.where(sd <= 1e-6, torch.ones_like(sd), sd)

            Xtr = (X_train - mu) / sd
            Xte = (X_test - mu) / sd

            ones_tr = torch.ones((Xtr.shape[0], 1), device=device, dtype=torch.float32)
            ones_te = torch.ones((Xte.shape[0], 1), device=device, dtype=torch.float32)
            Xtr_aug = torch.cat([ones_tr, Xtr.float()], dim=1)
            Xte_aug = torch.cat([ones_te, Xte.float()], dim=1)

            for seed in seeds:
                g = torch.Generator(device=device)
                g.manual_seed(int(seed) + abs(hash(fold)) % 100000)
                bidx = torch.randint(0, Xtr_aug.shape[0], (Xtr_aug.shape[0],), device=device, generator=g)
                Xb = Xtr_aug[bidx]
                yb = y_train[bidx].float()

                with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=torch.bfloat16, enabled=use_cuda):
                    XtX = Xb.T @ Xb
                    Xty = Xb.T @ yb
                I = torch.eye(XtX.shape[0], device=device, dtype=torch.float32)
                beta = torch.linalg.solve(XtX.float() + lam * I, Xty.float())

                yhat_tr = pred(Xtr_aug.float(), beta)
                yhat_te = pred(Xte_aug.float(), beta)

                resid = (y_train.float() - yhat_tr.float())
                sigma = torch.std(resid).clamp_min(1e-6)
                z = (y_test.float() - yhat_te.float()) / sigma

                nll = 0.5 * torch.log(2.0 * torch.tensor(math.pi, device=device) * sigma**2) + 0.5 * ((y_test.float() - yhat_te.float()) / sigma) ** 2
                nll_mean = float(torch.mean(nll).detach().cpu())
                calib = float(torch.std(z).detach().cpu())
                z_mean = float(torch.mean(z).detach().cpu())

                rows.append(
                    {
                        "dataset_fold": fold,
                        "seed": int(seed),
                        "n_train": int(Xtr_aug.shape[0]),
                        "n_test": int(Xte_aug.shape[0]),
                        "nll": nll_mean,
                        "calibration_std": calib,
                        "z_mean": z_mean,
                    }
                )

        if not rows:
            raise RuntimeError("no LODO seed/fold rows generated")

        met_df = pd.DataFrame(rows)
        met_df.to_csv(out_dir / "lodo_seed_fold_metrics.csv", index=False)

        fold_summary = (
            met_df.groupby("dataset_fold", as_index=False)
            .agg(
                n_seeds=("seed", "nunique"),
                nll_mean=("nll", "mean"),
                nll_median=("nll", "median"),
                calib_mean=("calibration_std", "mean"),
                calib_median=("calibration_std", "median"),
                z_mean_std=("z_mean", "std"),
            )
        )
        fold_summary.to_csv(out_dir / "lodo_fold_summary.csv", index=False)

        # Figure: NLL by fold.
        fig_nll = out_dir / "FIG_lodo_nll_by_fold.png"
        try:
            fig, ax = plt.subplots(figsize=(8.0, 4.6))
            order = fold_summary["dataset_fold"].astype(str).tolist()
            vals = pd.to_numeric(fold_summary["nll_median"], errors="coerce").to_numpy(dtype=float)
            ax.bar(np.arange(len(vals)), vals, color="#2a6f97")
            ax.set_xticks(np.arange(len(vals)), order, rotation=25, ha="right")
            ax.set_ylabel("Median NLL")
            ax.set_title("LODO cross-dataset NLL")
            ax.grid(alpha=0.2, axis="y")
            fig.tight_layout()
            fig.savefig(fig_nll, dpi=160)
            plt.close(fig)
        except Exception:
            pass

        ended = time.time()
        gpu_stage = _gpu_stage_util_summary(ctx.out_root / "gpu_util.csv", started, ended)

        summary = {
            "status": "PASS",
            "n_trials_total": int(len(all_df)),
            "datasets_used": sorted(set(all_df["dataset_id"].astype(str).tolist())),
            "n_folds": int(fold_summary.shape[0]),
            "n_seeds_requested": int(len(seeds)),
            "n_seeds_completed": int(met_df["seed"].nunique()),
            "compile_ok": bool(compile_ok),
            "compile_reason": compile_reason,
            "preload_gpu_tensors": bool(preload_gpu),
            "gpu_stage_util": gpu_stage,
        }
        _write_json(out_dir / "normative_lodo_summary.json", summary)

        outputs = [
            out_dir / "lodo_seed_fold_metrics.csv",
            out_dir / "lodo_fold_summary.csv",
            out_dir / "normative_lodo_summary.json",
            fig_nll,
        ]

        return _record_stage(
            ctx,
            stage=stage,
            status="PASS",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="GPU ridge LODO many-seed (bf16+compile fallback)",
            outputs=outputs,
            extra=summary,
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
            command="normative_lodo_manyseed",
            error=str(exc),
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

    y = fit[score_col].to_numpy(dtype=float)
    g = fit[group_col].to_numpy(dtype=float)
    a = fit["age"].to_numpy(dtype=float)
    s = fit["sex_num"].to_numpy(dtype=float)
    return _rlm_beta_from_arrays(y, g, a, s), int(len(fit))


def _rlm_beta_from_arrays(y: np.ndarray, g: np.ndarray, a: np.ndarray, s: np.ndarray) -> float:
    import statsmodels.api as sm

    yv = np.asarray(y, dtype=float)
    gv = np.asarray(g, dtype=float)
    av = np.asarray(a, dtype=float)
    sv = np.asarray(s, dtype=float)
    m = np.isfinite(yv) & np.isfinite(gv) & np.isfinite(av) & np.isfinite(sv)
    yv = yv[m]
    gv = gv[m]
    av = av[m]
    sv = sv[m]
    if yv.size < 8 or np.unique(gv).size < 2:
        return float("nan")
    X = np.column_stack(
        [
            np.ones(yv.shape[0], dtype=float),
            gv,
            av,
            sv,
        ]
    )
    try:
        model = sm.RLM(yv, X, M=sm.robust.norms.HuberT())
        res = model.fit()
        params = np.asarray(res.params, dtype=float)
        return float(params[1]) if params.size > 1 else float("nan")
    except Exception:
        return float("nan")


def _perm_group_beta_worker(payload: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]) -> np.ndarray:
    yv, gv, av, sv, n_iter, seed = payload
    rng = np.random.default_rng(int(seed))
    out = np.full(int(n_iter), np.nan, dtype=float)
    for i in range(int(n_iter)):
        perm_g = gv.copy()
        rng.shuffle(perm_g)
        out[i] = _rlm_beta_from_arrays(yv, perm_g, av, sv)
    return out


def _perm_p_group_beta(
    df: pd.DataFrame,
    score_col: str,
    group_col: str,
    *,
    n_perm: int,
    seed: int,
    n_jobs: int = 1,
) -> float:
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
        return float("nan")

    yv = fit[score_col].to_numpy(dtype=float)
    gv = fit[group_col].to_numpy(dtype=float)
    av = fit["age"].to_numpy(dtype=float)
    sv = fit["sex_num"].to_numpy(dtype=float)

    obs = _rlm_beta_from_arrays(yv, gv, av, sv)
    if not np.isfinite(obs):
        return float("nan")

    n_perm_i = int(max(1, n_perm))
    n_jobs_i = int(max(1, min(n_jobs, n_perm_i)))
    if n_jobs_i <= 1:
        null = _perm_group_beta_worker((yv, gv, av, sv, n_perm_i, int(seed)))
    else:
        chunk = int(math.ceil(n_perm_i / float(n_jobs_i)))
        jobs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]] = []
        for j in range(n_jobs_i):
            s = int(j * chunk)
            e = int(min(n_perm_i, s + chunk))
            if s >= e:
                continue
            jobs.append((yv, gv, av, sv, int(e - s), int(seed + 9973 * (j + 1))))
        null_parts: List[np.ndarray] = []
        ctx_mp = mp.get_context("fork")
        with cf.ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx_mp) as ex:
            for arr in ex.map(_perm_group_beta_worker, jobs):
                null_parts.append(np.asarray(arr, dtype=float))
        null = np.concatenate(null_parts, axis=0) if null_parts else np.asarray([], dtype=float)

    finite = null[np.isfinite(null)]
    if finite.size == 0:
        return float("nan")
    return float((1.0 + np.sum(np.abs(finite) >= abs(obs))) / (1.0 + finite.size))


def _perm_rlm_coef_worker(payload: Tuple[np.ndarray, np.ndarray, int, int]) -> np.ndarray:
    yv, Xv, n_iter, seed = payload
    import statsmodels.api as sm

    rng = np.random.default_rng(int(seed))
    out = np.full(int(n_iter), np.nan, dtype=float)
    for i in range(int(n_iter)):
        yp = np.asarray(yv, dtype=float).copy()
        rng.shuffle(yp)
        try:
            rnull = sm.RLM(yp, Xv, M=sm.robust.norms.HuberT()).fit()
            params = np.asarray(rnull.params, dtype=float)
            out[i] = float(params[1]) if params.size > 1 else float("nan")
        except Exception:
            continue
    return out


def _perm_p_rlm_coef(
    y: np.ndarray,
    X: np.ndarray,
    obs_beta: float,
    *,
    n_perm: int,
    seed: int,
    n_jobs: int = 1,
) -> Tuple[float, np.ndarray]:
    yv = np.asarray(y, dtype=float)
    Xv = np.asarray(X, dtype=float)
    if not np.isfinite(obs_beta):
        return float("nan"), np.asarray([], dtype=float)

    n_perm_i = int(max(1, n_perm))
    n_jobs_i = int(max(1, min(n_jobs, n_perm_i)))
    if n_jobs_i <= 1:
        null = _perm_rlm_coef_worker((yv, Xv, n_perm_i, int(seed)))
    else:
        chunk = int(math.ceil(n_perm_i / float(n_jobs_i)))
        jobs: List[Tuple[np.ndarray, np.ndarray, int, int]] = []
        for j in range(n_jobs_i):
            s = int(j * chunk)
            e = int(min(n_perm_i, s + chunk))
            if s >= e:
                continue
            jobs.append((yv, Xv, int(e - s), int(seed + 8191 * (j + 1))))
        null_parts: List[np.ndarray] = []
        ctx_mp = mp.get_context("fork")
        with cf.ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx_mp) as ex:
            for arr in ex.map(_perm_rlm_coef_worker, jobs):
                null_parts.append(np.asarray(arr, dtype=float))
        null = np.concatenate(null_parts, axis=0) if null_parts else np.asarray([], dtype=float)

    finite = null[np.isfinite(null)]
    if finite.size == 0:
        return float("nan"), finite
    p = float((1.0 + np.sum(np.abs(finite) >= abs(obs_beta))) / (1.0 + finite.size))
    return p, finite


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


def _extract_rest_features(ds_root: Path) -> Tuple[pd.DataFrame, List[str]]:
    eeg_files = []
    for p in ds_root.rglob("*"):
        if not p.is_file():
            continue
        if "derivatives" in p.parts:
            continue
        if _EEG_PAT.search(p.name):
            eeg_files.append(p)
    eeg_files = sorted(set(eeg_files))

    subj_to_file: Dict[str, Path] = {}
    for p in eeg_files:
        m = re.search(r"sub-([A-Za-z0-9]+)", str(p))
        if not m:
            continue
        sid = m.group(1)
        if sid not in subj_to_file:
            subj_to_file[sid] = p

    rows: List[Dict[str, Any]] = []
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
                return float(np.trapezoid(pxx[m], f[m]))

            theta = _band(4.0, 8.0)
            alpha = _band(8.0, 12.0)
            total = _band(1.0, 30.0)

            msl = (f >= 2.0) & (f <= 30.0) & np.isfinite(pxx) & (pxx > 0)
            if int(msl.sum()) >= 6:
                x = np.log10(f[msl])
                y = np.log10(pxx[msl])
                slope = float(np.polyfit(x, y, deg=1)[0])
            else:
                slope = float("nan")

            rows.append(
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

    return pd.DataFrame(rows), failures


def _compute_rest_deviation(df: pd.DataFrame, *, control_mask: Optional[pd.Series]) -> pd.DataFrame:
    import statsmodels.api as sm

    dev_df = df.copy()
    feature_cols = ["theta_alpha_ratio", "rel_alpha", "spectral_slope"]
    dev_df["age"] = pd.to_numeric(dev_df.get("age"), errors="coerce")
    dev_df["sex_num"] = _encode_sex(dev_df.get("sex", pd.Series([""] * len(dev_df))))

    for col in feature_cols:
        if control_mask is None:
            ctrl = dev_df[np.isfinite(pd.to_numeric(dev_df[col], errors="coerce"))].copy()
        else:
            ctrl = dev_df[control_mask & np.isfinite(pd.to_numeric(dev_df[col], errors="coerce"))].copy()

        if ctrl.empty:
            dev_df[f"pred_{col}"] = np.nan
            dev_df[f"dev_z_{col}"] = np.nan
            continue

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

    # Oriented composite.
    oriented_cols = []
    for col in feature_cols:
        zc = f"dev_z_{col}"
        if zc not in dev_df.columns:
            continue
        high_med = np.nanmedian(pd.to_numeric(dev_df[col], errors="coerce").to_numpy(dtype=float))
        sign = 1.0 if np.isfinite(high_med) else 1.0
        oc = f"oriented_{zc}"
        dev_df[oc] = sign * pd.to_numeric(dev_df[zc], errors="coerce")
        oriented_cols.append(oc)

    dev_df["composite_deviation"] = dev_df[oriented_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) if oriented_cols else np.nan
    return dev_df


def _infer_pd_group(part: pd.DataFrame, participants_json: Optional[Dict[str, Any]]) -> Tuple[Optional[pd.Series], str]:
    cols = [c for c in ["Group", "group", "Diagnosis", "diagnosis", "dx", "condition", "status"] if c in part.columns]
    level_map: Dict[str, str] = {}

    if participants_json and isinstance(participants_json, dict):
        for c in cols:
            lv = (((participants_json.get(c) or {}).get("Levels")) or {}) if isinstance(participants_json.get(c), dict) else {}
            if isinstance(lv, dict):
                for k, v in lv.items():
                    vv = str(v).lower()
                    if "parkinson" in vv or vv == "pd":
                        level_map[str(k)] = "PD"
                    elif "control" in vv or "healthy" in vv or "ctl" in vv:
                        level_map[str(k)] = "CN"

    for c in cols:
        s = part[c].astype(str).str.strip()
        mapped = pd.Series(["UNK"] * len(s), index=s.index)
        low = s.str.lower()

        mapped[low.str.contains("parkinson|\bpd\b", na=False, regex=True)] = "PD"
        mapped[low.str.contains("control|healthy|\bhc\b|\bctl\b", na=False, regex=True)] = "CN"

        if level_map:
            mapped2 = s.map(level_map)
            mapped = mapped.where(mapped != "UNK", mapped2.fillna("UNK"))

        n_pd = int((mapped == "PD").sum())
        n_cn = int((mapped == "CN").sum())
        if n_pd >= 10 and n_cn >= 10:
            return mapped, c

    return None, ""


def _infer_mortality_label(part: pd.DataFrame, participants_json: Optional[Dict[str, Any]]) -> Tuple[Optional[pd.Series], str, Dict[str, Any]]:
    cols = list(part.columns)
    cand_cols = [c for c in cols if re.search(r"death|mort|vital|deceas|surviv|status", c, flags=re.IGNORECASE)]

    diag: Dict[str, Any] = {"candidate_columns": cand_cols}

    for c in cand_cols:
        s = part[c]
        y = pd.Series(np.nan, index=s.index, dtype=float)

        # Numeric first.
        num = pd.to_numeric(s, errors="coerce")
        if np.isfinite(num).any():
            uniq = sorted(set(num.dropna().astype(int).tolist()))
            if set(uniq).issubset({0, 1}) and len(uniq) >= 2:
                y = num.astype(float)

        if y.isna().all():
            low = s.astype(str).str.lower().str.strip()
            y[low.str.contains("dead|deceased|death|died|mort", na=False, regex=True)] = 1.0
            y[low.str.contains("alive|censor|surviv", na=False, regex=True)] = 0.0

        n0 = int((y == 0).sum())
        n1 = int((y == 1).sum())
        if n0 >= 5 and n1 >= 5:
            return y, c, {"n_alive": n0, "n_dead": n1}

    return None, "", diag


def _stage_clinical_translation(ctx: RunContext) -> Dict[str, Any]:
    stage = "clinical_translation"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    ana = ctx.cfg_mega.get("analysis", {}) or {}
    n_perm = int(max(int(ana.get("clinical_perm_min", 5000)), int(ana.get("clinical_perm", 20000))))
    perm_jobs = int(max(1, min(ctx.cpu_workers, int(ana.get("clinical_perm_workers", ctx.cpu_workers)), 64)))

    outputs: List[Path] = []
    endpoint_rows: List[Dict[str, Any]] = []
    status_by_dataset: Dict[str, str] = {}
    reasons_by_dataset: Dict[str, str] = {}

    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"[{_iso_now()}] clinical_translation start n_perm={n_perm} perm_jobs={perm_jobs} cpu_workers={ctx.cpu_workers}\n")

    # -------- ds004504 dementia --------
    ds = "ds004504"
    ds_out = ctx.pack_clinical / ds
    ds_out.mkdir(parents=True, exist_ok=True)
    ds_root = ctx.data_root / ds
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"[{_iso_now()}] dataset_start {ds}\n")

    if not ds_root.exists() or not (ds_root / "participants.tsv").exists():
        reason = "dataset or participants.tsv missing"
        _write_stop_reason(ds_out / "STOP_REASON.md", f"{stage}:{ds}", reason)
        status_by_dataset[ds] = "SKIP"
        reasons_by_dataset[ds] = reason
    else:
        part = pd.read_csv(ds_root / "participants.tsv", sep="\t")
        pjson = None
        if (ds_root / "participants.json").exists():
            try:
                pjson = json.loads((ds_root / "participants.json").read_text(encoding="utf-8"))
            except Exception:
                pjson = None

        part["subject_id"] = part["participant_id"].map(_safe_subject)
        group_map = {"A": "AD", "F": "FTD", "C": "CN"}
        if pjson and isinstance(pjson, dict):
            lv = (((pjson.get("Group") or {}).get("Levels")) or {}) if isinstance(pjson.get("Group"), dict) else {}
            if isinstance(lv, dict):
                for k, v in lv.items():
                    vv = str(v).lower()
                    if "alzheimer" in vv:
                        group_map[str(k)] = "AD"
                    elif "frontotemporal" in vv or "ftd" in vv:
                        group_map[str(k)] = "FTD"
                    elif "healthy" in vv or "control" in vv:
                        group_map[str(k)] = "CN"

        part["group"] = part.get("Group", "").astype(str).map(group_map).fillna("UNK")
        part["age"] = pd.to_numeric(part.get("Age"), errors="coerce")
        part["sex"] = part.get("Gender", "").astype(str)
        part["mmse"] = pd.to_numeric(part.get("MMSE"), errors="coerce")

        feat_df, fails = _extract_rest_features(ds_root)
        if feat_df.empty:
            reason = "no usable resting EEG features"
            _write_stop_reason(ds_out / "STOP_REASON.md", f"{stage}:{ds}", reason, diagnostics={"n_failures": len(fails), "sample_failures": fails[:20]})
            status_by_dataset[ds] = "SKIP"
            reasons_by_dataset[ds] = reason
        else:
            feat_df = feat_df.merge(part[["subject_id", "group", "age", "sex", "mmse"]], on="subject_id", how="left")
            dev_df = _compute_rest_deviation(feat_df, control_mask=(feat_df["group"] == "CN"))
            feat_df.to_csv(ds_out / "spectral_subject_features.csv", index=False)
            dev_df.to_csv(ds_out / "normative_deviation_scores.csv", index=False)
            outputs.extend([ds_out / "spectral_subject_features.csv", ds_out / "normative_deviation_scores.csv"])

            endpoint_features = ["dev_z_theta_alpha_ratio", "dev_z_rel_alpha", "dev_z_spectral_slope", "composite_deviation"]
            rows_ds: List[Dict[str, Any]] = []

            for feat in endpoint_features:
                if feat not in dev_df.columns:
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
                    p = _perm_p_auc(y, s, n_perm=n_perm, seed=300 + seed_off)
                    return {
                        "dataset_id": ds,
                        "endpoint": label,
                        "feature": feat,
                        "type": "auc",
                        "n": int(len(sub)),
                        "estimate": float(auc),
                        "auc_raw": float(auc),
                        "auc_flipped": float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan"),
                        "ci95_lo": float(ci[0]),
                        "ci95_hi": float(ci[1]),
                        "perm_p": float(p),
                    }

                for tup in [("AUC_AD_vs_CN", "AD", "CN", 1), ("AUC_FTD_vs_CN", "FTD", "CN", 2), ("AUC_AD_vs_FTD", "AD", "FTD", 3)]:
                    rr = _auc_row(*tup)
                    if rr:
                        rows_ds.append(rr)

                # MMSE regression
                import statsmodels.api as sm

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
                    x_col_idx = list(X.columns).index("x") if "x" in X.columns else -1
                    y_arr = y.to_numpy(dtype=float)
                    X_arr = X.to_numpy(dtype=float)
                    try:
                        res = sm.RLM(y_arr, X_arr, M=sm.robust.norms.HuberT()).fit()
                        params = np.asarray(res.params, dtype=float)
                        beta = float(params[x_col_idx]) if x_col_idx >= 0 and params.size > x_col_idx else float("nan")
                    except Exception:
                        beta = float("nan")

                    finite = np.asarray([], dtype=float)
                    if np.isfinite(beta):
                        p_mmse, finite = _perm_p_rlm_coef(
                            y_arr,
                            X_arr,
                            beta,
                            n_perm=n_perm,
                            seed=411 + _stable_int_from_str(feat) % 100000,
                            n_jobs=perm_jobs,
                        )
                    else:
                        p_mmse = float("nan")

                    rows_ds.append(
                        {
                            "dataset_id": ds,
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

            end_df = pd.DataFrame(rows_ds)
            if not end_df.empty:
                end_df["perm_q"] = _bh_qvals(pd.to_numeric(end_df["perm_p"], errors="coerce").fillna(1.0).to_numpy(dtype=float).tolist())
                if not np.all(np.isfinite(pd.to_numeric(end_df["perm_q"], errors="coerce").to_numpy(dtype=float))):
                    return _record_stage(
                        ctx,
                        stage=stage,
                        status="FAIL",
                        rc=1,
                        started=started,
                        log_path=log_path,
                        summary_path=summary_path,
                        command="clinical translation",
                        error="NaN q-values in dementia endpoints",
                    )
            end_df.to_csv(ds_out / "dementia_endpoints.csv", index=False)
            outputs.append(ds_out / "dementia_endpoints.csv")

            incl = {
                "n_subjects_total_features": int(dev_df["subject_id"].nunique()),
                "n_group_CN": int((dev_df["group"] == "CN").sum()),
                "n_group_AD": int((dev_df["group"] == "AD").sum()),
                "n_group_FTD": int((dev_df["group"] == "FTD").sum()),
                "n_mmse_nonmissing": int(np.isfinite(pd.to_numeric(dev_df["mmse"], errors="coerce")).sum()),
                "n_failures_read": int(len(fails)),
            }
            _write_json(ds_out / "inclusion_exclusion_summary.json", incl)
            outputs.append(ds_out / "inclusion_exclusion_summary.json")

            status_by_dataset[ds] = "PASS" if not end_df.empty else "SKIP"
            reasons_by_dataset[ds] = "" if not end_df.empty else "no endpoints"
            endpoint_rows.extend(end_df.to_dict(orient="records") if not end_df.empty else [])
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(
                    f"[{_iso_now()}] dataset_done {ds} status={status_by_dataset[ds]} endpoints={len(end_df)} "
                    f"subjects={incl.get('n_subjects_total_features', 0)}\n"
                )

    # -------- ds004584 PD rest --------
    ds = "ds004584"
    ds_out = ctx.pack_clinical / ds
    ds_out.mkdir(parents=True, exist_ok=True)
    ds_root = ctx.data_root / ds
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"[{_iso_now()}] dataset_start {ds}\n")

    if not ds_root.exists() or not (ds_root / "participants.tsv").exists():
        reason = "dataset or participants.tsv missing"
        _write_stop_reason(ds_out / "STOP_REASON.md", f"{stage}:{ds}", reason)
        status_by_dataset[ds] = "SKIP"
        reasons_by_dataset[ds] = reason
    else:
        part = pd.read_csv(ds_root / "participants.tsv", sep="\t")
        part["subject_id"] = part["participant_id"].map(_safe_subject)
        pjson = None
        if (ds_root / "participants.json").exists():
            try:
                pjson = json.loads((ds_root / "participants.json").read_text(encoding="utf-8"))
            except Exception:
                pjson = None

        labels, label_col = _infer_pd_group(part, pjson)
        if labels is None:
            reason = "could not infer PD/control labels from participants"
            _write_stop_reason(ds_out / "STOP_REASON.md", f"{stage}:{ds}", reason, diagnostics={"columns": list(part.columns)})
            status_by_dataset[ds] = "SKIP"
            reasons_by_dataset[ds] = reason
        else:
            part["group"] = labels
            part["age"] = pd.to_numeric(part.get("Age", part.get("age")), errors="coerce")
            part["sex"] = part.get("Sex", part.get("sex", part.get("Gender", ""))).astype(str)

            feat_df, fails = _extract_rest_features(ds_root)
            if feat_df.empty:
                reason = "no usable resting EEG features"
                _write_stop_reason(ds_out / "STOP_REASON.md", f"{stage}:{ds}", reason, diagnostics={"n_failures": len(fails), "sample_failures": fails[:20]})
                status_by_dataset[ds] = "SKIP"
                reasons_by_dataset[ds] = reason
            else:
                feat_df = feat_df.merge(part[["subject_id", "group", "age", "sex"]], on="subject_id", how="left")
                ctrl_mask = feat_df["group"] == "CN"
                dev_df = _compute_rest_deviation(feat_df, control_mask=ctrl_mask)

                feat_df.to_csv(ds_out / "spectral_subject_features.csv", index=False)
                dev_df.to_csv(ds_out / "normative_deviation_scores.csv", index=False)
                outputs.extend([ds_out / "spectral_subject_features.csv", ds_out / "normative_deviation_scores.csv"])

                rows_ds: List[Dict[str, Any]] = []
                for feat in ["dev_z_theta_alpha_ratio", "dev_z_rel_alpha", "dev_z_spectral_slope", "composite_deviation"]:
                    if feat not in dev_df.columns:
                        continue
                    sub = dev_df[dev_df["group"].isin(["PD", "CN"])].copy()
                    sub["x"] = pd.to_numeric(sub[feat], errors="coerce")
                    sub = sub[np.isfinite(sub["x"])].copy()
                    if sub.empty:
                        continue
                    sub["grp"] = (sub["group"].astype(str) == "PD").astype(int)
                    y = sub["grp"].to_numpy(dtype=int)
                    x = sub["x"].to_numpy(dtype=float)
                    if len(np.unique(y)) < 2:
                        continue

                    auc, ci = _bootstrap_auc(y, x, n_boot=2000, seed=901)
                    p_auc = _perm_p_auc(y, x, n_perm=n_perm, seed=902)

                    rows_ds.append(
                        {
                            "dataset_id": ds,
                            "endpoint": "AUC_PD_vs_CN",
                            "feature": feat,
                            "type": "auc",
                            "n": int(len(sub)),
                            "estimate": float(auc),
                            "auc_raw": float(auc),
                            "auc_flipped": float(max(auc, 1.0 - auc)),
                            "ci95_lo": float(ci[0]),
                            "ci95_hi": float(ci[1]),
                            "perm_p": float(p_auc),
                        }
                    )

                    beta, nfit = _robust_group_beta(sub.rename(columns={"x": "score", "grp": "group_bin"}), "score", "group_bin")
                    p_beta = _perm_p_group_beta(
                        sub.rename(columns={"x": "score", "grp": "group_bin"}),
                        "score",
                        "group_bin",
                        n_perm=n_perm,
                        seed=903 + _stable_int_from_str(feat) % 100000,
                        n_jobs=perm_jobs,
                    )
                    rows_ds.append(
                        {
                            "dataset_id": ds,
                            "endpoint": "RobustBeta_PD_vs_CN",
                            "feature": feat,
                            "type": "robust_beta",
                            "n": int(nfit),
                            "estimate": float(beta),
                            "auc_raw": float("nan"),
                            "auc_flipped": float("nan"),
                            "ci95_lo": float("nan"),
                            "ci95_hi": float("nan"),
                            "perm_p": float(p_beta),
                        }
                    )

                end_df = pd.DataFrame(rows_ds)
                if not end_df.empty:
                    end_df["perm_q"] = _bh_qvals(pd.to_numeric(end_df["perm_p"], errors="coerce").fillna(1.0).to_numpy(dtype=float).tolist())
                end_df.to_csv(ds_out / "pd_rest_endpoints.csv", index=False)
                outputs.append(ds_out / "pd_rest_endpoints.csv")

                status_by_dataset[ds] = "PASS" if not end_df.empty else "SKIP"
                reasons_by_dataset[ds] = "" if not end_df.empty else "no endpoints"
                endpoint_rows.extend(end_df.to_dict(orient="records") if not end_df.empty else [])
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(
                        f"[{_iso_now()}] dataset_done {ds} status={status_by_dataset[ds]} "
                        f"endpoints={len(end_df)} subjects={int(dev_df['subject_id'].nunique())}\n"
                    )

    # -------- ds007020 PD mortality --------
    ds = "ds007020"
    ds_out = ctx.pack_clinical / ds
    ds_out.mkdir(parents=True, exist_ok=True)
    ds_root = ctx.data_root / ds
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"[{_iso_now()}] dataset_start {ds}\n")

    if not ds_root.exists() or not (ds_root / "participants.tsv").exists():
        reason = "dataset or participants.tsv missing"
        _write_stop_reason(ds_out / "STOP_REASON.md", f"{stage}:{ds}", reason)
        status_by_dataset[ds] = "SKIP"
        reasons_by_dataset[ds] = reason
    else:
        part = pd.read_csv(ds_root / "participants.tsv", sep="\t")
        part["subject_id"] = part["participant_id"].map(_safe_subject)
        pjson = None
        if (ds_root / "participants.json").exists():
            try:
                pjson = json.loads((ds_root / "participants.json").read_text(encoding="utf-8"))
            except Exception:
                pjson = None

        y, y_col, y_diag = _infer_mortality_label(part, pjson)
        if y is None:
            reason = "mortality labels not found in participants"
            _write_stop_reason(ds_out / "STOP_REASON.md", f"{stage}:{ds}", reason, diagnostics={"columns": list(part.columns), "label_probe": y_diag})
            status_by_dataset[ds] = "SKIP"
            reasons_by_dataset[ds] = reason
        else:
            part["mortality_label"] = y
            part["age"] = pd.to_numeric(part.get("Age", part.get("age")), errors="coerce")
            part["sex"] = part.get("Sex", part.get("sex", part.get("Gender", ""))).astype(str)

            feat_df, fails = _extract_rest_features(ds_root)
            if feat_df.empty:
                reason = "no usable resting EEG features"
                _write_stop_reason(ds_out / "STOP_REASON.md", f"{stage}:{ds}", reason, diagnostics={"n_failures": len(fails), "sample_failures": fails[:20]})
                status_by_dataset[ds] = "SKIP"
                reasons_by_dataset[ds] = reason
            else:
                feat_df = feat_df.merge(part[["subject_id", "mortality_label", "age", "sex"]], on="subject_id", how="left")
                # No explicit controls assumed; robust z within dataset.
                dev_df = _compute_rest_deviation(feat_df, control_mask=None)
                feat_df.to_csv(ds_out / "spectral_subject_features.csv", index=False)
                dev_df.to_csv(ds_out / "normative_deviation_scores.csv", index=False)
                outputs.extend([ds_out / "spectral_subject_features.csv", ds_out / "normative_deviation_scores.csv"])

                rows_ds: List[Dict[str, Any]] = []
                for feat in ["dev_z_theta_alpha_ratio", "dev_z_rel_alpha", "dev_z_spectral_slope", "composite_deviation"]:
                    if feat not in dev_df.columns:
                        continue
                    sub = dev_df.copy()
                    sub["x"] = pd.to_numeric(sub[feat], errors="coerce")
                    sub["grp"] = pd.to_numeric(sub["mortality_label"], errors="coerce")
                    sub = sub[np.isfinite(sub["x"]) & np.isfinite(sub["grp"])].copy()
                    if sub.empty or sub["grp"].nunique() < 2:
                        continue
                    yb = sub["grp"].to_numpy(dtype=int)
                    xb = sub["x"].to_numpy(dtype=float)
                    auc, ci = _bootstrap_auc(yb, xb, n_boot=2000, seed=1901)
                    p_auc = _perm_p_auc(yb, xb, n_perm=n_perm, seed=1902)

                    rows_ds.append(
                        {
                            "dataset_id": ds,
                            "endpoint": "AUC_mortality",
                            "feature": feat,
                            "type": "auc",
                            "n": int(len(sub)),
                            "estimate": float(auc),
                            "auc_raw": float(auc),
                            "auc_flipped": float(max(auc, 1.0 - auc)),
                            "ci95_lo": float(ci[0]),
                            "ci95_hi": float(ci[1]),
                            "perm_p": float(p_auc),
                        }
                    )

                    beta, nfit = _robust_group_beta(sub.rename(columns={"x": "score", "grp": "group_bin"}), "score", "group_bin")
                    p_beta = _perm_p_group_beta(
                        sub.rename(columns={"x": "score", "grp": "group_bin"}),
                        "score",
                        "group_bin",
                        n_perm=n_perm,
                        seed=1903 + _stable_int_from_str(feat) % 100000,
                        n_jobs=perm_jobs,
                    )
                    rows_ds.append(
                        {
                            "dataset_id": ds,
                            "endpoint": "RobustBeta_mortality",
                            "feature": feat,
                            "type": "robust_beta",
                            "n": int(nfit),
                            "estimate": float(beta),
                            "auc_raw": float("nan"),
                            "auc_flipped": float("nan"),
                            "ci95_lo": float("nan"),
                            "ci95_hi": float("nan"),
                            "perm_p": float(p_beta),
                        }
                    )

                end_df = pd.DataFrame(rows_ds)
                if not end_df.empty:
                    end_df["perm_q"] = _bh_qvals(pd.to_numeric(end_df["perm_p"], errors="coerce").fillna(1.0).to_numpy(dtype=float).tolist())
                end_df.to_csv(ds_out / "mortality_endpoints.csv", index=False)
                outputs.append(ds_out / "mortality_endpoints.csv")

                status_by_dataset[ds] = "PASS" if not end_df.empty else "SKIP"
                reasons_by_dataset[ds] = "" if not end_df.empty else "no endpoints"
                endpoint_rows.extend(end_df.to_dict(orient="records") if not end_df.empty else [])
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(
                        f"[{_iso_now()}] dataset_done {ds} status={status_by_dataset[ds]} "
                        f"endpoints={len(end_df)} subjects={int(dev_df['subject_id'].nunique())}\n"
                    )

    # Global clinical endpoint table + BH-FDR across all rows.
    all_end = pd.DataFrame(endpoint_rows)
    if not all_end.empty:
        all_end["perm_q_global"] = _bh_qvals(pd.to_numeric(all_end["perm_p"], errors="coerce").fillna(1.0).to_numpy(dtype=float).tolist())
        if not np.all(np.isfinite(pd.to_numeric(all_end["perm_q_global"], errors="coerce").to_numpy(dtype=float))):
            return _record_stage(
                ctx,
                stage=stage,
                status="FAIL",
                rc=1,
                started=started,
                log_path=log_path,
                summary_path=summary_path,
                command="clinical translation",
                outputs=outputs,
                error="NaN q-values in global clinical endpoints",
            )
    all_end.to_csv(ctx.pack_clinical / "clinical_endpoints_all.csv", index=False)
    outputs.append(ctx.pack_clinical / "clinical_endpoints_all.csv")

    # Stage status logic.
    hard_fail = []
    if status_by_dataset.get("ds004504") != "PASS":
        hard_fail.append("ds004504 must PASS for dementia clinical translation")

    status = "FAIL" if hard_fail else "PASS"
    err = " ; ".join(hard_fail) if hard_fail else ""

    # Optional datasets can SKIP but should mark partial.
    for ds_opt in ["ds004584", "ds007020"]:
        if status_by_dataset.get(ds_opt) == "SKIP":
            ctx.partial_reasons.append(f"{ds_opt} skipped: {reasons_by_dataset.get(ds_opt, '')}")

    _write_json(ctx.pack_clinical / "clinical_status_by_dataset.json", {"status_by_dataset": status_by_dataset, "reasons_by_dataset": reasons_by_dataset})
    outputs.append(ctx.pack_clinical / "clinical_status_by_dataset.json")

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=1 if status == "FAIL" else 0,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="clinical translation ds004504+ds004584+ds007020",
        outputs=outputs,
        error=err,
        extra={"status_by_dataset": status_by_dataset, "reasons_by_dataset": reasons_by_dataset},
    )


def _summarize_nvidia_smi_csv(path: Path) -> Dict[str, Any]:
    out = {
        "rows": 0,
        "util_gpu_mean": float("nan"),
        "util_gpu_median": float("nan"),
        "util_mem_mean": float("nan"),
        "util_mem_median": float("nan"),
        "mem_used_mb_mean": float("nan"),
        "mem_used_mb_median": float("nan"),
        "power_w_mean": float("nan"),
        "power_w_median": float("nan"),
    }
    if not path.exists() or path.stat().st_size == 0:
        return out
    try:
        df = pd.read_csv(path, header=None)
    except Exception:
        return out
    if df.empty or df.shape[1] < 8:
        return out
    out["rows"] = int(len(df))

    col_gpu = pd.to_numeric(df.iloc[:, 2], errors="coerce")
    col_memu = pd.to_numeric(df.iloc[:, 3], errors="coerce")
    col_mem_used = pd.to_numeric(df.iloc[:, 5], errors="coerce")
    col_power = pd.to_numeric(df.iloc[:, 6], errors="coerce")

    for name, s in [
        ("util_gpu", col_gpu),
        ("util_mem", col_memu),
        ("mem_used_mb", col_mem_used),
        ("power_w", col_power),
    ]:
        vals = s.to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out[f"{name}_mean"] = float(np.mean(vals))
        out[f"{name}_median"] = float(np.median(vals))

    return out


def _build_final_report(ctx: RunContext, run_status: str, run_error: str) -> Path:
    report = ctx.audit_dir / "NN_FINAL_MEGA_REPORT.md"

    stage_rows = [
        f"| {r['stage']} | {r['status']} | {r['returncode']} | {r['elapsed_sec']:.1f} | {Path(r['log']).name} | {Path(r['summary']).name} |"
        for r in ctx.stage_records
    ]

    ds_hashes = _read_json_if_exists(ctx.audit_dir / "dataset_hashes.json") or {}
    ds_rows = []
    for d in ds_hashes.get("datasets", []):
        commit = d.get("git_head", None)
        commit_txt = "null" if commit is None else str(commit)
        ds_rows.append(
            f"| {d.get('dataset_id')} | {commit_txt} | {d.get('n_event_files')} | {d.get('n_eeg_files')} | {d.get('status')} | {d.get('reason','')} |"
        )
    if not ds_rows:
        ds_rows = ["| <none> | <none> | <none> | <none> | <none> | <none> |"]

    mapping_summary = _read_json_if_exists(ctx.pack_mapping / "mapping_decode_summary.json") or {}
    map_rows = []
    for r in mapping_summary.get("rows", []):
        map_rows.append(f"| {r.get('dataset_id')} | {r.get('status')} | {r.get('selected_candidate_id','')} | {r.get('reason','')} |")
    if not map_rows:
        map_rows = ["| <none> | <none> | <none> | <none> |"]

    feat_rows = []
    feat_csv = ctx.pack_features / "features_summary_all.csv"
    if feat_csv.exists():
        try:
            ff = pd.read_csv(feat_csv)
            for _, r in ff.iterrows():
                feat_rows.append(f"| {r.get('dataset_id')} | {r.get('status')} | {r.get('method','')} | {r.get('n_h5')} | {r.get('n_trials')} | {r.get('reason','')} |")
        except Exception:
            pass
    if not feat_rows:
        feat_rows = ["| <none> | <none> | <none> | <none> | <none> | <none> |"]

    # Core lawc
    lawc_rows = ["| <none> | <none> | <none> | <none> | <none> | <none> |"]
    lawc_json = ctx.pack_core / "lawc_ultradeep" / "lawc_audit" / "locked_test_results.json"
    if lawc_json.exists():
        try:
            payload = json.loads(lawc_json.read_text(encoding="utf-8"))
            rr = payload.get("datasets", [])
            if rr:
                lawc_rows = []
                for r in rr:
                    lawc_rows.append(
                        f"| {r.get('dataset_id')} | {r.get('median_rho')} | {r.get('p_value')} | {r.get('q_value')} | {r.get('x_control_degrade_pass')} | {r.get('y_control_degrade_pass')} |"
                    )
        except Exception:
            pass

    meta = _read_json_if_exists(ctx.pack_core / "lawc_ultradeep" / "lawc_audit" / "meta_random_effects.json") or {}

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

    norm_summary = _read_json_if_exists(ctx.pack_normative / "normative_lodo_summary.json") or {}

    clin_rows = ["| <none> | <none> | <none> | <none> | <none> | <none> | <none> |"]
    clin_csv = ctx.pack_clinical / "clinical_endpoints_all.csv"
    if clin_csv.exists():
        try:
            dd = pd.read_csv(clin_csv)
            if not dd.empty:
                clin_rows = []
                for _, r in dd.iterrows():
                    clin_rows.append(
                        f"| {r.get('dataset_id')} | {r.get('endpoint')} | {r.get('feature')} | {r.get('n')} | {r.get('estimate')} | {r.get('perm_p')} | {r.get('perm_q_global', r.get('perm_q'))} |"
                    )
        except Exception:
            pass

    gpu_nvml = summarize_gpu_util_csv(ctx.out_root / "gpu_util.csv")
    gpu_smi = _summarize_nvidia_smi_csv(ctx.audit_dir / "nvidia_smi_1hz.csv")

    figure_candidates = [
        ctx.pack_core / "effect_sizes" / "FIG_slopes_uv_per_load.png",
        ctx.pack_core / "effect_sizes" / "FIG_delta_uv_high_vs_low.png",
        ctx.pack_core / "effect_sizes" / "FIG_waveforms_by_load.png",
        ctx.pack_mechanism / "FIG_load_vs_pupil.png",
        ctx.pack_mechanism / "FIG_pupil_vs_p3_partial.png",
        ctx.pack_mechanism / "FIG_mediation_ab.png",
        ctx.pack_mechanism / "FIG_mechanism_summary.png",
        ctx.pack_normative / "FIG_lodo_nll_by_fold.png",
    ]
    for p in sorted(ctx.pack_clinical.rglob("FIG_*.png")):
        figure_candidates.append(p)

    figure_lines = [f"- `{p}`" for p in figure_candidates if p.exists()]
    if not figure_lines:
        figure_lines = ["- <none>"]

    lines = [
        "# NN_FINAL_MEGA REPORT",
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
            "| Dataset | Commit | Event files | EEG files | Status | Reason |",
            "|---|---|---:|---:|---|---|",
            *ds_rows,
            "",
            "## Decode mapping",
            "| Dataset | Status | Candidate | Reason |",
            "|---|---|---|---|",
            *map_rows,
            "",
            "## Feature extraction",
            "| Dataset | Status | Method | HDF5 files | Trials | Reason |",
            "|---|---|---|---:|---:|---|",
            *feat_rows,
            "",
            "## Core Law-C locked results",
            "| Dataset | Median rho | p | q | X-control degrade | Y-control degrade |",
            "|---|---:|---:|---:|---|---|",
            *lawc_rows,
            "",
            "## Law-C random-effects meta",
            f"- `{json.dumps(meta, sort_keys=True)}`",
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
            "## Normative LODO",
            f"- `{json.dumps(norm_summary, sort_keys=True)}`",
            "",
            "## Clinical endpoints",
            "| Dataset | Endpoint | Feature | N | Estimate | Perm p | Perm q(global) |",
            "|---|---|---|---:|---:|---:|---:|",
            *clin_rows,
            "",
            "## GPU utilization summary",
            f"- NVML logger (`gpu_util.csv`): `{json.dumps(gpu_nvml, sort_keys=True)}`",
            f"- nvidia-smi 1Hz (`AUDIT/nvidia_smi_1hz.csv`): `{json.dumps(gpu_smi, sort_keys=True)}`",
            "",
            "## Partial reasons",
            *([f"- {x}" for x in ctx.partial_reasons] if ctx.partial_reasons else ["- <none>"]),
            "",
            "## Figure paths",
            *figure_lines,
            "",
            "## Provenance",
            f"- Repo fingerprint: `{ctx.audit_dir / 'repo_fingerprint.json'}`",
            f"- Dataset hashes: `{ctx.audit_dir / 'dataset_hashes.json'}`",
            f"- GPU monitor (nvidia-smi): `{ctx.audit_dir / 'nvidia_smi_1hz.csv'}`",
            f"- GPU monitor (NVML): `{ctx.out_root / 'gpu_util.csv'}`",
            f"- Config default: `{ctx.config}`",
            f"- Mega config: `{ctx.mega_config}`",
            f"- Datasets config: `{ctx.datasets_config}`",
            f"- Final bundle: `{ctx.outzip_dir / 'NN_FINAL_MEGA_SUBMISSION_PACKET.zip'}`",
        ]
    )

    _write_text(report, "\n".join(lines) + "\n")
    return report


def _stage_final_bundle(ctx: RunContext, run_status: str, run_error: str) -> Dict[str, Any]:
    stage = "final_bundle"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    report = _build_final_report(ctx, run_status, run_error)

    zpath = ctx.outzip_dir / "NN_FINAL_MEGA_SUBMISSION_PACKET.zip"

    # Snapshot configs used.
    cfg_dir = ctx.audit_dir / "configs_used"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for src in [ctx.config, ctx.mega_config, ctx.datasets_config, ctx.lawc_event_map, ctx.mechanism_event_map, ctx.pearl_event_map]:
        if src.exists():
            shutil.copy2(src, cfg_dir / src.name)

    include_roots = [
        ctx.audit_dir,
        ctx.pack_core,
        ctx.pack_mechanism,
        ctx.pack_normative,
        ctx.pack_clinical,
        ctx.pack_mapping,
        ctx.pack_features,
        ctx.out_root / "gpu_util.csv",
    ]

    added: List[str] = []
    try:
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root in include_roots:
                if isinstance(root, Path) and root.is_file():
                    rel = root.relative_to(ctx.out_root)
                    zf.write(root, rel.as_posix())
                    added.append(rel.as_posix())
                    continue
                if not isinstance(root, Path) or not root.exists():
                    continue
                for p in sorted(root.rglob("*")):
                    if p.is_dir():
                        continue
                    rel = p.relative_to(ctx.out_root)
                    zf.write(p, rel.as_posix())
                    added.append(rel.as_posix())
    except Exception as exc:
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="build report + zip",
            outputs=[report],
            error=str(exc),
        )

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="build report + zip",
        outputs=[report, zpath],
        extra={"zip_file_count": int(len(added))},
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path, default=None)
    ap.add_argument("--data_root", type=Path, default=Path("/filesystemHcog/openneuro"))
    ap.add_argument("--features_root", type=Path, default=None)

    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--mega_config", type=Path, default=Path("configs/nn_final_mega.yaml"))
    ap.add_argument("--datasets_config", type=Path, default=Path("configs/datasets_nn_final_mega.yaml"))

    ap.add_argument("--lawc_event_map", type=Path, default=Path("configs/lawc_event_map.yaml"))
    ap.add_argument("--mechanism_event_map", type=Path, default=Path("configs/mechanism_event_map.yaml"))
    ap.add_argument("--pearl_event_map", type=Path, default=Path("configs/pearl_event_map.yaml"))

    ap.add_argument("--wall_hours", type=float, default=10.0)
    ap.add_argument("--resume", type=str, default="false", help="true/false")

    ap.add_argument("--gpu_parallel_procs", type=int, default=0)
    ap.add_argument("--cpu_workers", type=int, default=0)
    return ap.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def main() -> int:
    args = parse_args()

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root or (Path("/filesystemHcog/runs") / f"{ts}_NN_FINAL_MEGA")
    features_root = args.features_root or Path(f"/filesystemHcog/features_cache_NN_FINAL_MEGA_{ts}")

    resume = _parse_bool(args.resume)

    if out_root.exists() and not resume:
        audit = out_root / "AUDIT"
        has_prior = (audit / "run_status.json").exists() or bool(list(audit.glob("*_summary.json")))
        if has_prior:
            print(f"ERROR: out_root exists and --resume is false: {out_root}", file=sys.stderr, flush=True)
            return 1

    runtime_env = os.environ.copy()
    runtime_env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'src'}:{runtime_env.get('PYTHONPATH', '')}" if runtime_env.get("PYTHONPATH") else str(REPO_ROOT / "src")
    )

    cfg_mega = _load_yaml(args.mega_config)
    cfg_datasets = _load_yaml(args.datasets_config)

    ctx = RunContext(
        out_root=out_root,
        audit_dir=out_root / "AUDIT",
        outzip_dir=out_root / "OUTZIP",
        pack_core=out_root / "PACK_CORE_LAWC",
        pack_mechanism=out_root / "PACK_MECHANISM",
        pack_normative=out_root / "PACK_NORMATIVE",
        pack_clinical=out_root / "PACK_CLINICAL",
        pack_mapping=out_root / "PACK_MAPPING",
        pack_features=out_root / "PACK_FEATURES",
        data_root=args.data_root,
        features_root=features_root,
        config=args.config,
        mega_config=args.mega_config,
        datasets_config=args.datasets_config,
        lawc_event_map=args.lawc_event_map,
        mechanism_event_map=args.mechanism_event_map,
        pearl_event_map=args.pearl_event_map,
        wall_hours=float(args.wall_hours),
        resume=bool(resume),
        start_ts=time.time(),
        deadline_ts=time.time() + float(args.wall_hours) * 3600.0,
        stage_records=[],
        stage_status={},
        stage_outputs={},
        monitor_proc=None,
        monitor_handle=None,
        nvml_logger=None,
        runtime_env=runtime_env,
        cfg_mega=cfg_mega,
        cfg_datasets=cfg_datasets,
        gpu_parallel_procs=int(max(0, args.gpu_parallel_procs)),
        cpu_workers=int(max(0, args.cpu_workers)),
        partial_reasons=[],
    )

    ctx.out_root.mkdir(parents=True, exist_ok=True)
    ctx.audit_dir.mkdir(parents=True, exist_ok=True)
    ctx.outzip_dir.mkdir(parents=True, exist_ok=True)

    run_status = "PASS"
    run_error = ""

    stages = [
        _stage_preflight,
        _stage_compile_gate,
        _stage_stage_datasets,
        _stage_decode_mapping_all,
        _stage_extract_features_all,
        _stage_core_lawc_ultradeep,
        _stage_mechanism_deep,
        _stage_normative_lodo_manyseed,
        _stage_clinical_translation,
    ]

    try:
        for fn in stages:
            st = fn.__name__.replace("_stage_", "")

            if ctx.resume and st != "preflight":
                resumed = _load_resume_record(ctx, st)
                if resumed is not None:
                    ctx.stage_records.append(resumed)
                    ctx.stage_status[st] = resumed["status"]
                    if resumed["status"] == "SKIP":
                        ctx.partial_reasons.append(f"{st} resumed as SKIP")
                    continue

            rec = fn(ctx)
            if rec["status"] == "FAIL":
                run_status = "FAIL"
                run_error = rec.get("error", "stage failed")
                break
            if rec["status"] == "SKIP":
                ctx.partial_reasons.append(f"{st} skipped: {rec.get('error', '')}")

    except Exception as exc:
        run_status = "FAIL"
        run_error = f"{type(exc).__name__}: {exc}\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    if run_status == "PASS" and ctx.partial_reasons:
        run_status = "PARTIAL_PASS"

    # final bundle always runs.
    try:
        rec_final = _stage_final_bundle(ctx, run_status, run_error)
        if rec_final["status"] == "FAIL" and run_status != "FAIL":
            run_status = "FAIL"
            run_error = rec_final.get("error", "final bundle failed")
    except Exception as exc:
        run_status = "FAIL"
        run_error = f"final_bundle exception: {exc}"

    _stop_monitors(ctx)

    report_path = ctx.audit_dir / "NN_FINAL_MEGA_REPORT.md"
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
        "partial_reasons": ctx.partial_reasons,
    }
    _write_json(ctx.audit_dir / "run_status.json", run_status_payload)

    print(f"OUT_ROOT={ctx.out_root}", flush=True)
    print(f"REPORT={report_path}", flush=True)
    _print_stage_table(ctx)
    print(f"BUNDLE={ctx.outzip_dir / 'NN_FINAL_MEGA_SUBMISSION_PACKET.zip'}", flush=True)

    return 0 if run_status in {"PASS", "PARTIAL_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
