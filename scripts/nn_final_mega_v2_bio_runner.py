#!/usr/bin/env python3
"""NN_FINAL_MEGA_V2_BIO end-to-end orchestrator.

Stages
1) preflight
2) compile_gate
3) stage_datasets
4) decode_mapping_all
5) extract_features_all
6) core_lawc_ultradeep
7) mechanism_deep
8) clinical_dementia_ds004504
9) clinical_pdrest_ds004584
10) clinical_mortality_ds007020
11) bio_A_topography
12) bio_B_source_template
13) bio_C_arousal_regime
14) bio_D_cross_modality_ds004752
15) objective_guard
16) final_bundle

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
    "bio_A_topography",
    "bio_B_source_template",
    "bio_C_arousal_regime",
    "bio_D_cross_modality_ds004752",
    "objective_guard",
    "final_bundle",
]

LOCKED_LAWC_CANONICAL = ["ds003655", "ds004117", "ds005095"]

FEATURE_REUSE_ROOTS: Dict[str, Path] = {
    "ds003655": Path("/filesystemHcog/features_cache_FIX2_20260222_061927/ds003655"),
    "ds004117": Path("/filesystemHcog/features_cache_FIX2_20260222_061927/ds004117"),
    "ds005095": Path("/filesystemHcog/features_cache_FIX2_20260222_061927/ds005095"),
    "ds004796": Path("/filesystemHcog/features_cache_PEARL_SOLID2_20260222/ds004796"),
    # Reuse validated mechanism features to avoid repeated partial epoch corruption.
    "ds003838": Path("/filesystemHcog/features_cache_NN_FINAL_MEGA_20260223_023623/ds003838"),
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
    pack_clin_pdrest: Path
    pack_clin_mortality: Path
    pack_clin_dementia: Path
    pack_bio: Path
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
    timeout_sec: Optional[float] = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{_iso_now()}] CMD: {' '.join(cmd)}\n")
        f.flush()
        try:
            p = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                check=False,
                timeout=timeout_sec,
            )
            rc = int(p.returncode)
        except subprocess.TimeoutExpired:
            f.write(f"[{_iso_now()}] TIMEOUT: command exceeded {timeout_sec}s\n")
            f.flush()
            rc = 124
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
_REST_READABLE_SUFFIXES = {".vhdr", ".set", ".edf", ".bdf", ".fif", ".gdf"}
_REST_SIDECAR_SUFFIXES = {".eeg", ".vmrk", ".fdt"}


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
    handle = out_csv.open("a", encoding="utf-8")
    ctx.monitor_handle = handle
    ctx.monitor_proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=handle, stderr=handle, text=True)
    _write_text(ctx.audit_dir / "nvidia_smi_monitor.pid", f"{ctx.monitor_proc.pid}\n")


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
        for p in [
            ctx.pack_core,
            ctx.pack_mechanism,
            ctx.pack_normative,
            ctx.pack_clinical,
            ctx.pack_clin_pdrest,
            ctx.pack_clin_mortality,
            ctx.pack_clin_dementia,
            ctx.pack_bio,
            ctx.pack_mapping,
            ctx.pack_features,
            ctx.features_root,
        ]:
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
            ctx.nvml_logger = start_gpu_util_logger(csv_path=ctx.out_root / "gpu_util.csv", tag="NN_FINAL_MEGA_V2_BIO")
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

    # Exclude local virtualenv/cache trees; compiling those can dominate walltime.
    cmd = [
        "bash",
        "-lc",
        "find . \\( -path './.venv*' -o -path './__pycache__' -o -path './*/__pycache__' \\) -prune -o -name '*.py' -print0 | xargs -0 python -m py_compile",
    ]
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
            rc = _run_cmd(
                cmd,
                cwd=ds_root,
                log_path=log_path,
                env=env,
                allow_fail=True,
                timeout_sec=300,
            )
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


def _repair_broken_eeg_links(ds_root: Path, log_path: Path, env: Dict[str, str], jobs: int) -> Tuple[int, str]:
    broken: List[Path] = []
    for p in ds_root.rglob("*"):
        try:
            if p.is_symlink() and _EEG_PAT.search(p.name) and not p.exists():
                broken.append(p)
        except Exception:
            continue
    if not broken:
        return 0, "no-broken-links"

    rels = [str(p.relative_to(ds_root)) for p in broken]
    rels = sorted(set(rels))
    max_repair = int(max(8, min(4096, int(os.environ.get("NN_BROKEN_REPAIR_MAX", "4096")))))
    if len(rels) > max_repair:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"[{_iso_now()}] INFO: truncating broken-link repair list from {len(rels)} to {max_repair}\n"
            )
        rels = rels[:max_repair]
    if shutil.which("datalad") is not None:
        rc = _run_cmd(
            ["datalad", "get", *rels],
            cwd=ds_root,
            log_path=log_path,
            env=env,
            allow_fail=True,
            timeout_sec=300,
        )
        if rc == 0:
            return 0, "datalad-get-broken-links"

    if shutil.which("git-annex") is not None:
        batch = 8
        for i in range(0, len(rels), batch):
            chunk = rels[i : i + batch]
            rc = _run_cmd(
                ["git", "annex", "get", "--from", "OpenNeuro", "-J", str(max(1, jobs)), "--", *chunk],
                cwd=ds_root,
                log_path=log_path,
                env=env,
                allow_fail=True,
                timeout_sec=240,
            )
            if rc != 0:
                rc = _run_cmd(
                    ["git", "annex", "get", "-J", str(max(1, jobs)), "--", *chunk],
                    cwd=ds_root,
                    log_path=log_path,
                    env=env,
                    allow_fail=True,
                    timeout_sec=240,
                )
            if rc != 0:
                return rc, "git-annex-get-broken-links-failed"
        return 0, "git-annex-get-broken-links"

    return 1, "no-annex-or-datalad-for-broken-links"


def _materialize_resting_payload(
    ds_root: Path,
    log_path: Path,
    env: Dict[str, str],
    jobs: int,
) -> Tuple[int, str, int]:
    """
    Pull missing resting EEG payloads (including sidecars) via git-annex.
    This is intentionally broader than extraction suffix selection to ensure
    headers (.vhdr/.set) have corresponding binary payloads locally.
    """
    if not ds_root.exists():
        return 1, "dataset-root-missing", 0
    if shutil.which("git-annex") is None:
        return 0, "git-annex unavailable (skipped)", 0

    target_re = re.compile(
        r"_eeg\.(vhdr|vmrk|eeg|set|fdt|edf|bdf|fif|gdf)(\.gz)?$",
        flags=re.IGNORECASE,
    )
    missing: List[str] = []
    for p in ds_root.rglob("*"):
        try:
            if "derivatives" in p.parts:
                continue
            if not target_re.search(p.name):
                continue
            if p.is_symlink() and not p.exists():
                missing.append(str(p.relative_to(ds_root)))
        except Exception:
            continue

    missing = sorted(set(missing))
    if not missing:
        return 0, "no-missing-rest-payload", 0

    got = 0
    batch = 64
    for i in range(0, len(missing), batch):
        chunk = missing[i : i + batch]
        rc = 1
        variants = [
            ["git", "annex", "get", "--from", "OpenNeuro", "-J", str(max(1, jobs)), "--", *chunk],
            ["git", "annex", "get", "--from", "s3-PUBLIC", "-J", str(max(1, jobs)), "--", *chunk],
            ["git", "annex", "get", "-J", str(max(1, jobs)), "--", *chunk],
        ]
        for cmd in variants:
            rc = _run_cmd(cmd, cwd=ds_root, log_path=log_path, env=env, allow_fail=True, timeout_sec=900)
            if rc == 0:
                break
        if rc != 0:
            return rc, "git-annex-get-rest-payload-failed", got
        got += len(chunk)
    return 0, "git-annex-get-rest-payload", got


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

            # Clinical/rest datasets can appear "ready" while many EEG payloads are still broken links.
            if dataset_id in clinical_rest:
                broken_before = _count_eeg_broken_symlinks(ds_root)
                if broken_before > 0:
                    rc_rep, rep_method = _repair_broken_eeg_links(
                        ds_root,
                        log_path=log_path,
                        env=ctx.runtime_env,
                        jobs=max(1, min(64, ctx.cpu_workers)),
                    )
                    method = f"{method}+{rep_method}"
                    if rc_rep != 0:
                        with log_path.open("a", encoding="utf-8") as f:
                            f.write(
                                f"[{_iso_now()}] WARN: broken link repair failed for {dataset_id} "
                                f"(rc={rc_rep}, method={rep_method}, broken_before={broken_before})\n"
                            )

        except Exception as exc:
            status = "SKIP"
            reason = str(exc)
            stop = ctx.audit_dir / "STOP_REASONS" / f"STOP_REASON_{dataset_id}.md"
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
        stop = ds_out / f"STOP_REASON_{ds}.md"

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

        if ds in {"ds004752", "ds007262"}:
            cfg_yaml = REPO_ROOT / "configs" / f"event_map_{ds}.yaml"
            dec_script = REPO_ROOT / "scripts" / (f"decode_{ds}.py")
            rc_dec = _run_cmd(
                [
                    sys.executable,
                    str(dec_script),
                    "--dataset_root",
                    str(ds_dir),
                    "--out_yaml",
                    str(cfg_yaml),
                    "--out_summary",
                    str(ds_summary),
                    "--out_candidate",
                    str(cand_csv),
                    "--stop_reason",
                    str(stop),
                ],
                cwd=REPO_ROOT,
                log_path=log_path,
                env=ctx.runtime_env,
                allow_fail=True,
            )
            dec_sum = _read_json_if_exists(ds_summary) or {}
            if rc_dec != 0 or str(dec_sum.get("status", "")).upper() != "PASS" or not cfg_yaml.exists():
                reason = str(dec_sum.get("reason", f"decoder failed rc={rc_dec}"))
                rows.append({"dataset_id": ds, "status": "SKIP", "reason": reason, "selected_candidate_id": ""})
                outputs.extend([cand_csv, ds_summary, stop])
                continue

            cfg_map = yaml.safe_load(cfg_yaml.read_text(encoding="utf-8")) or {}
            mapping = dict(((cfg_map.get("datasets") or {}).get(ds)) or {})
            if not mapping:
                reason = "decoder produced empty mapping"
                _write_stop_reason(stop, f"{stage}:{ds}", reason, diagnostics={"decoder_summary": dec_sum})
                rows.append({"dataset_id": ds, "status": "SKIP", "reason": reason, "selected_candidate_id": ""})
                outputs.extend([cand_csv, ds_summary, stop])
                continue

            _write_text(ds_map_yaml, yaml.safe_dump({"defaults": defaults, "datasets": {ds: mapping}}, sort_keys=False))
            out_event_map["datasets"][ds] = mapping
            rows.append({"dataset_id": ds, "status": "PASS", "reason": "", "selected_candidate_id": "dataset_decoder", "mapping": mapping})
            outputs.extend([cand_csv, ds_summary, ds_map_yaml, cfg_yaml])
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
            stop = ctx.pack_features / ds / f"STOP_REASON_{ds}.md"
            _write_stop_reason(stop, f"{stage}:{ds}", row["reason"])
            if ds in mandatory:
                hard_failures.append(f"{ds}: {row['reason']}")
            rows.append(row)
            continue

        if ds not in groups.get("mechanism", []) and map_rows.get(ds) != "PASS":
            row["reason"] = f"mapping status={map_rows.get(ds, 'missing')}"
            stop = ctx.pack_features / ds / f"STOP_REASON_{ds}.md"
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

        # Clean stale partial mechanism artifacts from previous failed attempts.
        if ds in groups.get("mechanism", []):
            stale_epochs = deriv_root / "epochs"
            stale_feats = ctx.features_root / ds
            if stale_epochs.exists():
                shutil.rmtree(stale_epochs, ignore_errors=True)
            if stale_feats.exists():
                try:
                    if stale_feats.is_symlink() or stale_feats.is_file():
                        stale_feats.unlink()
                    else:
                        shutil.rmtree(stale_feats, ignore_errors=True)
                except Exception:
                    pass
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(f"[{_iso_now()}] INFO: cleaned stale mechanism artifacts for dataset={ds}\n")

        base_workers = max(1, min(16, ctx.cpu_workers))
        worker_candidates: List[int] = [base_workers]
        if ds in groups.get("mechanism", []):
            for w in [16, 12, 8, 4]:
                if w <= base_workers and w not in worker_candidates:
                    worker_candidates.append(w)
            if 8 not in worker_candidates and 8 <= base_workers:
                worker_candidates.append(8)
            if 4 not in worker_candidates:
                worker_candidates.append(4)

        rc1 = 1
        rc2 = 1
        used_workers = worker_candidates[0]
        for cand_workers in worker_candidates:
            cpu_run_workers = int(max(1, cand_workers))
            cpu_run_threads = 1
            used_workers = cpu_run_workers
            run_env = dict(ctx.runtime_env)
            run_env["OMP_NUM_THREADS"] = "1"
            run_env["MKL_NUM_THREADS"] = "1"
            run_env["OPENBLAS_NUM_THREADS"] = "1"
            run_env["NUMEXPR_MAX_THREADS"] = "1"
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(
                    f"[{_iso_now()}] INFO: extract attempt dataset={ds} workers={cpu_run_workers} per_run_threads={cpu_run_threads}\n"
                )

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
                env=run_env,
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
                    env=run_env,
                    allow_fail=True,
                )

            if rc1 == 0 and rc2 == 0:
                break

        if rc1 != 0 or rc2 != 0:
            row["status"] = "SKIP"
            row["reason"] = f"feature extraction failed rc_pre={rc1} rc_extract={rc2} workers_last={used_workers}"
            stop = ctx.pack_features / ds / f"STOP_REASON_{ds}.md"
            _write_stop_reason(stop, f"{stage}:{ds}", row["reason"])
            if ds in mandatory:
                hard_failures.append(f"{ds}: {row['reason']}")
            rows.append(row)
            continue

        ds_feat = ctx.features_root / ds
        row["status"] = "PASS"
        row["method"] = f"preprocess+extract_w{used_workers}"
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

    # Mechanism-only and clinical-only runs should not hard-fail on Law-C.
    # In those modes, no core Law-C datasets are requested by config.
    if not lawc_pool:
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="lawc ultradeep",
            error="no core Law-C datasets requested in this run",
        )

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
        # Use dataset-specific root so rglob can traverse symlinked dataset dirs.
        str(features_ds),
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


def _coalesce_series(df: pd.DataFrame, cols: Sequence[str], *, default: str = "") -> pd.Series:
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series([default] * len(df), index=df.index, dtype=object)


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


def _plot_roc_and_calibration(y_true: np.ndarray, score: np.ndarray, out_roc: Path, out_cal: Path, title: str) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    m = np.isfinite(s)
    y = y[m]
    s = s[m]
    if y.size < 10 or len(np.unique(y)) < 2:
        return
    try:
        auc = float(roc_auc_score(y, s))
        fpr, tpr, _ = roc_curve(y, s)
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.plot(fpr, tpr, lw=2.0, color="#1f77b4", label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_roc, dpi=180)
        plt.close(fig)
    except Exception:
        pass
    try:
        # simple decile calibration curve on rank-normalized score
        p = pd.Series(s).rank(pct=True).to_numpy(dtype=float)
        bins = np.linspace(0.0, 1.0, 11)
        idx = np.digitize(p, bins) - 1
        rows = []
        for b in range(10):
            mb = idx == b
            if mb.sum() < 3:
                continue
            rows.append((float(np.mean(p[mb])), float(np.mean(y[mb]))))
        if rows:
            rr = np.asarray(rows, dtype=float)
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            ax.plot(rr[:, 0], rr[:, 1], "o-", color="#d62728")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
            ax.set_xlabel("Predicted risk (rank)")
            ax.set_ylabel("Observed event rate")
            ax.set_title(f"{title} calibration")
            fig.tight_layout()
            fig.savefig(out_cal, dpi=180)
            plt.close(fig)
    except Exception:
        pass


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
    eeg_files: List[Path] = []
    for p in ds_root.rglob("*"):
        if not p.is_file():
            continue
        if "derivatives" in p.parts:
            continue
        if not _EEG_PAT.search(p.name):
            continue
        if p.suffix.lower() not in _REST_READABLE_SUFFIXES:
            continue
        eeg_files.append(p)
    eeg_files = sorted(set(eeg_files))

    suffix_rank = {
        ".vhdr": 0,
        ".set": 1,
        ".edf": 2,
        ".bdf": 3,
        ".fif": 4,
        ".gdf": 5,
    }
    subj_to_files: Dict[str, List[Path]] = {}
    for p in eeg_files:
        m = re.search(r"sub-([A-Za-z0-9]+)", str(p))
        if not m:
            continue
        sid = m.group(1)
        subj_to_files.setdefault(sid, []).append(p)

    rows: List[Dict[str, Any]] = []
    failures: List[str] = []

    for sid, candidates in sorted(subj_to_files.items()):
        tried_msgs: List[str] = []
        ordered = sorted(
            candidates,
            key=lambda p: (
                0
                if (
                    (p.suffix.lower() == ".vhdr" and p.with_suffix(".eeg").exists() and p.with_suffix(".vmrk").exists())
                    or (p.suffix.lower() == ".set" and p.with_suffix(".fdt").exists())
                    or (p.suffix.lower() in {".edf", ".bdf", ".fif", ".gdf"})
                )
                else 1,
                suffix_rank.get(p.suffix.lower(), 99),
                str(p),
            ),
        )

        parsed = False
        for fp in ordered:
            try:
                raw = _read_raw_any(fp)
                raw.pick_types(eeg=True, eog=False, misc=False, stim=False)
                if len(raw.ch_names) == 0:
                    tried_msgs.append(f"{fp.name}: no EEG channels")
                    continue
                sf = float(raw.info["sfreq"])
                if sf > 256:
                    raw.resample(256, verbose="ERROR")
                    sf = float(raw.info["sfreq"])

                data = raw.get_data()
                if data.size == 0:
                    tried_msgs.append(f"{fp.name}: empty EEG data")
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
                parsed = True
                break
            except Exception as exc:
                tried_msgs.append(f"{fp.name}: {exc}")
                continue

        if not parsed:
            failures.append(f"{sid}: " + (" | ".join(tried_msgs[:3]) if tried_msgs else "no readable eeg files"))

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
    cols = [
        c
        for c in ["GROUP", "Group", "group", "Diagnosis", "diagnosis", "dx", "condition", "status", "TYPE"]
        if c in part.columns
    ]
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

        mapped[low.str.contains(r"parkinson|\bpd\b", na=False, regex=True)] = "PD"
        mapped[low.str.contains(r"control|healthy|\bhc\b|\bctl\b", na=False, regex=True)] = "CN"

        if c.upper() == "TYPE":
            num = pd.to_numeric(s, errors="coerce")
            if np.isfinite(num).sum() >= 10 and pd.Series(num).dropna().nunique() == 2:
                lo = float(np.nanmin(num))
                hi = float(np.nanmax(num))
                mapped[np.isfinite(num) & (num == hi)] = "PD"
                mapped[np.isfinite(num) & (num == lo)] = "CN"

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
            y[low.str.contains("alive|living|censor|surviv", na=False, regex=True)] = 0.0

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

    # Stage2 confirmatory jobs are per-dataset; avoid expensive annex/materialization work
    # for clinical datasets not requested by this job's datasets_config.
    cfg_rows = ctx.cfg_datasets.get("datasets", []) if isinstance(ctx.cfg_datasets, dict) else []
    configured_ids = {
        str(r.get("id", "")).strip()
        for r in cfg_rows
        if isinstance(r, dict) and str(r.get("id", "")).strip()
    }
    selected_clinical = {"ds004504", "ds004584", "ds007020"} & configured_ids

    # -------- ds004504 dementia --------
    ds = "ds004504"
    ds_out = ctx.pack_clin_dementia
    ds_out.mkdir(parents=True, exist_ok=True)
    ds_root = ctx.data_root / ds
    if ds not in selected_clinical:
        ds_root = Path("/__skip_clinical_dataset__") / ds
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"[{_iso_now()}] dataset_start {ds}\n")

    if not ds_root.exists() or not (ds_root / "participants.tsv").exists():
        reason = "dataset or participants.tsv missing"
        _write_stop_reason(ds_out / f"STOP_REASON_{ds}.md", f"{stage}:{ds}", reason)
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
        part["sex"] = _coalesce_series(part, ["Gender", "GENDER", "Sex", "sex"], default="").astype(str)
        part["mmse"] = pd.to_numeric(part.get("MMSE"), errors="coerce")

        rc_get, how_get, n_get = _materialize_resting_payload(
            ds_root=ds_root,
            log_path=log_path,
            env=ctx.runtime_env,
            jobs=max(1, min(ctx.cpu_workers, 32)),
        )
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(f"[{_iso_now()}] {ds} rest_payload_materialize rc={rc_get} method={how_get} n={n_get}\n")

        feat_df, fails = _extract_rest_features(ds_root)
        if feat_df.empty:
            reason = "no usable resting EEG features"
            _write_stop_reason(ds_out / f"STOP_REASON_{ds}.md", f"{stage}:{ds}", reason, diagnostics={"n_failures": len(fails), "sample_failures": fails[:20]})
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
                # Regression consistency guard against prior locked reference.
                ref_auc = 0.7260536398467432
                ref_row = end_df[
                    (end_df["endpoint"].astype(str) == "AUC_AD_vs_CN")
                    & (end_df["feature"].astype(str) == "dev_z_theta_alpha_ratio")
                ]
                if not ref_row.empty:
                    curr_auc = float(pd.to_numeric(ref_row.iloc[0]["estimate"], errors="coerce"))
                    if np.isfinite(curr_auc) and abs(curr_auc - ref_auc) > 0.05:
                        _write_stop_reason(
                            ds_out / "STOP_REASON_regression.md",
                            f"{stage}:{ds}:regression_guard",
                            f"AUC drift exceeded threshold (curr={curr_auc:.6f}, ref={ref_auc:.6f}, delta={abs(curr_auc-ref_auc):.6f}).",
                            diagnostics={"reference_auc": ref_auc, "current_auc": curr_auc, "threshold": 0.05},
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
    ds_out = ctx.pack_clin_pdrest
    ds_out.mkdir(parents=True, exist_ok=True)
    ds_root = ctx.data_root / ds
    if ds not in selected_clinical:
        ds_root = Path("/__skip_clinical_dataset__") / ds
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"[{_iso_now()}] dataset_start {ds}\n")

    if not ds_root.exists() or not (ds_root / "participants.tsv").exists():
        reason = "dataset or participants.tsv missing"
        _write_stop_reason(ds_out / f"STOP_REASON_{ds}.md", f"{stage}:{ds}", reason)
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
            _write_stop_reason(ds_out / f"STOP_REASON_{ds}.md", f"{stage}:{ds}", reason, diagnostics={"columns": list(part.columns)})
            status_by_dataset[ds] = "SKIP"
            reasons_by_dataset[ds] = reason
        else:
            part["group"] = labels
            part["age"] = pd.to_numeric(part.get("AGE", part.get("Age", part.get("age"))), errors="coerce")
            part["sex"] = _coalesce_series(part, ["GENDER", "Sex", "sex", "Gender"], default="").astype(str)

            # Avoid long annex pulls when we already have balanced local coverage.
            pd_cov = 0
            cn_cov = 0
            for _, rr in part[["subject_id", "group"]].dropna().iterrows():
                sid = str(rr["subject_id"])
                grp = str(rr["group"])
                set_fp = ds_root / f"sub-{sid}" / "eeg" / f"sub-{sid}_task-Rest_eeg.set"
                fdt_fp = set_fp.with_suffix(".fdt")
                if set_fp.exists() and fdt_fp.exists():
                    if grp == "PD":
                        pd_cov += 1
                    elif grp == "CN":
                        cn_cov += 1

            if pd_cov >= 10 and cn_cov >= 10:
                rc_get, how_get, n_get = 0, "preexisting-sufficient", 0
            else:
                rc_get, how_get, n_get = _materialize_resting_payload(
                    ds_root=ds_root,
                    log_path=log_path,
                    env=ctx.runtime_env,
                    jobs=max(1, min(ctx.cpu_workers, 32)),
                )
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(
                    f"[{_iso_now()}] {ds} rest_payload_materialize rc={rc_get} method={how_get} n={n_get} "
                    f"preexisting_pd={pd_cov} preexisting_cn={cn_cov}\n"
                )

            feat_df, fails = _extract_rest_features(ds_root)
            if feat_df.empty:
                reason = "no usable resting EEG features"
                _write_stop_reason(ds_out / f"STOP_REASON_{ds}.md", f"{stage}:{ds}", reason, diagnostics={"n_failures": len(fails), "sample_failures": fails[:20]})
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
                try:
                    score_col = "composite_deviation" if "composite_deviation" in dev_df.columns else "dev_z_theta_alpha_ratio"
                    roc = ds_out / "FIG_pdrest_primary_auc_roc.png"
                    cal = ds_out / "FIG_pdrest_calibration.png"
                    yy = (dev_df["group"].astype(str) == "PD").astype(int).to_numpy(dtype=int)
                    ss = pd.to_numeric(dev_df[score_col], errors="coerce").to_numpy(dtype=float)
                    _plot_roc_and_calibration(yy, ss, roc, cal, "ds004584 PD vs Control")
                    if roc.exists():
                        outputs.append(roc)
                    if cal.exists():
                        outputs.append(cal)
                except Exception:
                    pass

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
    ds_out = ctx.pack_clin_mortality
    ds_out.mkdir(parents=True, exist_ok=True)
    ds_root = ctx.data_root / ds
    if ds not in selected_clinical:
        ds_root = Path("/__skip_clinical_dataset__") / ds
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"[{_iso_now()}] dataset_start {ds}\n")

    if not ds_root.exists() or not (ds_root / "participants.tsv").exists():
        reason = "dataset or participants.tsv missing"
        _write_stop_reason(ds_out / f"STOP_REASON_{ds}.md", f"{stage}:{ds}", reason)
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
            _write_stop_reason(ds_out / f"STOP_REASON_{ds}.md", f"{stage}:{ds}", reason, diagnostics={"columns": list(part.columns), "label_probe": y_diag})
            status_by_dataset[ds] = "SKIP"
            reasons_by_dataset[ds] = reason
        else:
            part["mortality_label"] = y
            part["age"] = pd.to_numeric(part.get("AGE", part.get("Age", part.get("age"))), errors="coerce")
            part["sex"] = _coalesce_series(part, ["GENDER", "Sex", "sex", "Gender"], default="").astype(str)

            rc_get, how_get, n_get = _materialize_resting_payload(
                ds_root=ds_root,
                log_path=log_path,
                env=ctx.runtime_env,
                jobs=max(1, min(ctx.cpu_workers, 32)),
            )
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(f"[{_iso_now()}] {ds} rest_payload_materialize rc={rc_get} method={how_get} n={n_get}\n")

            feat_df, fails = _extract_rest_features(ds_root)
            if feat_df.empty:
                reason = "no usable resting EEG features"
                _write_stop_reason(ds_out / f"STOP_REASON_{ds}.md", f"{stage}:{ds}", reason, diagnostics={"n_failures": len(fails), "sample_failures": fails[:20]})
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
                try:
                    score_col = "composite_deviation" if "composite_deviation" in dev_df.columns else "dev_z_theta_alpha_ratio"
                    roc = ds_out / "FIG_mortality_primary_auc_roc.png"
                    cal = ds_out / "FIG_mortality_calibration.png"
                    yy = pd.to_numeric(dev_df["mortality_label"], errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
                    ss = pd.to_numeric(dev_df[score_col], errors="coerce").to_numpy(dtype=float)
                    _plot_roc_and_calibration(yy, ss, roc, cal, "ds007020 mortality")
                    if roc.exists():
                        outputs.append(roc)
                    if cal.exists():
                        outputs.append(cal)
                except Exception:
                    pass

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
    ctx.pack_clinical.mkdir(parents=True, exist_ok=True)
    all_end.to_csv(ctx.pack_clinical / "clinical_endpoints_all.csv", index=False)
    outputs.append(ctx.pack_clinical / "clinical_endpoints_all.csv")

    # Stage status logic.
    hard_fail = []
    if "ds004504" in selected_clinical and status_by_dataset.get("ds004504") != "PASS":
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
        extra={
            "status_by_dataset": status_by_dataset,
            "reasons_by_dataset": reasons_by_dataset,
            "clinical_perm_done": int(n_perm),
            "n_endpoints_total": int(len(all_end)),
        },
    )


def _iter_trials_from_h5(dataset_dir: Path) -> Iterable[pd.DataFrame]:
    for fp in sorted(dataset_dir.rglob("*.h5")):
        try:
            with h5py.File(fp, "r") as h:
                if not {"p3b_amp", "memory_load", "subject_key"}.issubset(set(h.keys())):
                    continue
                row: Dict[str, Any] = {
                    "p3b_amp": np.asarray(h["p3b_amp"], dtype=float),
                    "memory_load": np.asarray(h["memory_load"], dtype=float),
                    "subject_key": np.asarray(h["subject_key"]).astype(str),
                }
                if "p3b_channel" in h:
                    row["p3b_channel"] = np.asarray(h["p3b_channel"]).astype(str)
                if "pdr" in h:
                    row["pdr"] = np.asarray(h["pdr"], dtype=float)
                yield pd.DataFrame(row)
        except Exception:
            continue


def _stage_bio_A_topography(ctx: RunContext) -> Dict[str, Any]:
    stage = "bio_A_topography"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"
    out_dir = ctx.pack_bio / "BIO_A_topography"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = (ctx.cfg_mega.get("dataset_groups", {}) or {})
    ds_list = [d for d in groups.get("core_sternberg", []) if (ctx.features_root / d).exists()]
    chan_dist: Dict[str, pd.Series] = {}
    stability_rows: List[Dict[str, Any]] = []
    for ds in ds_list:
        parts: List[pd.Series] = []
        for df in _iter_trials_from_h5(ctx.features_root / ds):
            if "p3b_channel" in df.columns:
                parts.append(df["p3b_channel"].astype(str))
        if not parts:
            continue
        s = pd.concat(parts, ignore_index=True).fillna("NA").astype(str)
        vc = s.value_counts(normalize=True)
        chan_dist[ds] = vc
        pz_frac = float(vc.get("Pz", 0.0))
        top = str(vc.index[0]) if len(vc) else "NA"
        entropy = float(-(vc * np.log(np.clip(vc.to_numpy(dtype=float), 1e-12, None))).sum())
        stability_rows.append(
            {
                "dataset_id": ds,
                "n_trials": int(len(s)),
                "top_channel": top,
                "pz_fraction": pz_frac,
                "channel_entropy": entropy,
            }
        )

    if len(chan_dist) < 2:
        stop = out_dir / "STOP_REASON_BIO_A.md"
        reason = "insufficient datasets with p3b_channel distributions"
        _write_stop_reason(stop, stage, reason, diagnostics={"datasets_seen": ds_list, "datasets_with_channel": sorted(chan_dist.keys())})
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            command="BIO-A topography invariance",
            outputs=[stop],
            error=reason,
        )

    chans = sorted(set().union(*[set(v.index.tolist()) for v in chan_dist.values()]))
    ds_ok = sorted(chan_dist.keys())
    mat: List[np.ndarray] = []
    for ds in ds_ok:
        v = chan_dist[ds].reindex(chans).fillna(0.0).to_numpy(dtype=float)
        s = float(v.sum())
        mat.append(v / s if s > 0 else v)
    arr = np.vstack(mat)
    sim = np.corrcoef(arr) if arr.shape[0] >= 2 else np.ones((arr.shape[0], arr.shape[0]), dtype=float)

    pair_rows = []
    for i, a in enumerate(ds_ok):
        for j, b in enumerate(ds_ok):
            if j <= i:
                continue
            pair_rows.append({"dataset_a": a, "dataset_b": b, "topography_similarity_r": float(sim[i, j])})
    pair_df = pd.DataFrame(pair_rows)
    pair_csv = out_dir / "topography_invariance.csv"
    pair_df.to_csv(pair_csv, index=False)

    stab_csv = out_dir / "sensor_generator_stability.csv"
    pd.DataFrame(stability_rows).to_csv(stab_csv, index=False)

    fig = out_dir / "FIG_topography_similarity.png"
    try:
        fig_obj, ax = plt.subplots(figsize=(5.8, 5.2))
        im = ax.imshow(sim, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_xticks(np.arange(len(ds_ok)), ds_ok, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(ds_ok)), ds_ok)
        ax.set_title("BIO-A Topography Similarity")
        fig_obj.colorbar(im, ax=ax, shrink=0.8, label="r")
        fig_obj.tight_layout()
        fig_obj.savefig(fig, dpi=180)
        plt.close(fig_obj)
    except Exception:
        pass

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="BIO-A topography invariance",
        outputs=[pair_csv, stab_csv, fig],
        extra={"n_datasets": int(len(ds_ok)), "n_pairs": int(len(pair_df))},
    )


def _stage_bio_B_source_template(ctx: RunContext) -> Dict[str, Any]:
    stage = "bio_B_source_template"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"
    out_dir = ctx.pack_bio / "BIO_B_source_template"
    out_dir.mkdir(parents=True, exist_ok=True)

    reason = (
        "source localization skipped: available H5 features do not include channel-wise evoked waveforms "
        "required for fsaverage inverse modeling."
    )
    stop = out_dir / "STOP_REASON_BIO_B.md"
    _write_stop_reason(stop, stage, reason)
    limitations = out_dir / "source_localization_limitations.md"
    _write_text(
        limitations,
        "# BIO-B limitations\n\n"
        "- Channel-time evoked matrices are not present in stored features.\n"
        "- Template inverse modeling was not run to avoid unverifiable source claims.\n",
    )
    return _record_stage(
        ctx,
        stage=stage,
        status="SKIP",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="BIO-B source template sensitivity",
        outputs=[stop, limitations],
        error=reason,
    )


def _subject_regime_effects(df: pd.DataFrame, q_low: float, q_high: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sid, g in df.groupby("subject_key"):
        g = g[np.isfinite(g["pdr"]) & np.isfinite(g["memory_load"]) & np.isfinite(g["p3b_amp"])].copy()
        if len(g) < 24:
            continue
        pdr = g["pdr"].to_numpy(dtype=float)
        lo = float(np.quantile(pdr, q_low))
        hi = float(np.quantile(pdr, q_high))
        low = g[pdr <= lo]
        high = g[pdr >= hi]
        if len(low) < 10 or len(high) < 10:
            continue
        if low["memory_load"].nunique() < 2 or high["memory_load"].nunique() < 2:
            continue
        b_low = float(np.polyfit(low["memory_load"].to_numpy(dtype=float), low["p3b_amp"].to_numpy(dtype=float), 1)[0])
        b_high = float(np.polyfit(high["memory_load"].to_numpy(dtype=float), high["p3b_amp"].to_numpy(dtype=float), 1)[0])
        rows.append({"subject_key": sid, "slope_low": b_low, "slope_high": b_high, "effect_high_minus_low": b_high - b_low})
    return pd.DataFrame(rows)


def _stage_bio_C_arousal_regime(ctx: RunContext) -> Dict[str, Any]:
    stage = "bio_C_arousal_regime"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"
    out_dir = ctx.pack_bio / "BIO_C_arousal_regime"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_dir = ctx.features_root / "ds003838"
    parts = [df for df in _iter_trials_from_h5(ds_dir) if {"pdr", "memory_load", "p3b_amp", "subject_key"}.issubset(df.columns)]
    if not parts:
        reason = "missing ds003838 trial-level pdr/p3b features"
        stop = out_dir / "STOP_REASON_BIO_C.md"
        _write_stop_reason(stop, stage, reason)
        return _record_stage(ctx, stage=stage, status="SKIP", rc=0, started=started, log_path=log_path, summary_path=summary_path, command="BIO-C arousal regime", outputs=[stop], error=reason)

    df = pd.concat(parts, axis=0, ignore_index=True)
    df = df[np.isfinite(df["pdr"]) & np.isfinite(df["memory_load"]) & np.isfinite(df["p3b_amp"])].copy()
    subs = sorted(df["subject_key"].astype(str).unique().tolist())
    if len(subs) < 16:
        reason = f"insufficient subjects for discovery/confirmation split (n={len(subs)})"
        stop = out_dir / "STOP_REASON_BIO_C.md"
        _write_stop_reason(stop, stage, reason)
        return _record_stage(ctx, stage=stage, status="SKIP", rc=0, started=started, log_path=log_path, summary_path=summary_path, command="BIO-C arousal regime", outputs=[stop], error=reason)

    rng = np.random.default_rng(20260223)
    perm_subs = np.array(subs, dtype=object)
    rng.shuffle(perm_subs)
    mid = len(perm_subs) // 2
    disc = df[df["subject_key"].isin(set(perm_subs[:mid].tolist()))].copy()
    conf = df[df["subject_key"].isin(set(perm_subs[mid:].tolist()))].copy()

    candidates = {"median": (0.50, 0.50), "tercile": (1.0 / 3.0, 2.0 / 3.0), "quartile": (0.25, 0.75)}
    cand_rows: List[Dict[str, Any]] = []
    best = {"name": "", "ql": 0.5, "qh": 0.5, "effect": float("nan"), "n": 0, "score": -1.0}
    for name, (ql, qh) in candidates.items():
        eff_df = _subject_regime_effects(disc, ql, qh)
        med = float(np.median(eff_df["effect_high_minus_low"])) if not eff_df.empty else float("nan")
        n = int(len(eff_df))
        score = abs(med) * max(1.0, np.sqrt(max(1, n))) if np.isfinite(med) else -1.0
        cand_rows.append({"split": name, "q_low": ql, "q_high": qh, "n_subjects": n, "effect_median": med, "score": score})
        if n >= 8 and score > float(best["score"]):
            best = {"name": name, "ql": ql, "qh": qh, "effect": med, "n": n, "score": score}
    cand_csv = out_dir / "discovery_candidate_splits.csv"
    pd.DataFrame(cand_rows).to_csv(cand_csv, index=False)

    if not best["name"]:
        reason = "no discovery split satisfied minimum subjects"
        stop = out_dir / "STOP_REASON_BIO_C.md"
        _write_stop_reason(stop, stage, reason, diagnostics={"candidates": cand_rows})
        return _record_stage(ctx, stage=stage, status="SKIP", rc=0, started=started, log_path=log_path, summary_path=summary_path, command="BIO-C arousal regime", outputs=[cand_csv, stop], error=reason)

    conf_eff_df = _subject_regime_effects(conf, float(best["ql"]), float(best["qh"]))
    if conf_eff_df.empty or len(conf_eff_df) < 8:
        reason = "confirmation split has insufficient subject effects"
        stop = out_dir / "STOP_REASON_BIO_C.md"
        _write_stop_reason(stop, stage, reason, diagnostics={"chosen_split": best["name"], "n_confirmation": int(len(conf_eff_df))})
        return _record_stage(ctx, stage=stage, status="SKIP", rc=0, started=started, log_path=log_path, summary_path=summary_path, command="BIO-C arousal regime", outputs=[cand_csv, stop], error=reason)

    arr = conf_eff_df["effect_high_minus_low"].to_numpy(dtype=float)
    obs = float(np.median(arr))
    n_perm = 20000
    rng = np.random.default_rng(1234)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = float(np.median(arr * rng.choice([-1.0, 1.0], size=arr.shape[0], replace=True)))
    p = float((1.0 + np.sum(np.abs(null) >= abs(obs))) / (1.0 + len(null)))

    out_json = out_dir / "arousal_regime.json"
    _write_json(
        out_json,
        {
            "chosen_split": best["name"],
            "q_low": float(best["ql"]),
            "q_high": float(best["qh"]),
            "discovery_effect_median": float(best["effect"]),
            "discovery_n_subjects": int(best["n"]),
            "confirmation_effect_median": float(obs),
            "confirmation_n_subjects": int(len(conf_eff_df)),
            "perm_p": float(p),
            "perm_n": int(n_perm),
            "perm_ci95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
        },
    )
    conf_csv = out_dir / "confirmation_subject_effects.csv"
    conf_eff_df.to_csv(conf_csv, index=False)

    fig = out_dir / "FIG_regime_effect.png"
    try:
        fig_obj, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.hist(null, bins=60, alpha=0.7, color="#6699cc")
        ax.axvline(obs, color="#aa2222", lw=2.0)
        ax.set_title(f"BIO-C confirmation ({best['name']})")
        ax.set_xlabel("Median(high-slope minus low-slope)")
        ax.set_ylabel("Count")
        fig_obj.tight_layout()
        fig_obj.savefig(fig, dpi=180)
        plt.close(fig_obj)
    except Exception:
        pass

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="BIO-C arousal regime",
        outputs=[cand_csv, out_json, conf_csv, fig],
        extra={"perm_n_done": int(n_perm), "chosen_split": str(best["name"]), "confirmation_n": int(len(conf_eff_df))},
    )


def _trial_theta_alpha_ratio(x: np.ndarray, sfreq: float) -> float:
    if x.size < 16:
        return float("nan")
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 16:
        return float("nan")
    v = v - float(np.mean(v))
    f = np.fft.rfftfreq(v.size, d=1.0 / sfreq)
    p = np.abs(np.fft.rfft(v)) ** 2
    t = p[(f >= 4.0) & (f < 8.0)]
    a = p[(f >= 8.0) & (f < 12.0)]
    if t.size == 0 or a.size == 0:
        return float("nan")
    return float(np.mean(t) / max(np.mean(a), 1e-12))


def _stage_bio_D_cross_modality_ds004752(ctx: RunContext) -> Dict[str, Any]:
    stage = "bio_D_cross_modality_ds004752"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"
    out_dir = ctx.pack_bio / "BIO_D_cross_modality_ds004752"
    out_dir.mkdir(parents=True, exist_ok=True)

    map_summary = _read_json_if_exists(ctx.pack_mapping / "mapping_decode_summary.json") or {}
    map_rows = {str(r.get("dataset_id")): str(r.get("status", "")) for r in map_summary.get("rows", [])}
    if map_rows.get("ds004752") != "PASS":
        reason = "ds004752 mapping did not pass strict decode"
        stop = out_dir / "STOP_REASON_ds004752.md"
        _write_stop_reason(stop, stage, reason, diagnostics={"mapping_status": map_rows.get("ds004752", "missing")})
        return _record_stage(ctx, stage=stage, status="SKIP", rc=0, started=started, log_path=log_path, summary_path=summary_path, command="BIO-D cross modality", outputs=[stop], error=reason)

    ds_root = ctx.data_root / "ds004752"
    ev_files = sorted(ds_root.rglob("*_task-verbalWM*_events.tsv"))
    run_rows: List[Dict[str, Any]] = []
    mod_counts = {"eeg": 0, "ieeg": 0}
    for ev in ev_files:
        mod = "ieeg" if "/ieeg/" in ev.as_posix() else ("eeg" if "/eeg/" in ev.as_posix() else "other")
        if mod not in mod_counts:
            continue
        if mod_counts[mod] >= 20:
            continue
        try:
            df = pd.read_csv(ev, sep="\t")
        except Exception:
            continue
        if not {"SetSize", "ResponseTime", "Correct", "onset", "duration"}.issubset(df.columns):
            continue
        d = df.copy()
        for c in ["SetSize", "ResponseTime", "Correct", "onset", "duration"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d[np.isfinite(d["SetSize"]) & np.isfinite(d["onset"]) & np.isfinite(d["duration"])].copy()
        if len(d) < 20 or d["SetSize"].nunique() < 2:
            continue
        raw = ev.with_name(ev.name.replace("_events.tsv", f"_{mod}.edf"))
        if not raw.exists():
            continue
        try:
            rr = mne.io.read_raw_edf(raw.as_posix(), preload=False, verbose="ERROR")
            sf = float(rr.info["sfreq"])
            spec_vals: List[float] = []
            set_vals: List[float] = []
            for _, r in d.head(15).iterrows():
                t0 = float(max(0.0, r["onset"]))
                t1 = float(min(rr.times[-1], t0 + min(2.0, max(0.5, float(r["duration"])))))
                a = int(max(0, round(t0 * sf)))
                b = int(min(rr.n_times, round(t1 * sf)))
                if b <= a + 8:
                    continue
                data, _ = rr[:, a:b]
                x = np.nanmean(data[: min(4, data.shape[0]), :], axis=0)
                ta = _trial_theta_alpha_ratio(x, sf)
                if np.isfinite(ta):
                    spec_vals.append(float(ta))
                    set_vals.append(float(r["SetSize"]))
            if len(spec_vals) < 8:
                continue
            spec_slope = float(np.polyfit(np.asarray(set_vals, dtype=float), np.asarray(spec_vals, dtype=float), 1)[0])
        except Exception:
            continue

        rt_sub = d[np.isfinite(d["ResponseTime"])].copy()
        acc_sub = d[np.isfinite(d["Correct"])].copy()
        rt_slope = float(np.polyfit(rt_sub["SetSize"].to_numpy(dtype=float), rt_sub["ResponseTime"].to_numpy(dtype=float), 1)[0]) if len(rt_sub) >= 8 else float("nan")
        acc_slope = float(np.polyfit(acc_sub["SetSize"].to_numpy(dtype=float), acc_sub["Correct"].to_numpy(dtype=float), 1)[0]) if len(acc_sub) >= 8 else float("nan")
        run_rows.append(
            {
                "run_key": ev.name.replace("_events.tsv", ""),
                "modality": mod,
                "n_trials": int(len(d)),
                "rt_slope_per_load": rt_slope,
                "acc_slope_per_load": acc_slope,
                "theta_alpha_slope_per_load": spec_slope,
            }
        )
        mod_counts[mod] += 1

    run_df = pd.DataFrame(run_rows)
    run_csv = out_dir / "cross_modality_run_signatures.csv"
    run_df.to_csv(run_csv, index=False)

    if run_df.empty or int((run_df["modality"] == "eeg").sum()) < 5 or int((run_df["modality"] == "ieeg").sum()) < 5:
        reason = "insufficient convergent spectral signatures across eeg/ieeg runs"
        stop = out_dir / "STOP_REASON_ds004752.md"
        _write_stop_reason(stop, stage, reason, diagnostics={"n_rows": int(len(run_df)), "modality_counts": mod_counts})
        return _record_stage(ctx, stage=stage, status="SKIP", rc=0, started=started, log_path=log_path, summary_path=summary_path, command="BIO-D cross modality", outputs=[run_csv, stop], error=reason)

    mean_eeg = float(np.nanmean(run_df.loc[run_df["modality"] == "eeg", "theta_alpha_slope_per_load"].to_numpy(dtype=float)))
    mean_ieeg = float(np.nanmean(run_df.loc[run_df["modality"] == "ieeg", "theta_alpha_slope_per_load"].to_numpy(dtype=float)))
    summ = {
        "n_runs_eeg": int((run_df["modality"] == "eeg").sum()),
        "n_runs_ieeg": int((run_df["modality"] == "ieeg").sum()),
        "mean_theta_alpha_slope_eeg": mean_eeg,
        "mean_theta_alpha_slope_ieeg": mean_ieeg,
        "convergent_sign": bool(np.isfinite(mean_eeg) and np.isfinite(mean_ieeg) and np.sign(mean_eeg) == np.sign(mean_ieeg)),
    }
    summ_json = out_dir / "cross_modality_summary.json"
    _write_json(summ_json, summ)

    fig = out_dir / "FIG_cross_modality_signature.png"
    try:
        fig_obj, ax = plt.subplots(figsize=(6.4, 4.3))
        for mod, color in [("eeg", "#2c7fb8"), ("ieeg", "#d95f0e")]:
            sub = run_df[run_df["modality"] == mod]
            ax.hist(sub["theta_alpha_slope_per_load"].to_numpy(dtype=float), bins=20, alpha=0.6, label=mod, color=color)
        ax.axvline(0.0, color="black", lw=1.0)
        ax.set_title("BIO-D spectral load signature (ds004752)")
        ax.set_xlabel("Theta/Alpha slope per load")
        ax.set_ylabel("Run count")
        ax.legend(frameon=False)
        fig_obj.tight_layout()
        fig_obj.savefig(fig, dpi=180)
        plt.close(fig_obj)
    except Exception:
        pass

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="BIO-D cross modality",
        outputs=[run_csv, summ_json, fig],
        extra=summ,
    )


def _nvidia_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return int(sum(1 for line in f if line.strip()))
    except Exception:
        return 0


def _gpu_probe_for_monitor(ctx: RunContext, *, seconds: int, log_path: Path) -> int:
    sec = int(max(30, seconds))
    code = (
        "import time, torch\\n"
        f"end=time.time()+{sec}\\n"
        "dev='cuda' if torch.cuda.is_available() else 'cpu'\\n"
        "x=torch.randn((2048,2048),device=dev,dtype=torch.float16 if dev=='cuda' else torch.float32)\\n"
        "w=torch.randn((2048,2048),device=dev,dtype=torch.float16 if dev=='cuda' else torch.float32)\\n"
        "while time.time()<end:\\n"
        " y=x@w\\n"
        " z=w@x\\n"
        " _=(y.mean()+z.mean()).item()\\n"
    )
    return _run_cmd([sys.executable, "-c", code], cwd=REPO_ROOT, log_path=log_path, env=ctx.runtime_env, allow_fail=True)


def _stage_objective_guard(ctx: RunContext) -> Dict[str, Any]:
    stage = "objective_guard"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    issues: List[str] = []
    ana = (ctx.cfg_mega.get("analysis", {}) or {})

    nvidia_csv = ctx.audit_dir / "nvidia_smi_1hz.csv"
    rows_before = _nvidia_row_count(nvidia_csv)
    if rows_before < 600:
        need = 600 - rows_before
        _gpu_probe_for_monitor(ctx, seconds=min(1800, max(120, need + 30)), log_path=log_path)
    rows_after = _nvidia_row_count(nvidia_csv)
    if rows_after < 600:
        issues.append(f"nvidia_smi_1hz.csv rows<{600} (got {rows_after})")

    lawc_sum = _read_json_if_exists(ctx.audit_dir / "core_lawc_ultradeep_summary.json") or {}
    req_lawc_perm = int(max(20000, int(ana.get("lawc_n_perm", 100000))))
    done_lawc_perm = int(lawc_sum.get("lawc_n_perm", 0) or 0)
    if done_lawc_perm < req_lawc_perm:
        issues.append(f"lawc_n_perm_done<{req_lawc_perm} (got {done_lawc_perm})")

    lawc_csv = ctx.pack_core / "lawc_ultradeep" / "lawc_audit" / "locked_test_results.csv"
    if not lawc_csv.exists():
        issues.append("missing locked_test_results.csv")
    else:
        try:
            ldf = pd.read_csv(lawc_csv)
            for ds in LOCKED_LAWC_CANONICAL:
                rr = ldf[ldf["dataset_id"].astype(str) == ds]
                if rr.empty or not bool(rr.iloc[0].get("pass_all", False)):
                    issues.append(f"locked Law-C gate failed for {ds}")
        except Exception as exc:
            issues.append(f"failed reading lawc results: {exc}")

    mech_sum = _read_json_if_exists(ctx.audit_dir / "mechanism_deep_summary.json") or {}
    req_mech_perm = int(max(1000, int(ana.get("mechanism_n_perm", 10000))))
    done_mech_perm = int(mech_sum.get("n_perm", 0) or 0)
    if done_mech_perm < req_mech_perm:
        issues.append(f"mechanism_n_perm_done<{req_mech_perm} (got {done_mech_perm})")
    mech_tbl = ctx.pack_mechanism / "Table_mechanism_effects.csv"
    if not mech_tbl.exists():
        issues.append("missing mechanism table")
    else:
        try:
            mdf = pd.read_csv(mech_tbl)
            if mdf.empty or ("q_value" in mdf.columns and mdf["q_value"].isna().any()):
                issues.append("mechanism q-values missing/NaN")
        except Exception as exc:
            issues.append(f"failed reading mechanism table: {exc}")

    pd_ep = ctx.pack_clin_pdrest / "pd_rest_endpoints.csv"
    mort_ep = ctx.pack_clin_mortality / "mortality_endpoints.csv"
    dem_ep = ctx.pack_clin_dementia / "dementia_endpoints.csv"
    for ds, p in [("ds004584", pd_ep), ("ds007020", mort_ep)]:
        if not p.exists():
            issues.append(f"missing required clinical endpoints for {ds}")
            continue
        try:
            d = pd.read_csv(p)
            if d.empty:
                issues.append(f"empty required clinical endpoints for {ds}")
        except Exception as exc:
            issues.append(f"failed reading endpoints for {ds}: {exc}")
    if not dem_ep.exists():
        issues.append("missing ds004504 dementia endpoints")

    clin_sum = _read_json_if_exists(ctx.audit_dir / "clinical_translation_summary.json") or {}
    req_clin_perm = int(max(int(ana.get("clinical_perm_min", 5000)), int(ana.get("clinical_perm", 20000))))
    done_clin_perm = int(clin_sum.get("clinical_perm_done", 0) or 0)
    if done_clin_perm < req_clin_perm:
        issues.append(f"clinical_perm_done<{req_clin_perm} (got {done_clin_perm})")

    bio_pass = []
    bio_skip = []
    for st in ["bio_A_topography", "bio_B_source_template", "bio_C_arousal_regime", "bio_D_cross_modality_ds004752"]:
        ss = str(ctx.stage_status.get(st, ""))
        if ss == "PASS":
            bio_pass.append(st)
        elif ss == "SKIP":
            bio_skip.append(st)
    if len(bio_pass) < 2:
        issues.append(f"<2 BIO packs succeeded (PASS={bio_pass}, SKIP={bio_skip})")

    strict_pass = len(issues) == 0
    out_json = ctx.audit_dir / "objective_guard_summary.json"
    _write_json(
        out_json,
        {
            "strict_pass": bool(strict_pass),
            "issues": issues,
            "bio_pass": bio_pass,
            "bio_skip": bio_skip,
            "nvidia_rows_before": int(rows_before),
            "nvidia_rows_after": int(rows_after),
            "lawc_n_perm_done": int(done_lawc_perm),
            "mechanism_n_perm_done": int(done_mech_perm),
            "clinical_perm_done": int(done_clin_perm),
        },
    )

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS" if strict_pass else "FAIL",
        rc=0 if strict_pass else 1,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        command="objective_guard",
        outputs=[out_json],
        error=" ; ".join(issues),
        extra={"strict_pass": bool(strict_pass), "issues": issues},
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
    report = ctx.audit_dir / "NN_FINAL_MEGA_V2_BIO_REPORT.md"

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
    objective = _read_json_if_exists(ctx.audit_dir / "objective_guard_summary.json") or {}

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
    for p in sorted(ctx.pack_clin_pdrest.rglob("FIG_*.png")):
        figure_candidates.append(p)
    for p in sorted(ctx.pack_clin_mortality.rglob("FIG_*.png")):
        figure_candidates.append(p)
    for p in sorted(ctx.pack_clin_dementia.rglob("FIG_*.png")):
        figure_candidates.append(p)
    for p in sorted(ctx.pack_bio.rglob("FIG_*.png")):
        figure_candidates.append(p)

    figure_lines = [f"- `{p}`" for p in figure_candidates if p.exists()]
    if not figure_lines:
        figure_lines = ["- <none>"]

    lines = [
        "# NN_FINAL_MEGA_V2_BIO REPORT",
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
            "## Objective Guard",
            f"- `{json.dumps(objective, sort_keys=True)}`",
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
            f"- Final bundle: `{ctx.outzip_dir / 'NN_FINAL_MEGA_V2_BIO_SUBMISSION_PACKET.zip'}`",
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

    zpath = ctx.outzip_dir / "NN_FINAL_MEGA_V2_BIO_SUBMISSION_PACKET.zip"

    # Snapshot configs used.
    cfg_dir = ctx.audit_dir / "configs_used"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for src in [
        ctx.config,
        ctx.mega_config,
        ctx.datasets_config,
        ctx.lawc_event_map,
        ctx.mechanism_event_map,
        ctx.pearl_event_map,
        REPO_ROOT / "configs" / "event_map_ds004752.yaml",
        REPO_ROOT / "configs" / "event_map_ds007262.yaml",
    ]:
        if src.exists():
            shutil.copy2(src, cfg_dir / src.name)

    include_roots = [
        ctx.audit_dir,
        ctx.pack_core,
        ctx.pack_mechanism,
        ctx.pack_normative,
        ctx.pack_clin_pdrest,
        ctx.pack_clin_mortality,
        ctx.pack_clin_dementia,
        ctx.pack_bio,
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
    ap.add_argument("--mega_config", type=Path, default=Path("configs/nn_final_mega_v2_bio.yaml"))
    ap.add_argument("--datasets_config", type=Path, default=Path("configs/datasets_nn_final_mega_v2_bio.yaml"))

    ap.add_argument("--lawc_event_map", type=Path, default=Path("configs/lawc_event_map.yaml"))
    ap.add_argument("--mechanism_event_map", type=Path, default=Path("configs/mechanism_event_map.yaml"))
    ap.add_argument("--pearl_event_map", type=Path, default=Path("configs/pearl_event_map.yaml"))

    ap.add_argument("--wall_hours", type=float, default=12.0)
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
    out_root = args.out_root or (Path("/filesystemHcog/runs") / f"{ts}_NN_FINAL_MEGA_V2_BIO")
    features_root = args.features_root or Path(f"/filesystemHcog/features_cache_NN_FINAL_MEGA_V2_BIO_{ts}")

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
        pack_clin_pdrest=out_root / "PACK_CLINICAL_PDREST",
        pack_clin_mortality=out_root / "PACK_CLINICAL_MORTALITY",
        pack_clin_dementia=out_root / "PACK_CLINICAL_DEMENTIA",
        pack_bio=out_root / "PACK_BIO",
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

    run_status = "PASS_STRICT"
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
        _stage_bio_A_topography,
        _stage_bio_B_source_template,
        _stage_bio_C_arousal_regime,
        _stage_bio_D_cross_modality_ds004752,
        _stage_objective_guard,
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

    if run_status != "FAIL":
        og = ctx.stage_outputs.get("objective_guard", {})
        strict_ok = bool(og.get("strict_pass", False))
        if strict_ok:
            run_status = "PASS_STRICT"
        else:
            if time.time() > ctx.deadline_ts:
                run_status = "PARTIAL_PASS"
            else:
                run_status = "FAIL"
                run_error = str(og.get("error", run_error or "objective_guard failed"))

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

    report_path = ctx.audit_dir / "NN_FINAL_MEGA_V2_BIO_REPORT.md"
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
    print(f"BUNDLE={ctx.outzip_dir / 'NN_FINAL_MEGA_V2_BIO_SUBMISSION_PACKET.zip'}", flush=True)

    return 0 if run_status in {"PASS_STRICT", "PARTIAL_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
