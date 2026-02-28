#!/usr/bin/env python3
"""Overnight reviewer-proof orchestration for FIX2 evaluation pack."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from aggregate_results import _compute_repo_fingerprint
except Exception:
    _compute_repo_fingerprint = None


STAGE_NAMES = (
    "preflight",
    "compile_gate",
    "inspect_events",
    "lawc_deep",
    "lawc_sensitivity",
    "rt_linkage",
    "module04_manyseed",
    "plot_seed_stability",
    "zip_bundle",
)


class RunContext:
    def __init__(
        self,
        *,
        repo_root: Path,
        out_root: Path,
        features_root: Path,
        data_root: Path,
        datasets: List[str],
        event_map: Path,
        config: Path,
        wall_hours: float,
        lawc_n_perm: int,
        rtlink_n_perm: int,
        seeds: List[int],
        gpu_parallel_procs: int,
        cpu_workers: int,
        resume: bool,
    ) -> None:
        self.repo_root = repo_root
        self.out_root = out_root
        self.audit_dir = out_root / "AUDIT"
        self.log_dir = self.audit_dir / "logs"
        self.outzip_dir = out_root / "OUTZIP"
        self.features_root = features_root
        self.data_root = data_root
        self.datasets = datasets
        self.event_map = event_map
        self.config = config
        self.wall_hours = wall_hours
        self.lawc_n_perm = int(lawc_n_perm)
        self.rtlink_n_perm = int(rtlink_n_perm)
        self.seeds = list(seeds)
        self.gpu_parallel_procs = int(max(1, gpu_parallel_procs))
        self.cpu_workers = int(max(1, cpu_workers))
        self.resume = bool(resume)

        self.start_ts = time.time()
        self.deadline = self.start_ts + float(self.wall_hours) * 3600.0

        self.stage_records: List[Dict[str, Any]] = []
        self.stop_reason_path = self.audit_dir / "STOP_REASON.md"

        self.monitor_proc: Optional[subprocess.Popen] = None
        self.monitor_log = self.audit_dir / "nvidia_smi_1hz.csv"

        self.failed_stage: Optional[str] = None


# ------------------------- generic helpers ----------------------------------

def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_list(v: Optional[Sequence[Any]]) -> List[Any]:
    return list(v) if v is not None else []


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _tail_lines(path: Path, n: int = 120) -> str:
    raw = _read_file(path)
    if not raw:
        return ""
    lines = raw.splitlines()
    return "\n".join(lines[-max(1, int(n)) :])


def _fmt_num(v: Any, nd: int = 6) -> str:
    try:
        fv = float(v)
    except Exception:
        return "nan"
    if not np.isfinite(fv):
        return "nan"
    return f"{fv:.{nd}g}"


def _parse_seeds(spec: str) -> List[int]:
    out: List[int] = []
    for raw in str(spec).split(","):
        t = str(raw).strip()
        if not t:
            continue
        if "-" in t:
            a, b = t.split("-", 1)
            a_i = int(a.strip())
            b_i = int(b.strip())
            if a_i <= b_i:
                out.extend(list(range(a_i, b_i + 1)))
            else:
                out.extend(list(range(a_i, b_i - 1, -1)))
        else:
            out.append(int(t))
    dedup = sorted(set(out))
    return dedup


def _seed_done(seed_root: Path, seed: int) -> bool:
    run_id = f"seed_{seed}"
    metric = seed_root / f"seed_{seed}" / "reports" / "normative" / run_id / "normative_metrics.json"
    return metric.exists()


def _run_cmd(
    cmd: List[str],
    *,
    log_path: Path,
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
    shell: bool = False,
) -> int:
    """Run a command and append stdout/stderr to log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{_iso_now()}] CMD: {' '.join(cmd) if isinstance(cmd, list) else cmd}\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            shell=shell,
            text=True,
            check=False,
        )
        return int(proc.returncode)


def _write_stage_marker(ctx: RunContext, name: str, status: str, summary_path: Path, log_path: Path) -> None:
    (ctx.audit_dir / f"{name}.status").write_text(f"{status}\n", encoding="utf-8")


def _record_stage(
    ctx: RunContext,
    name: str,
    status: str,
    started: float,
    log_path: Path,
    summary_path: Optional[Path],
    rc: Optional[int] = None,
    command: Optional[str] = None,
    error: Optional[str] = None,
    outputs: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    ended = time.time()
    record = {
        "name": name,
        "status": status,
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "ended_at": datetime.fromtimestamp(ended, tz=timezone.utc).isoformat(),
        "elapsed_sec": float(max(0.0, ended - started)),
        "returncode": rc,
        "log": str(log_path),
        "summary": str(summary_path) if summary_path is not None else "",
        "command": command or "",
        "error": error or "",
        "outputs": [str(x) for x in _as_list(outputs)],
    }
    _write_stage_marker(ctx, name, status, Path(record["summary"]), log_path)

    if summary_path is not None:
        _write_json(summary_path, record)

    ctx.stage_records.append(record)
    return record


def _start_monitor(ctx: RunContext) -> None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
        "-l",
        "1",
    ]
    try:
        handle = ctx.monitor_log.open("w", encoding="utf-8")
        ctx.monitor_proc = subprocess.Popen(cmd, stdout=handle, stderr=handle, text=True, cwd=str(ctx.repo_root))
    except Exception as exc:
        raise RuntimeError(f"Failed to start 1Hz GPU monitor to {ctx.monitor_log}: {exc}")


def _stop_monitor(ctx: RunContext) -> None:
    proc = ctx.monitor_proc
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    finally:
        ctx.monitor_proc = None


# -------------------------- stages -----------------------------------------

def _stage_preflight(ctx: RunContext) -> Dict[str, Any]:
    name = "preflight"
    started = time.time()
    log_path = ctx.audit_dir / "preflight.log"
    summary_path = ctx.audit_dir / "preflight_summary.json"
    outputs: List[Path] = []

    errors: List[str] = []
    try:
        if not ctx.data_root.exists():
            raise RuntimeError(f"data_root does not exist: {ctx.data_root}")
        if not ctx.features_root.exists():
            raise RuntimeError(f"features_root does not exist: {ctx.features_root}")

        ctx.out_root.mkdir(parents=True, exist_ok=True)
        ctx.audit_dir.mkdir(parents=True, exist_ok=True)
        ctx.log_dir.mkdir(parents=True, exist_ok=True)
        ctx.outzip_dir.mkdir(parents=True, exist_ok=True)

        preflight = {
            "timestamp_utc": _iso_now(),
            "wall_hours": ctx.wall_hours,
            "features_root": str(ctx.features_root),
            "data_root": str(ctx.data_root),
            "datasets": ctx.datasets,
            "config": str(ctx.config),
            "event_map": str(ctx.event_map),
            "seed_set": ctx.seeds,
            "gpu_parallel_procs": ctx.gpu_parallel_procs,
            "cpu_workers": ctx.cpu_workers,
            "repo_root": str(ctx.repo_root),
        }

        preflight["python_version"] = sys.version.replace("\n", " ")

        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{_iso_now()}] preflight start\n")

        # torch / cuda snapshot
        try:
            import torch

            preflight["torch_version"] = str(getattr(torch, "__version__", "<missing>"))
            preflight["torch_cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                preflight["torch_cuda_version"] = str(getattr(torch.version, "cuda", "<missing>"))
                preflight["cuda_device"] = torch.cuda.get_device_name(0)
            else:
                preflight["torch_cuda_version"] = "<unavailable>"
                preflight["cuda_device"] = "<unavailable>"
        except Exception as exc:
            preflight["torch_version"] = "<unavailable>"
            preflight["torch_cuda_available"] = False
            preflight["torch_import_error"] = str(exc)

        # pip / nvidia snapshot artifacts
        pip_freeze = ctx.audit_dir / "pip_freeze.txt"
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                check=False,
                cwd=str(ctx.repo_root),
            )
            pip_freeze.write_text(r.stdout if r.returncode == 0 else r.stdout + "\n" + r.stderr, encoding="utf-8")
            preflight["pip_freeze_path"] = str(pip_freeze)
            outputs.append(pip_freeze)
        except Exception as exc:
            errors.append(f"pip freeze failed: {exc}")

        nvidia_l = ctx.audit_dir / "nvidia_smi_L.txt"
        try:
            r = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                check=False,
                cwd=str(ctx.repo_root),
            )
            nvidia_l.write_text((r.stdout if r.returncode == 0 else r.stdout + "\n" + r.stderr), encoding="utf-8")
            preflight["nvidia_smi_L_path"] = str(nvidia_l)
            outputs.append(nvidia_l)
        except Exception as exc:
            errors.append(f"nvidia-smi -L failed: {exc}")

        nvidia_snapshot = ctx.audit_dir / "nvidia_smi_snapshot.csv"
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=str(ctx.repo_root),
            )
            nvidia_snapshot.write_text((r.stdout if r.returncode == 0 else r.stdout + "\n" + r.stderr), encoding="utf-8")
            preflight["nvidia_smi_snapshot_path"] = str(nvidia_snapshot)
            outputs.append(nvidia_snapshot)
        except Exception as exc:
            errors.append(f"nvidia-smi snapshot failed: {exc}")

        # repo fingerprint
        if _compute_repo_fingerprint is not None:
            preflight["repo_fingerprint"] = _compute_repo_fingerprint(ctx.repo_root)
        else:
            preflight["repo_fingerprint"] = {
                "repo_root": str(ctx.repo_root),
                "note": "_compute_repo_fingerprint unavailable in aggregate_results",
            }
        repo_fp = ctx.audit_dir / "repo_fingerprint.json"
        _write_json(repo_fp, preflight["repo_fingerprint"])
        outputs.append(repo_fp)

        if errors:
            raise RuntimeError("; ".join(errors))

        # start the 1Hz monitor now
        _start_monitor(ctx)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{_iso_now()}] preflight complete\n")

        return _record_stage(
            ctx,
            name=name,
            status="PASS",
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            rc=0,
            command="preflight",
            outputs=outputs,
        )
    except Exception as exc:
        _stop_monitor(ctx)
        return _record_stage(
            ctx,
            name=name,
            status="FAIL",
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            rc=1,
            command="preflight",
            error=str(exc),
            outputs=outputs,
        )


def _stage_compile_gate(ctx: RunContext) -> Dict[str, Any]:
    name = "compile_gate"
    started = time.time()
    log_path = ctx.audit_dir / "compile_gate.log"
    summary_path = ctx.audit_dir / "compile_gate_summary.json"
    cmd = "find . -name '*.py' -print0 | xargs -0 python -m py_compile"
    rc = _run_cmd(["bash", "-lc", cmd], log_path=log_path, cwd=ctx.repo_root, shell=False)
    status = "PASS" if rc == 0 else "FAIL"
    return _record_stage(
        ctx,
        name=name,
        status=status,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        rc=rc,
        command=cmd,
        outputs=[ctx.repo_root],
    )


def _stage_inspect_events(ctx: RunContext) -> Dict[str, Any]:
    name = "inspect_events"
    started = time.time()
    log_path = ctx.audit_dir / "inspect_events.log"
    summary_path = ctx.audit_dir / "inspect_events_summary.json"
    out_dir = ctx.audit_dir
    cmd = [
        sys.executable,
        "scripts/inspect_events.py",
        "--data_root",
        str(ctx.data_root),
        "--event_map",
        str(ctx.event_map),
        "--datasets",
        ",".join(ctx.datasets),
        "--out_dir",
        str(out_dir),
    ]
    rc = _run_cmd(cmd, log_path=log_path, cwd=ctx.repo_root)
    status = "PASS" if rc == 0 else "FAIL"
    outputs = sorted(out_dir.glob("events_inspection_*.md"))
    record = _record_stage(
        ctx,
        name=name,
        status=status,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        rc=rc,
        command=" ".join(cmd),
        outputs=outputs,
        error=None if rc == 0 else _read_file(log_path).splitlines()[-20:][0] if out_dir.exists() else "inspect failed",
    )

    if rc != 0:
        fail_text = _read_file(log_path)
        summary = {"status": "FAIL", "reason": fail_text, "stage": name, "report": str(out_dir)}
        _write_json(summary_path, summary)
        if "fail-closed" in fail_text.lower() or "Events inspection fail-closed errors" in fail_text:
            _write_stop_reason(ctx, name, fail_text, fail_text)
    else:
        _write_json(summary_path, {"status": "PASS", "stage": name, "outputs": [str(x) for x in outputs]})

    return record


def _stage_lawc_deep(ctx: RunContext) -> Dict[str, Any]:
    name = "lawc_deep"
    started = time.time()
    log_path = ctx.audit_dir / "lawc_deep.log"
    summary_path = ctx.audit_dir / "lawc_deep_summary.json"

    out_root = ctx.out_root / "lawc_audit_deep"
    cmd = [
        sys.executable,
        "05_audit_lawc.py",
        "--features_root",
        str(ctx.features_root),
        "--out_root",
        str(out_root),
        "--event_map",
        str(ctx.event_map),
        "--datasets",
        ",".join(ctx.datasets),
        "--n_perm",
        str(ctx.lawc_n_perm),
        "--workers",
        str(ctx.cpu_workers),
    ]
    rc = _run_cmd(cmd, log_path=log_path, cwd=ctx.repo_root)

    outputs: List[Path] = []
    if rc == 0:
        src_dir = out_root / "lawc_audit"
        if src_dir.exists():
            for fname in ["locked_test_results.json", "locked_test_results.csv", "negative_controls.csv"]:
                src = src_dir / fname
                if src.exists():
                    dst = out_root / fname
                    shutil.copy2(src, dst)
                    outputs.append(dst)
        audit_md = out_root / "AUDIT" / "LawcAudit.md"
        if audit_md.exists():
            dst = out_root / "LawcAudit_DEEP.md"
            shutil.copy2(audit_md, dst)
            outputs.append(dst)

    status = "PASS" if rc == 0 else "FAIL"
    payload: Dict[str, Any]
    if (out_root / "locked_test_results.json").exists():
        try:
            payload = json.loads((out_root / "locked_test_results.json").read_text(encoding="utf-8"))
        except Exception as exc:
            payload = {"pass": False, "status_reason": f"failed to parse locked_test_results.json: {exc}"}
    elif rc == 0:
        payload = {"pass": False, "status_reason": "missing locked_test_results.json"}
    else:
        payload = {"pass": False, "status_reason": "law-c command failed"}

    if rc != 0:
        status = "FAIL"
    elif not bool(payload.get("pass", False)):
        status = "FAIL"
        reason = str(payload.get("status_reason", "Law-C deep audit did not pass all datasets"))
        if not reason:
            reason = "Law-C deep audit did not pass all datasets"
        _write_stop_reason(ctx, name, "Law-C deep audit failed", reason)
    if rc != 0:
        err = _read_file(log_path)
        if "fail-closed" in err.lower() or "law-c audit failure" in err.lower():
            _write_stop_reason(ctx, name, "Law-C deep audit failed", err)

    summary = {
        "status": status,
        "command": " ".join(cmd),
        "returncode": rc,
        "lawc_audit_deep": str(out_root),
        "payload_status": payload.get("pass", False) if isinstance(payload, dict) else False,
        "datasets": [d.get("dataset_id") for d in payload.get("datasets", [])] if isinstance(payload, dict) else [],
        "outputs": [str(x) for x in outputs],
    }
    _write_json(summary_path, summary)

    return _record_stage(
        ctx,
        name=name,
        status=status,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        rc=rc,
        command=" ".join(cmd),
        outputs=outputs,
        error=summary.get("error") if status != "PASS" else None,
    )


def _stage_lawc_sensitivity(ctx: RunContext) -> Dict[str, Any]:
    name = "lawc_sensitivity"
    started = time.time()
    log_path = ctx.audit_dir / "lawc_sensitivity.log"
    summary_path = ctx.audit_dir / "lawc_sensitivity_summary.json"
    out_dir = ctx.out_root / "lawc_sensitivity"

    cmd = [
        sys.executable,
        "scripts/lawc_sensitivity.py",
        "--features_root",
        str(ctx.features_root),
        "--data_root",
        str(ctx.data_root),
        "--event_map",
        str(ctx.event_map),
        "--datasets",
        ",".join(ctx.datasets),
        "--out_dir",
        str(out_dir),
    ]
    rc = _run_cmd(cmd, log_path=log_path, cwd=ctx.repo_root)
    status = "PASS" if rc == 0 else "FAIL"

    outputs = sorted(out_dir.glob("*")) if out_dir.exists() else []

    if rc != 0:
        err = _read_file(log_path)
        summary = {
            "status": status,
            "returncode": rc,
            "command": " ".join(cmd),
            "error": err.splitlines()[-1] if err else "runner_error",
        }
    else:
        summary = {
            "status": status,
            "returncode": rc,
            "command": " ".join(cmd),
            "outputs": [str(x) for x in outputs],
        }

    _write_json(summary_path, summary)
    return _record_stage(
        ctx,
        name=name,
        status=status,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        rc=rc,
        command=" ".join(cmd),
        outputs=outputs,
        error=summary.get("error") if status == "FAIL" else None,
    )


def _stage_rt_linkage(ctx: RunContext) -> Dict[str, Any]:
    name = "rt_linkage"
    started = time.time()
    log_path = ctx.audit_dir / "rt_linkage.log"
    summary_path = ctx.audit_dir / "rt_linkage_summary.json"
    out_dir = ctx.out_root / "rt_linkage_audit"

    cmd = [
        sys.executable,
        "scripts/rt_linkage_audit.py",
        "--features_root",
        str(ctx.features_root),
        "--datasets",
        ",".join(ctx.datasets),
        "--out_dir",
        str(out_dir),
        "--n_perm",
        str(ctx.rtlink_n_perm),
        "--seed",
        "1234",
        "--workers",
        str(ctx.cpu_workers),
    ]
    rc = _run_cmd(cmd, log_path=log_path, cwd=ctx.repo_root)
    status = "PASS" if rc == 0 else "FAIL"
    outputs = sorted(out_dir.glob("*")) if out_dir.exists() else []

    if rc != 0:
        err = _read_file(log_path)
        summary = {
            "status": status,
            "returncode": rc,
            "command": " ".join(cmd),
            "error": err,
        }
    else:
        summary = {
            "status": status,
            "returncode": rc,
            "command": " ".join(cmd),
            "outputs": [str(x) for x in outputs],
        }

    _write_json(summary_path, summary)
    return _record_stage(
        ctx,
        name=name,
        status=status,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        rc=rc,
        command=" ".join(cmd),
        outputs=outputs,
        error=summary.get("error") if status == "FAIL" else None,
    )


def _seed_worker_cmd(ctx: RunContext, seed: int, seed_log: Path) -> List[str]:
    return [
        "bash",
        "scripts/run_module.sh",
        "--module",
        "04",
        "--features_root",
        str(ctx.features_root),
        "--out_root",
        str(ctx.out_root / "module04_manyseed"),
        "--config",
        str(ctx.config),
        "--seeds",
        str(seed),
        "--healthy_cohort",
        "healthy",
        "--clinical_cohort",
        "clinical",
        "--healthy_dataset_ids",
        ",".join(ctx.datasets),
        "--gpu_log_csv",
        str(ctx.out_root / "module04_manyseed" / "gpu_util.csv"),
        "--gpu_log_tag",
        f"module04_seed{seed}",
    ]


def _run_seed_process(ctx: RunContext, seed: int, module_root: Path, env: Dict[str, str]) -> subprocess.Popen:
    seed_cmd = [
        "bash",
        "scripts/run_module.sh",
        "--module",
        "04",
        "--features_root",
        str(ctx.features_root),
        "--out_root",
        str(module_root),
        "--config",
        str(ctx.config),
        "--seeds",
        str(seed),
        "--healthy_cohort",
        "healthy",
        "--clinical_cohort",
        "clinical",
        "--healthy_dataset_ids",
        ",".join(ctx.datasets),
        "--gpu_log_csv",
        str(module_root / "gpu_util.csv"),
        "--gpu_log_tag",
        f"module04_seed{seed}",
    ]
    if not ctx.resume:
        seed_cmd.append("--no-resume")

    log_path = ctx.audit_dir / "logs" / f"module04_seed_{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        seed_cmd,
        cwd=str(ctx.repo_root),
        stdout=log_path.open("a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )
    return proc


def _stage_module04_manyseed(ctx: RunContext) -> Dict[str, Any]:
    name = "module04_manyseed"
    started = time.time()
    log_path = ctx.audit_dir / "module04_manyseed.log"
    summary_path = ctx.audit_dir / "module04_manyseed_summary.json"
    module_root = ctx.out_root / "module04_manyseed"
    module_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["NUMEXPR_NUM_THREADS"] = "2"

    finished: List[int] = []
    skipped: List[int] = []
    failed: List[int] = []
    budget_skip: List[int] = []
    running: Dict[int, subprocess.Popen] = {}
    seed_queue = list(ctx.seeds)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{_iso_now()}] module04_manyseed start, seeds={ctx.seeds}\n")

    def maybe_mark_completed(seed: int) -> bool:
        return _seed_done(module_root, seed)

    # Seed status pre-pass for resumed runs.
    if ctx.resume:
        for s in list(seed_queue):
            if maybe_mark_completed(s):
                finished.append(s)
                skipped.append(s)
                seed_queue.remove(s)

    grace = max(300.0, min(900.0, 0.04 * ctx.wall_hours * 3600.0))

    while seed_queue or running:
        now = time.time()
        time_left = ctx.deadline - now

        # launch while we have slots and budget to avoid overrun
        while seed_queue and len(running) < ctx.gpu_parallel_procs and time_left > grace:
            seed = seed_queue.pop(0)
            if ctx.resume and maybe_mark_completed(seed):
                skipped.append(seed)
                finished.append(seed)
                continue

            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"[{_iso_now()}] launch seed={seed}\n")

            proc = _run_seed_process(ctx, seed, module_root, env)
            running[seed] = proc
            time.sleep(1)
            time_left = ctx.deadline - time.time()

        if not running:
            # no active jobs
            if seed_queue and time_left <= grace:
                budget_skip.extend(seed_queue)
                seed_queue = []
            elif not running:
                break

        # poll running jobs
        done_seeds = []
        for seed, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            done_seeds.append(seed)
            done_log = ctx.audit_dir / "logs" / f"module04_seed_{seed}.log"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"[{_iso_now()}] seed={seed} rc={rc}\n")

            if rc == 0 and maybe_mark_completed(seed):
                finished.append(seed)
            else:
                failed.append(seed)

        for seed in done_seeds:
            proc = running.pop(seed, None)
            if proc is not None:
                try:
                    proc.stdout.close() if proc.stdout else None
                except Exception:
                    pass

        if running:
            time.sleep(2)

        if seed_queue and time_left <= grace and len(running) >= ctx.gpu_parallel_procs:
            # active set is full and we have no budget to open new ones.
            continue

        if not running and seed_queue and time_left <= grace:
            budget_skip.extend(seed_queue)
            seed_queue = []

    # aggregate only completed seeds
    aggregate_outputs: List[Path] = []
    for s in sorted(finished):
        seed_dir = module_root / f"seed_{s}"
        if seed_dir.exists():
            aggregate_outputs.append(seed_dir)

    aggregate_rc = 0
    agg_path = module_root / "aggregate_results_manyseed.json"
    if finished:
        cmd = [
            sys.executable,
            "aggregate_results.py",
            "--out_root",
            str(module_root),
            "--seeds",
            ",".join(str(s) for s in sorted(set(finished))),
        ]
        agg_log = ctx.audit_dir / "module04_aggregate.log"
        aggregate_rc = _run_cmd(cmd, log_path=agg_log, cwd=ctx.repo_root)
        if aggregate_rc == 0:
            src = module_root / "aggregate_results.json"
            if src.exists():
                shutil.copy2(src, agg_path)
                aggregate_outputs.append(agg_path)
    else:
        aggregate_rc = 0

    # plot stability
    plot_rc = 0
    if finished:
        plot_cmd = [
            sys.executable,
            "scripts/plot_seed_stability.py",
            "--module04_root",
            str(module_root),
        ]
        plot_log = ctx.audit_dir / "module04_plot.log"
        plot_rc = _run_cmd(plot_cmd, log_path=plot_log, cwd=ctx.repo_root)
        if plot_rc == 0:
            fig = module_root / "FIG_normative_seed_stability.png"
            if fig.exists():
                aggregate_outputs.append(fig)

    if failed or aggregate_rc != 0 or plot_rc != 0:
        status = "FAIL"
        error_msg = f"module04 seed failures={failed}, aggregate_rc={aggregate_rc}, plot_rc={plot_rc}"
    elif budget_skip:
        status = "SKIP"
        error_msg = f"budget guard skipped seeds: {budget_skip}"
    else:
        status = "PASS"
        error_msg = ""

    summary = {
        "status": status,
        "seeds_total": len(ctx.seeds),
        "seeds_started": len(set(finished + failed) - set(skipped)),
        "seeds_completed": len(set(finished)),
        "seeds_skipped_existing_or_resumed": sorted(skipped),
        "seeds_budget_skipped": sorted(set(budget_skip)),
        "seeds_failed": sorted(set(failed)),
        "aggregate_results": str(agg_path) if agg_path.exists() else "",
        "plot_rc": plot_rc,
        "aggregate_rc": aggregate_rc,
        "outputs": [str(x) for x in aggregate_outputs],
        "error": error_msg,
    }
    _write_json(summary_path, summary)

    return _record_stage(
        ctx,
        name=name,
        status=status,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        rc=0 if status != "FAIL" else 1,
        command="scripts/run_module.sh (parallel seeds)",
        outputs=aggregate_outputs,
        error=error_msg if status == "FAIL" else None,
    )


def _stage_plot_seed_stability(ctx: RunContext) -> Dict[str, Any]:
    """Compatibility wrapper for environments where many-seed run was not requested."""
    name = "plot_seed_stability"
    started = time.time()
    log_path = ctx.audit_dir / "plot_seed_stability.log"
    summary_path = ctx.audit_dir / "plot_seed_stability_summary.json"
    out_dir = ctx.out_root / "module04_manyseed"

    if not out_dir.exists():
        return _record_stage(
            ctx,
            name=name,
            status="SKIP",
            started=started,
            log_path=log_path,
            summary_path=summary_path,
            rc=0,
            command="none",
            error="module04_manyseed output missing",
        )

    cmd = [
        sys.executable,
        "scripts/plot_seed_stability.py",
        "--module04_root",
        str(out_dir),
    ]
    rc = _run_cmd(cmd, log_path=log_path, cwd=ctx.repo_root)
    status = "PASS" if rc == 0 else "FAIL"
    outputs = [out_dir / "FIG_normative_seed_stability.png", out_dir / "seed_stability_summary.json"] if rc == 0 else []
    _write_json(summary_path, {"status": status, "command": " ".join(cmd), "returncode": rc})

    return _record_stage(
        ctx,
        name=name,
        status=status,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        rc=rc,
        command=" ".join(cmd),
        outputs=outputs,
        error=None if status == "PASS" else _read_file(log_path),
    )


def _stage_zip_bundle(ctx: RunContext) -> Dict[str, Any]:
    name = "zip_bundle"
    started = time.time()
    log_path = ctx.audit_dir / "zip_bundle.log"
    summary_path = ctx.audit_dir / "zip_bundle_summary.json"

    out_zip = ctx.outzip_dir / "overnight_reviewerproof_bundle.zip"
    ctx.outzip_dir.mkdir(parents=True, exist_ok=True)

    include_dirs = [
        ctx.audit_dir,
        ctx.out_root / "lawc_audit_deep",
        ctx.out_root / "lawc_sensitivity",
        ctx.out_root / "rt_linkage_audit",
    ]
    explicit_module04 = [
        ctx.out_root / "module04_manyseed" / "aggregate_results_manyseed.json",
        ctx.out_root / "module04_manyseed" / "FIG_normative_seed_stability.png",
        ctx.out_root / "module04_manyseed" / "seed_stability_metrics.csv",
        ctx.out_root / "module04_manyseed" / "seed_stability_summary.json",
    ]
    added = []
    try:
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(f"[{_iso_now()}] creating zip -> {out_zip}\n")
            with open(out_zip, "wb") as zf_bin:
                pass

        import zipfile

        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root in include_dirs:
                if not root.exists():
                    continue
                for path in sorted(root.rglob("*")):
                    if path.is_dir():
                        continue
                    if path.name.startswith("."):
                        continue
                    rel = path.relative_to(ctx.out_root)
                    zf.write(path, rel.as_posix())
                    added.append(rel)

            # Explicitly include only requested module04_manyseed reviewer artifacts.
            for p in explicit_module04:
                if not p.exists() or p.is_dir():
                    continue
                rel = p.relative_to(ctx.out_root)
                if rel in added:
                    continue
                zf.write(p, rel.as_posix())
                added.append(rel)

        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(f"[{_iso_now()}] zip complete\n")

        status = "PASS"
        rc = 0
        err = ""
    except Exception as exc:
        status = "FAIL"
        rc = 1
        err = str(exc)

    _write_json(
        summary_path,
        {
            "status": status,
            "zip": str(out_zip),
            "n_files": len(added),
            "files": [str(x) for x in added],
            "error": err,
        },
    )
    return _record_stage(
        ctx,
        name=name,
        status=status,
        started=started,
        log_path=log_path,
        summary_path=summary_path,
        rc=rc,
        command="zip bundle",
        outputs=[out_zip] if out_zip.exists() else [],
        error=err or None,
    )


def _write_stop_reason(ctx: RunContext, stage: str, title: str, details: str) -> None:
    lines = [
        "# OVERNIGHT REVIEW STOP REASON",
        f"- Stage: `{stage}`",
        f"- Timestamp: {_iso_now()}",
        f"- Output root: `{ctx.out_root}`",
        "",
        "## Why execution stopped",
        f"{title}",
        "",
        "## What to provide",
        "- Event/load mapping is ambiguous or missing in `configs/lawc_event_map.yaml`.",
        "- Please update dataset-specific `event_filter`, `load_column`, and `load_sign` using inspection output files under AUDIT.",
        "",
        "## Fail-closed context",
    ]
    for line in details.splitlines()[-200:]:
        lines.append(f"- {line}")

    lines.extend(
        [
            "",
            "## Inspection outputs",
            f"- `AUDIT/events_inspection_summary.json`",
            f"- `AUDIT/events_inspection_<dataset>.md`",
        ]
    )
    _write_text(ctx.stop_reason_path, "\n".join(lines) + "\n")


def _write_oversight_report(ctx: RunContext, run_status: str, run_error: str) -> Path:
    path = ctx.audit_dir / "OVERSIGHT_REPORT.md"
    failed = ctx.stage_records[-1] if ctx.stage_records else None
    failed_stage = failed.get("name") if failed else (ctx.failed_stage or "<unknown>")
    failed_log = Path(str(failed.get("log"))) if failed and failed.get("log") else None
    log_tail = _tail_lines(failed_log) if failed_log is not None else ""

    lines = [
        "# OVERSIGHT REPORT",
        "",
        f"- Output root: `{ctx.out_root}`",
        f"- Run status: `{run_status}`",
        f"- Failed stage: `{failed_stage}`",
        f"- Timestamp: `{_iso_now()}`",
        "",
        "## Error",
        "```text",
        run_error or "<no explicit error>",
        "```",
    ]
    if failed is not None:
        lines.extend(
            [
                "",
                "## Failed Stage Context",
                f"- stage: `{failed.get('name')}`",
                f"- returncode: `{failed.get('returncode')}`",
                f"- elapsed_sec: `{failed.get('elapsed_sec')}`",
                f"- log: `{failed.get('log')}`",
                f"- summary: `{failed.get('summary')}`",
            ]
        )
    if log_tail:
        lines.extend(["", "## Log Tail", "```text", log_tail, "```"])

    if ctx.stop_reason_path.exists():
        lines.extend(["", f"- STOP reason file: `{ctx.stop_reason_path}`"])
    _write_text(path, "\n".join(lines) + "\n")
    return path


def _summarize_metric_block(block: Dict[str, Any], label: str) -> str:
    vals = np.asarray(block.get("values", []), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return f"- {label}: n=0 (no finite values)"
    return (
        f"- {label}: n={vals.size}, mean={_fmt_num(block.get('mean'))}, "
        f"ci95={block.get('ci95')}, worst_min={_fmt_num(np.min(vals))}, worst_max={_fmt_num(np.max(vals))}"
    )


def _build_final_report(ctx: RunContext, run_status: str, run_error: Optional[str] = None) -> Path:
    report_path = ctx.audit_dir / "OVERNIGHT_REVIEW_REPORT.md"

    stage_rows = []
    for rec in ctx.stage_records:
        stage_rows.append(
            f"| {rec['name']} | {rec['status']} | {rec['returncode']} | "
            f"{rec['elapsed_sec']:.1f} | {Path(rec['log']).name} | {Path(rec['summary']).name} |"
        )

    lawc_path = ctx.out_root / "lawc_audit_deep" / "locked_test_results.json"
    lawc_rows = []
    lawc_fallback_rows = []
    if lawc_path.exists():
        try:
            payload = json.loads(lawc_path.read_text(encoding="utf-8"))
            for row in payload.get("datasets", []):
                lawc_rows.append(
                    f"| {row.get('dataset_id')} | {row.get('median_rho', np.nan):.6g} | "
                    f"{row.get('p_value', np.nan):.6g} | {row.get('q_value', np.nan):.6g} | "
                    f"{row.get('pass_all', False)} |"
                )
                rt_sources = json.dumps(row.get("rt_source_counts", {}), sort_keys=True)
                channels = json.dumps(row.get("channel_counts", {}), sort_keys=True)
                lawc_fallback_rows.append(
                    f"| {row.get('dataset_id')} | {row.get('fallback_non_pz_trials', 0)} | "
                    f"{_fmt_num(row.get('rt_nonmissing_rate', np.nan), nd=4)} | `{rt_sources}` | `{channels}` |"
                )
        except Exception:
            lawc_rows.append("| parse_error | parse_error | parse_error | parse_error | parse_error |")
            lawc_fallback_rows.append("| parse_error | parse_error | parse_error | parse_error | parse_error |")
    else:
        lawc_rows.append("| SKIP | SKIP | SKIP | SKIP | SKIP |")
        lawc_fallback_rows.append("| SKIP | SKIP | SKIP | SKIP | SKIP |")

    sensitivity_rows = []
    sens_csv = ctx.out_root / "lawc_sensitivity" / "sensitivity_table.csv"
    if sens_csv.exists():
        import csv

        with sens_csv.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                sensitivity_rows.append(
                    f"| {row.get('dataset_id')} | {row.get('transform')} | {row.get('status')} | "
                    f"{row.get('median_rho')} | {row.get('sign_consistency_posfrac')} |"
                )
    else:
        sensitivity_rows.append("| SKIP | SKIP | SKIP | SKIP | SKIP |")

    rt_rows = []
    rt_csv = ctx.out_root / "rt_linkage_audit" / "rt_linkage_results.csv"
    if rt_csv.exists():
        import csv

        with rt_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row.get("dataset_id") == "pooled":
                continue
            rt_rows.append(
                f"| {row.get('dataset_id')} | {row.get('observed_median_effect')} | "
                f"{row.get('p_value')} | {row.get('q_value')} | {row.get('control_weaker_than_primary')} |"
            )
    else:
        rt_rows.append("| SKIP | SKIP | SKIP | SKIP | SKIP |")

    many_seed = ""
    agg = ctx.out_root / "module04_manyseed" / "aggregate_results_manyseed.json"
    if agg.exists():
        try:
            data = json.loads(agg.read_text(encoding="utf-8"))
            norm = data.get("normative_metrics", {})
            rt = data.get("rt_linkage_effect", {})
            many_seed = "\n".join(
                [
                    _summarize_metric_block(norm.get("healthy_nll", {}), "Healthy NLL"),
                    _summarize_metric_block(norm.get("healthy_calibration_z_std", {}), "Healthy z_std"),
                    _summarize_metric_block(
                        norm.get("healthy_z_stability_subject_mean_std", {}),
                        "Healthy z-stability subject_mean_std",
                    ),
                    _summarize_metric_block(rt.get("healthy_mean_beta_margin", {}), "Healthy RT beta margin"),
                ]
            )
        except Exception:
            many_seed = "\n- parse_error"
    else:
        many_seed = "\n- SKIP (aggregate_results_manyseed.json missing)"

    lines = [
        "# OVERNIGHT REVIEW REPORT",
        "",
        f"- Output root: `{ctx.out_root}`",
        f"- Run status: `{run_status}`",
        f"- Resume: `{ctx.resume}`",
        f"- Start: `{datetime.fromtimestamp(ctx.start_ts, tz=timezone.utc).isoformat()}`",
        f"- End: `{datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat()}`",
    ]

    if run_error:
        lines.extend(["", f"## Run error", f"```", run_error, "```"])

    lines.extend(
        [
            "",
            "## Stage status table",
            "| Stage | Status | Return code | Runtime (s) | Log | Summary |",
            "|---|---|---:|---:|---|---|",
            *stage_rows,
        ]
    )

    lines.extend(
        [
            "",
            "## Law-C Deep Audit",
            "| Dataset | Median rho | p | q | PASS |",
            "|---|---:|---:|---:|---|",
            *lawc_rows,
            "",
            "### Law-C Fallback Counters",
            "| Dataset | Non-Pz fallback trials | RT non-missing | RT sources | P3 channel counts |",
            "|---|---:|---:|---|---|",
            *lawc_fallback_rows,
            "",
            "## Law-C Sensitivity",
            "| Dataset | Transform | Status | Median rho | Sign(consistency >0) |",
            "|---|---|---|---:|---:|",
            *sensitivity_rows,
            "",
            "## RT Linkage Audit",
            "| Dataset | Observed median beta | p | q | Control weaker than primary |",
            "|---|---:|---:|---:|---|",
            *rt_rows,
            "",
            "## Module04 Many-Seed",
            many_seed,
            "",
            "## Artefacts",
            f"- PRE: `{ctx.audit_dir}`",
            f"- Law-C deep: `{ctx.out_root / 'lawc_audit_deep'}`",
            f"- Law-C sensitivity: `{ctx.out_root / 'lawc_sensitivity'}`",
            f"- RT linkage: `{ctx.out_root / 'rt_linkage_audit'}`",
            f"- Module04 many-seed: `{ctx.out_root / 'module04_manyseed'}`",
            f"- Zip: `{ctx.outzip_dir / 'overnight_reviewerproof_bundle.zip'}`",
        ]
    )

    if ctx.stop_reason_path.exists():
        lines.extend(["", f"- STOP reason: `{ctx.stop_reason_path}`"])
    oversight_path = ctx.audit_dir / "OVERSIGHT_REPORT.md"
    if oversight_path.exists():
        lines.extend(["", f"- OVERSIGHT: `{oversight_path}`"])

    _write_text(report_path, "\n".join(lines) + "\n")

    return report_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", type=Path, default=Path("/filesystemHcog/features_cache_FIX2_20260222_061927"))
    ap.add_argument("--data_root", type=Path, default=Path("/filesystemHcog/openneuro"))
    ap.add_argument("--datasets", type=str, default="ds005095,ds003655,ds004117")
    ap.add_argument("--event_map", type=Path, default=Path("configs/lawc_event_map.yaml"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--out_root", type=Path, default=None)
    ap.add_argument("--wall_hours", type=float, default=8.0)
    ap.add_argument("--lawc_n_perm", type=int, default=20000)
    ap.add_argument("--rtlink_n_perm", type=int, default=20000)
    ap.add_argument("--seeds", type=str, default="0-99")
    ap.add_argument("--gpu_parallel_procs", type=int, default=6)
    ap.add_argument("--cpu_workers", type=int, default=32)
    ap.add_argument("--resume", action="store_true", default=False)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_root = args.out_root
    if out_root is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_root = Path("/filesystemHcog/runs") / f"{ts}_OVERNIGHT_REVIEW_FIX2"

    ctx = RunContext(
        repo_root=REPO_ROOT,
        out_root=out_root,
        features_root=args.features_root,
        data_root=args.data_root,
        datasets=[x.strip() for x in str(args.datasets).split(",") if x.strip()],
        event_map=args.event_map,
        config=args.config,
        wall_hours=args.wall_hours,
        lawc_n_perm=args.lawc_n_perm,
        rtlink_n_perm=args.rtlink_n_perm,
        seeds=_parse_seeds(args.seeds),
        gpu_parallel_procs=args.gpu_parallel_procs,
        cpu_workers=args.cpu_workers,
        resume=args.resume,
    )

    if ctx.out_root.exists() and not ctx.resume:
        print(f"ERROR: out_root exists and --resume not set: {ctx.out_root}", file=sys.stderr, flush=True)
        return 1

    ctx.out_root.mkdir(parents=True, exist_ok=True)
    ctx.audit_dir.mkdir(parents=True, exist_ok=True)
    ctx.log_dir.mkdir(parents=True, exist_ok=True)
    ctx.outzip_dir.mkdir(parents=True, exist_ok=True)

    run_status = "PASS"
    run_error = ""

    stage_sequence = [
        lambda: _stage_preflight(ctx),
        lambda: _stage_compile_gate(ctx),
        lambda: _stage_inspect_events(ctx),
        lambda: _stage_lawc_deep(ctx),
        lambda: _stage_lawc_sensitivity(ctx),
        lambda: _stage_rt_linkage(ctx),
        lambda: _stage_module04_manyseed(ctx),
        lambda: _stage_plot_seed_stability(ctx),
        lambda: _stage_zip_bundle(ctx),
    ]

    try:
        for fn in stage_sequence:
            rec = fn()
            if rec["status"] == "FAIL":
                run_status = "FAIL"
                run_error = rec.get("error") or "stage failed"
                ctx.failed_stage = rec["name"]
                break

            if rec["status"] == "SKIP" and rec["name"] == "inspect_events":
                # SKIP for event inspection is failure-by-design for this pipeline.
                run_status = "FAIL"
                run_error = "inspect_events skipped/failed"
                ctx.failed_stage = rec["name"]
                break

            if rec["name"] == "module04_manyseed" and rec["status"] == "SKIP":
                # Budget-limited continuation is not a hard fail, continue.
                continue

            if ctx.failed_stage:
                break
    except Exception as exc:
        run_status = "FAIL"
        run_error = f"{type(exc).__name__}: {exc}"
        if hasattr(exc, "__traceback__"):
            run_error += "\n" + "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)  # type: ignore[arg-type]
            )
    finally:
        # If stage 1 passed, keep monitoring until exit.
        if ctx.monitor_proc is not None:
            _stop_monitor(ctx)

    report_path = _build_final_report(ctx, run_status, run_error)
    oversight_path: Optional[Path] = None
    if run_status != "PASS":
        oversight_path = _write_oversight_report(ctx, run_status, run_error)

    # persist run manifest
    _write_json(
        ctx.audit_dir / "run_status.json",
        {
            "status": run_status,
            "error": run_error,
            "failed_stage": ctx.failed_stage,
            "report": str(report_path),
            "out_root": str(ctx.out_root),
            "stage_records": ctx.stage_records,
        },
    )

    print(f"FINAL_REPORT={report_path}", flush=True)
    print(f"OUT_ROOT={ctx.out_root}", flush=True)
    print(f"BUNDLE_ZIP={ctx.outzip_dir / 'overnight_reviewerproof_bundle.zip'}", flush=True)
    if oversight_path is not None:
        print(f"OVERSIGHT={oversight_path}", flush=True)

    if run_status != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
