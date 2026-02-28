#!/usr/bin/env python3
"""NN_SOLID_1_2 runner: mechanism + clinical translation reviewer pack."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


STAGE_ORDER = [
    "preflight",
    "compile_gate",
    "data_staging",
    "mechanism_deep",
    "lawc_ultradeep",
    "effect_size_pack",
    "clinical_translation",
    "zip_bundle",
]


@dataclass
class RunContext:
    out_root: Path
    audit_dir: Path
    outzip_dir: Path
    data_root: Path
    features_root_healthy: Path
    features_root_mechanism: Path
    mechanism_dataset: str
    sternberg_datasets: List[str]
    event_map: Path
    mechanism_event_map: Path
    config: Path
    datasets_stage_config: Path
    clinical_bids_root: Path
    clinical_severity_csv: Path
    wall_hours: float
    lawc_n_perm: int
    mechanism_n_perm: int
    mechanism_seeds: str
    gpu_parallel_procs: int
    cpu_workers: int
    resume: bool

    start_ts: float
    deadline_ts: float
    stage_records: List[Dict[str, Any]]
    monitor_proc: Optional[subprocess.Popen]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _tail(path: Path, n: int = 120) -> str:
    txt = _read_text(path)
    if not txt:
        return ""
    lines = txt.splitlines()
    return "\n".join(lines[-max(1, int(n)) :])


def _run_cmd(cmd: List[str], *, cwd: Path, log_path: Path, env: Optional[Dict[str, str]] = None) -> int:
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
        return int(proc.returncode)


def _record_stage(
    ctx: RunContext,
    *,
    stage: str,
    status: str,
    started: float,
    rc: int,
    log_path: Path,
    summary_path: Path,
    command: str,
    outputs: Optional[Sequence[Path]] = None,
    error: str = "",
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
        "outputs": [str(x) for x in (outputs or [])],
        "error": error,
    }
    _write_json(summary_path, rec)
    _write_text(ctx.audit_dir / f"{stage}.status", status + "\n")
    ctx.stage_records.append(rec)
    return rec


def _start_gpu_monitor(ctx: RunContext) -> None:
    out_csv = ctx.audit_dir / "nvidia_smi_1hz.csv"
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
        "-l",
        "1",
    ]
    handle = out_csv.open("w", encoding="utf-8")
    ctx.monitor_proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=handle, stderr=handle, text=True)


def _stop_gpu_monitor(ctx: RunContext) -> None:
    p = ctx.monitor_proc
    if p is None:
        return
    try:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)
    finally:
        ctx.monitor_proc = None


def _budget_exhausted(ctx: RunContext) -> bool:
    return time.time() > ctx.deadline_ts


def _stage_preflight(ctx: RunContext) -> Dict[str, Any]:
    stage = "preflight"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"
    outputs: List[Path] = []

    try:
        ctx.out_root.mkdir(parents=True, exist_ok=True)
        ctx.audit_dir.mkdir(parents=True, exist_ok=True)
        ctx.outzip_dir.mkdir(parents=True, exist_ok=True)

        if not ctx.data_root.exists():
            raise RuntimeError(f"DATA_ROOT missing: {ctx.data_root}")
        if not ctx.features_root_healthy.exists():
            raise RuntimeError(f"FEATURES_ROOT_HEALTHY missing: {ctx.features_root_healthy}")

        py_info = ctx.audit_dir / "python_version.txt"
        _write_text(py_info, sys.version + "\n")
        outputs.append(py_info)

        pip_freeze = ctx.audit_dir / "pip_freeze.txt"
        r = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
        _write_text(pip_freeze, (r.stdout if r.returncode == 0 else r.stdout + "\n" + r.stderr))
        outputs.append(pip_freeze)

        nvl = ctx.audit_dir / "nvidia_smi_L.txt"
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
        _write_text(nvl, r.stdout if r.returncode == 0 else r.stdout + "\n" + r.stderr)
        outputs.append(nvl)

        nvs = ctx.audit_dir / "nvidia_smi_snapshot.csv"
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        _write_text(nvs, r.stdout if r.returncode == 0 else r.stdout + "\n" + r.stderr)
        outputs.append(nvs)

        torch_info = {
            "torch_version": "<unavailable>",
            "cuda_available": False,
            "cuda_version": "<unavailable>",
            "device_name": "<unavailable>",
        }
        try:
            import torch

            torch_info["torch_version"] = str(torch.__version__)
            torch_info["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                torch_info["cuda_version"] = str(getattr(torch.version, "cuda", "<unavailable>"))
                torch_info["device_name"] = str(torch.cuda.get_device_name(0))
        except Exception as exc:
            torch_info["error"] = str(exc)

        torch_json = ctx.audit_dir / "torch_cuda_info.json"
        _write_json(torch_json, torch_info)
        outputs.append(torch_json)

        repo_fp = (
            _compute_repo_fingerprint(REPO_ROOT)
            if _compute_repo_fingerprint is not None
            else {
                "repo_root": str(REPO_ROOT),
                "git_head": None,
                "repo_fingerprint_sha256": None,
                "file_count_hashed": 0,
                "note": "_compute_repo_fingerprint unavailable",
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
            started=started,
            rc=0,
            log_path=log_path,
            summary_path=summary_path,
            command="preflight",
            outputs=outputs,
        )
    except Exception as exc:
        _stop_gpu_monitor(ctx)
        _write_text(log_path, traceback.format_exc())
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            started=started,
            rc=1,
            log_path=log_path,
            summary_path=summary_path,
            command="preflight",
            error=str(exc),
            outputs=outputs,
        )


def _stage_compile_gate(ctx: RunContext) -> Dict[str, Any]:
    stage = "compile_gate"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"
    cmd = ["bash", "-lc", "find . -name '*.py' -print0 | xargs -0 python -m py_compile"]
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path)
    return _record_stage(
        ctx,
        stage=stage,
        status="PASS" if rc == 0 else "FAIL",
        started=started,
        rc=rc,
        log_path=log_path,
        summary_path=summary_path,
        command="find . -name '*.py' -print0 | xargs -0 python -m py_compile",
    )


def _stage_data_staging(ctx: RunContext) -> Dict[str, Any]:
    stage = "data_staging"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    cmd = [
        sys.executable,
        "00_stage_data.py",
        "--config",
        str(ctx.datasets_stage_config),
        "--openneuro_root",
        str(ctx.data_root),
        "--manifest_out",
        str(ctx.audit_dir / "stage_data_manifest.json"),
    ]
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path)

    ds_root = ctx.data_root / ctx.mechanism_dataset
    eeg_count = len(list(ds_root.rglob("*_eeg.*"))) if ds_root.exists() else 0
    pupil_count = len(list(ds_root.rglob("*_pupil.tsv"))) + len(list(ds_root.rglob("*_pupil.tsv.gz")))

    mech_root = ctx.features_root_mechanism / ctx.mechanism_dataset
    mech_h5 = len(list(mech_root.rglob("*.h5"))) if mech_root.exists() else 0

    status = "PASS"
    error = ""
    if rc != 0:
        status = "FAIL"
        error = "00_stage_data.py failed"
    elif not ds_root.exists():
        status = "FAIL"
        error = f"mechanism dataset missing: {ds_root}"
    elif eeg_count == 0:
        status = "FAIL"
        error = f"no EEG files found in {ds_root}"
    elif pupil_count == 0:
        status = "FAIL"
        error = f"no pupil TSV files found in {ds_root}"
    elif mech_h5 == 0:
        status = "FAIL"
        error = (
            f"mechanism features missing at {mech_root}. "
            "Provide ds003838 feature cache (expected *.h5) or extend runner to run feature extraction first."
        )

    payload = {
        "status": status,
        "command": " ".join(cmd),
        "returncode": rc,
        "dataset_root": str(ds_root),
        "eeg_file_count": int(eeg_count),
        "pupil_file_count": int(pupil_count),
        "mechanism_features_root": str(mech_root),
        "mechanism_feature_files": int(mech_h5),
        "error": error,
    }
    _write_json(summary_path, payload)

    if status == "FAIL":
        _write_text(ctx.audit_dir / "STOP_REASON.md", "\n".join([
            "# STOP_REASON",
            f"- Stage: `{stage}`",
            f"- Reason: {error}",
            "- Required: ds003838 EEG + pupil files under DATA_ROOT and ds003838 feature H5 cache.",
            f"- Data root checked: `{ds_root}`",
            f"- Feature root checked: `{mech_root}`",
            "",
            "## Log tail",
            "```text",
            _tail(log_path, 120),
            "```",
        ]) + "\n")

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        started=started,
        rc=rc if status == "PASS" else 1,
        log_path=log_path,
        summary_path=summary_path,
        command=" ".join(cmd),
        error=error,
        outputs=[ctx.audit_dir / "stage_data_manifest.json"],
    )


def _stage_mechanism_deep(ctx: RunContext) -> Dict[str, Any]:
    stage = "mechanism_deep"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"
    out_dir = ctx.out_root / "MechanismPack"

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["NUMEXPR_NUM_THREADS"] = "2"

    cmd = [
        sys.executable,
        "scripts/mechanism_deep.py",
        "--features_root",
        str(ctx.features_root_mechanism),
        "--data_root",
        str(ctx.data_root),
        "--dataset_id",
        str(ctx.mechanism_dataset),
        "--config",
        str(ctx.config),
        "--event_map",
        str(ctx.mechanism_event_map),
        "--out_dir",
        str(out_dir),
        "--seeds",
        str(ctx.mechanism_seeds),
        "--parallel_procs",
        str(ctx.gpu_parallel_procs),
        "--n_perm",
        str(ctx.mechanism_n_perm),
        "--min_trials",
        "20",
    ]
    if ctx.resume:
        cmd.append("--resume")

    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, env=env)

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
    error = "" if status == "PASS" else f"mechanism stage failed: rc={rc}, missing={missing}"

    payload = {
        "status": status,
        "command": " ".join(cmd),
        "returncode": rc,
        "out_dir": str(out_dir),
        "missing_required": missing,
        "error": error,
    }
    _write_json(summary_path, payload)

    if status == "FAIL":
        _write_text(
            ctx.audit_dir / "STOP_REASON.md",
            "\n".join(
                [
                    "# STOP_REASON",
                    f"- Stage: `{stage}`",
                    f"- Reason: {error}",
                    "",
                    "## Log tail",
                    "```text",
                    _tail(log_path, 160),
                    "```",
                ]
            )
            + "\n",
        )

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        started=started,
        rc=rc if status == "PASS" else 1,
        log_path=log_path,
        summary_path=summary_path,
        command=" ".join(cmd),
        outputs=[p for p in required if p.exists()],
        error=error,
    )


def _stage_lawc_ultradeep(ctx: RunContext) -> Dict[str, Any]:
    stage = "lawc_ultradeep"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    out_dir = ctx.out_root / "lawc_audit_ultradeep"
    cmd = [
        sys.executable,
        "05_audit_lawc.py",
        "--features_root",
        str(ctx.features_root_healthy),
        "--out_root",
        str(out_dir),
        "--event_map",
        str(ctx.event_map),
        "--datasets",
        ",".join(ctx.sternberg_datasets),
        "--n_perm",
        str(ctx.lawc_n_perm),
        "--workers",
        str(ctx.cpu_workers),
    ]
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path)

    copied: List[Path] = []
    src_dir = out_dir / "lawc_audit"
    if src_dir.exists():
        for name in ["locked_test_results.json", "locked_test_results.csv", "negative_controls.csv"]:
            src = src_dir / name
            if src.exists():
                dst = out_dir / name
                shutil.copy2(src, dst)
                copied.append(dst)

    payload_path = out_dir / "locked_test_results.json"
    payload: Dict[str, Any] = {}
    if payload_path.exists():
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    lawc_pass = bool(payload.get("pass", False)) if payload else False
    status = "PASS" if rc == 0 and lawc_pass else "FAIL"
    error = "" if status == "PASS" else f"law-c ultradeep failed: rc={rc}, pass={lawc_pass}"

    out_payload = {
        "status": status,
        "command": " ".join(cmd),
        "returncode": rc,
        "pass": lawc_pass,
        "out_dir": str(out_dir),
        "outputs": [str(x) for x in copied],
        "error": error,
    }
    _write_json(summary_path, out_payload)

    if status == "FAIL":
        _write_text(
            ctx.audit_dir / "STOP_REASON.md",
            "\n".join(
                [
                    "# STOP_REASON",
                    f"- Stage: `{stage}`",
                    f"- Reason: {error}",
                    "",
                    "## Log tail",
                    "```text",
                    _tail(log_path, 160),
                    "```",
                ]
            )
            + "\n",
        )

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        started=started,
        rc=rc if status == "PASS" else 1,
        log_path=log_path,
        summary_path=summary_path,
        command=" ".join(cmd),
        outputs=copied,
        error=error,
    )


def _stage_effect_size_pack(ctx: RunContext) -> Dict[str, Any]:
    stage = "effect_size_pack"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"
    out_dir = ctx.out_root / "EffectSizePack"

    cmd = [
        sys.executable,
        "scripts/effect_size_pack.py",
        "--features_root",
        str(ctx.features_root_healthy),
        "--datasets",
        ",".join(ctx.sternberg_datasets),
        "--out_dir",
        str(out_dir),
        "--n_boot",
        "5000",
        "--seed",
        "123",
    ]
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path)

    required = [
        out_dir / "effect_size_summary.csv",
        out_dir / "per_subject_effect_sizes.csv",
        out_dir / "grand_average_by_load.csv",
        out_dir / "FIG_slopes_uv_per_load.png",
        out_dir / "FIG_delta_uv_high_vs_low.png",
        out_dir / "FIG_waveforms_by_load.png",
    ]
    missing = [str(p) for p in required if not p.exists()]
    status = "PASS" if rc == 0 and not missing else "FAIL"
    error = "" if status == "PASS" else f"effect_size_pack failed: rc={rc}, missing={missing}"

    payload = {
        "status": status,
        "command": " ".join(cmd),
        "returncode": rc,
        "out_dir": str(out_dir),
        "missing_required": missing,
        "error": error,
    }
    _write_json(summary_path, payload)

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        started=started,
        rc=rc if status == "PASS" else 1,
        log_path=log_path,
        summary_path=summary_path,
        command=" ".join(cmd),
        outputs=[p for p in required if p.exists()],
        error=error,
    )


def _clinical_stop_reason(ctx: RunContext) -> Path:
    out_dir = ctx.out_root / "ClinicalPack"
    out_dir.mkdir(parents=True, exist_ok=True)
    stop = out_dir / "STOP_REASON.md"
    _write_text(
        stop,
        "\n".join(
            [
                "# ClinicalPack STOP_REASON",
                "",
                "Clinical stage skipped because required clinical inputs are missing.",
                "",
                "## Missing inputs",
                f"- CLINICAL_BIDS_ROOT: `{ctx.clinical_bids_root}`",
                f"- CLINICAL_SEVERITY_CSV: `{ctx.clinical_severity_csv}`",
                "",
                "## Expected clinical folder structure",
                "- `<CLINICAL_BIDS_ROOT>/<dataset_or_site>/sub-*/[ses-*/]eeg/*_eeg.*`",
                "- `<CLINICAL_BIDS_ROOT>/<dataset_or_site>/sub-*/[ses-*/]eeg/*_events.tsv[.gz]`",
                "",
                "## Required severity CSV columns",
                "- `subject_id`",
                "- one or more severity outcomes (e.g., PANSS/MMSE/HAMD)",
                "- recommended: `age`, `sex`",
                "",
                "## Expected event map format",
                "- Update `configs/clinical_event_map.yaml` with explicit dataset mappings:",
                "```yaml",
                "datasets:",
                "  <dataset_id>:",
                "    event_filter: \"<pandas query>\"",
                "    load_column: \"<column>\"",
                "    load_regex: \"<optional regex>\"",
                "```",
            ]
        )
        + "\n",
    )
    return stop


def _stage_clinical_translation(ctx: RunContext) -> Dict[str, Any]:
    stage = "clinical_translation"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"
    out_dir = ctx.out_root / "ClinicalPack"

    if not ctx.clinical_bids_root.exists() or not ctx.clinical_severity_csv.exists():
        stop = _clinical_stop_reason(ctx)
        payload = {
            "status": "SKIP",
            "reason": "missing clinical root and/or severity csv",
            "clinical_bids_root_exists": bool(ctx.clinical_bids_root.exists()),
            "clinical_severity_csv_exists": bool(ctx.clinical_severity_csv.exists()),
            "stop_reason": str(stop),
        }
        _write_json(summary_path, payload)
        _write_text(log_path, json.dumps(payload, indent=2) + "\n")
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            started=started,
            rc=0,
            log_path=log_path,
            summary_path=summary_path,
            command="clinical skip (inputs missing)",
            outputs=[stop],
        )

    cmd = [
        sys.executable,
        "scripts/clinical_apply.py",
        "--clinical_bids_root",
        str(ctx.clinical_bids_root),
        "--clinical_severity_csv",
        str(ctx.clinical_severity_csv),
        "--healthy_features_root",
        str(ctx.features_root_healthy),
        "--event_map",
        str(REPO_ROOT / "configs" / "clinical_event_map.yaml"),
        "--out_dir",
        str(out_dir),
        "--n_perm",
        "20000",
        "--gpu_parallel_procs",
        str(ctx.gpu_parallel_procs),
        "--cpu_workers",
        str(ctx.cpu_workers),
    ]
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path)

    status = "PASS" if rc == 0 else "FAIL"
    err = "" if status == "PASS" else "clinical_apply.py failed"

    clinical_summary = out_dir / "clinical_apply_summary.json"
    if clinical_summary.exists():
        try:
            payload = json.loads(clinical_summary.read_text(encoding="utf-8"))
            status_from_payload = str(payload.get("status", "")).upper()
            if status_from_payload in {"SKIP", "PASS", "FAIL"}:
                status = status_from_payload
                if status == "FAIL" and not err:
                    err = str(payload.get("reason", "clinical stage failed"))
        except Exception:
            pass

    _write_json(
        summary_path,
        {
            "status": status,
            "command": " ".join(cmd),
            "returncode": rc,
            "out_dir": str(out_dir),
            "error": err,
        },
    )

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        started=started,
        rc=0 if status != "FAIL" else 1,
        log_path=log_path,
        summary_path=summary_path,
        command=" ".join(cmd),
        outputs=list(out_dir.glob("*")),
        error=err,
    )


def _stage_zip_bundle(ctx: RunContext) -> Dict[str, Any]:
    stage = "zip_bundle"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    out_zip = ctx.outzip_dir / "nn_solid_1_2_bundle.zip"
    ctx.outzip_dir.mkdir(parents=True, exist_ok=True)

    include_roots = [
        ctx.audit_dir,
        ctx.out_root / "MechanismPack",
        ctx.out_root / "ClinicalPack",
        ctx.out_root / "EffectSizePack",
        ctx.out_root / "lawc_audit_ultradeep",
    ]

    added: List[str] = []
    error = ""
    try:
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root in include_roots:
                if not root.exists():
                    continue
                for path in sorted(root.rglob("*")):
                    if path.is_dir():
                        continue
                    rel = path.relative_to(ctx.out_root)
                    zf.write(path, rel.as_posix())
                    added.append(rel.as_posix())
        status = "PASS"
        rc = 0
    except Exception as exc:
        status = "FAIL"
        rc = 1
        error = str(exc)

    _write_json(
        summary_path,
        {
            "status": status,
            "zip": str(out_zip),
            "n_files": len(added),
            "files": added,
            "error": error,
        },
    )

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        started=started,
        rc=rc,
        log_path=log_path,
        summary_path=summary_path,
        command="zip bundle",
        outputs=[out_zip] if out_zip.exists() else [],
        error=error,
    )


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_report(ctx: RunContext, run_status: str, run_error: str) -> Path:
    report = ctx.audit_dir / "NN_SOLID_1_2_REPORT.md"

    stage_rows = [
        f"| {r['stage']} | {r['status']} | {r['returncode']} | {r['elapsed_sec']:.1f} | {Path(r['log']).name} | {Path(r['summary']).name} |"
        for r in ctx.stage_records
    ]

    lawc_rows = ["| SKIP | SKIP | SKIP | SKIP | SKIP |"]
    lawc_json = ctx.out_root / "lawc_audit_ultradeep" / "locked_test_results.json"
    if lawc_json.exists():
        try:
            payload = json.loads(lawc_json.read_text(encoding="utf-8"))
            lawc_rows = []
            for d in payload.get("datasets", []):
                lawc_rows.append(
                    f"| {d.get('dataset_id')} | {d.get('median_rho', float('nan')):.6g} | {d.get('p_value', float('nan')):.6g} | {d.get('q_value', float('nan')):.6g} | {d.get('pass_all', False)} |"
                )
        except Exception:
            lawc_rows = ["| parse_error | parse_error | parse_error | parse_error | parse_error |"]

    eff_rows = ["| SKIP | SKIP | SKIP | SKIP | SKIP |"]
    eff_csv = ctx.out_root / "EffectSizePack" / "effect_size_summary.csv"
    if eff_csv.exists():
        eff = _read_csv_rows(eff_csv)
        if eff:
            eff_rows = []
            for r in eff:
                eff_rows.append(
                    f"| {r.get('dataset_id')} | {r.get('n_subjects')} | {r.get('slope_median_uv_per_load')} [{r.get('slope_ci95_lo')}, {r.get('slope_ci95_hi')}] | {r.get('delta_median_uv')} [{r.get('delta_ci95_lo')}, {r.get('delta_ci95_hi')}] | reviewer_pack |"
                )

    mech_rows = ["| SKIP | SKIP | SKIP | SKIP | SKIP | SKIP |"]
    mech_csv = ctx.out_root / "MechanismPack" / "Table_mechanism_effects.csv"
    if mech_csv.exists():
        rows = _read_csv_rows(mech_csv)
        if rows:
            mech_rows = []
            for r in rows:
                mech_rows.append(
                    f"| {r.get('metric')} | {r.get('observed_median')} | [{r.get('ci95_low')}, {r.get('ci95_high')}] | {r.get('p_value')} | {r.get('q_value')} | {r.get('control_pupil_degrade')}/{r.get('control_load_degrade')} |"
                )

    clinical_lines: List[str] = []
    stop_reason = ctx.out_root / "ClinicalPack" / "STOP_REASON.md"
    if stop_reason.exists():
        clinical_lines.append(f"- Clinical stage SKIP: `{stop_reason}`")
    else:
        clinical_summary = ctx.out_root / "ClinicalPack" / "clinical_apply_summary.json"
        if clinical_summary.exists():
            clinical_lines.append(f"- Clinical summary: `{clinical_summary}`")
        else:
            clinical_lines.append("- Clinical outputs not found")

    mech_agg = ctx.out_root / "MechanismPack" / "aggregate_mechanism.json"
    mech_seed_summary = ""
    if mech_agg.exists():
        try:
            ag = json.loads(mech_agg.read_text(encoding="utf-8"))
            ss = (ag.get("seed_stability") or {}).get("aggregate_metrics", {})
            lines = []
            for key in ["a", "b", "c_prime", "ab", "interaction"]:
                block = ss.get(key, {})
                lines.append(
                    f"- {key}: n={block.get('n_seeds', 0)}, mean={block.get('mean')}, ci95={block.get('ci95')}, worst=[{block.get('worst_min')}, {block.get('worst_max')}]"
                )
            mech_seed_summary = "\n".join(lines)
        except Exception:
            mech_seed_summary = "- parse_error"
    else:
        mech_seed_summary = "- aggregate_mechanism.json missing"

    repo_fp_path = ctx.audit_dir / "repo_fingerprint.json"

    run_cmd = (
        f"python scripts/nn_solid_1_2_runner.py --wall_hours {ctx.wall_hours} --out_root {ctx.out_root} "
        f"--data_root {ctx.data_root} --features_root_healthy {ctx.features_root_healthy} "
        f"--features_root_mechanism {ctx.features_root_mechanism} --mechanism_dataset {ctx.mechanism_dataset} "
        f"--sternberg_datasets {','.join(ctx.sternberg_datasets)} --event_map {ctx.event_map} "
        f"--config {ctx.config} --clinical_bids_root {ctx.clinical_bids_root} "
        f"--clinical_severity_csv {ctx.clinical_severity_csv} --lawc_n_perm {ctx.lawc_n_perm} "
        f"--mechanism_n_perm {ctx.mechanism_n_perm} --mechanism_seeds {ctx.mechanism_seeds} "
        f"--gpu_parallel_procs {ctx.gpu_parallel_procs} --cpu_workers {ctx.cpu_workers}"
    )

    lines = [
        "# NN_SOLID_1_2 REPORT",
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
            "## Stage status table",
            "| Stage | Status | Return code | Runtime (s) | Log | Summary |",
            "|---|---|---:|---:|---|---|",
            *stage_rows,
            "",
            "## Law-C ultradeep",
            "| Dataset | Median rho | p | q | PASS |",
            "|---|---:|---:|---:|---|",
            *lawc_rows,
            "",
            "## Effect sizes (uV)",
            "| Dataset | N subjects | Slope median [CI95] | Delta median [CI95] | Notes |",
            "|---|---:|---|---|---|",
            *eff_rows,
            "",
            "## Mechanism results",
            "| Metric | Observed median | CI95 | p | q | Controls degrade (pupil/load) |",
            "|---|---:|---|---:|---:|---|",
            *mech_rows,
            "",
            "### Mechanism seed stability",
            mech_seed_summary,
            "",
            "## Clinical translation",
            *clinical_lines,
            "",
            "## Commands and provenance",
            f"- Runner command: `{run_cmd}`",
            f"- Stage data config: `{ctx.datasets_stage_config}`",
            f"- Mechanism event map: `{ctx.mechanism_event_map}`",
            f"- Law-C event map: `{ctx.event_map}`",
            f"- Repo fingerprint: `{repo_fp_path}`",
            f"- GPU monitor CSV: `{ctx.audit_dir / 'nvidia_smi_1hz.csv'}`",
            "",
            "## Figure paths",
            f"- `{ctx.out_root / 'MechanismPack' / 'FIG_load_vs_pupil.png'}`",
            f"- `{ctx.out_root / 'MechanismPack' / 'FIG_pupil_vs_p3_partial.png'}`",
            f"- `{ctx.out_root / 'MechanismPack' / 'FIG_mediation_ab.png'}`",
            f"- `{ctx.out_root / 'MechanismPack' / 'FIG_mechanism_summary.png'}`",
            f"- `{ctx.out_root / 'EffectSizePack' / 'FIG_slopes_uv_per_load.png'}`",
            f"- `{ctx.out_root / 'EffectSizePack' / 'FIG_delta_uv_high_vs_low.png'}`",
            f"- `{ctx.out_root / 'EffectSizePack' / 'FIG_waveforms_by_load.png'}`",
            "",
            "## Bundled artifact",
            f"- `{ctx.out_root / 'OUTZIP' / 'nn_solid_1_2_bundle.zip'}`",
        ]
    )

    _write_text(report, "\n".join(lines) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, default=Path("/filesystemHcog/openneuro"))
    ap.add_argument(
        "--features_root_healthy",
        type=Path,
        default=Path("/filesystemHcog/features_cache_FIX2_20260222_061927"),
    )
    ap.add_argument(
        "--features_root_mechanism",
        type=Path,
        default=Path("/filesystemHcog/features_cache_FIX1_20260222_060109"),
    )
    ap.add_argument("--mechanism_dataset", type=str, default="ds003838")
    ap.add_argument("--sternberg_datasets", type=str, default="ds005095,ds003655,ds004117")
    ap.add_argument("--event_map", type=Path, default=Path("configs/lawc_event_map.yaml"))
    ap.add_argument("--mechanism_event_map", type=Path, default=Path("configs/mechanism_event_map.yaml"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--datasets_stage_config", type=Path, default=Path("configs/datasets_mechanism.yaml"))
    ap.add_argument("--clinical_bids_root", type=Path, default=Path("/filesystemHcog/clinical_bids"))
    ap.add_argument("--clinical_severity_csv", type=Path, default=Path("/filesystemHcog/clinical_bids/clinical_severity.csv"))
    ap.add_argument("--wall_hours", type=float, default=10.0)
    ap.add_argument("--lawc_n_perm", type=int, default=50000)
    ap.add_argument("--mechanism_n_perm", type=int, default=2000)
    ap.add_argument("--mechanism_seeds", type=str, default="0-49")
    ap.add_argument("--gpu_parallel_procs", type=int, default=6)
    ap.add_argument("--cpu_workers", type=int, default=32)
    ap.add_argument("--out_root", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    out_root = args.out_root
    if out_root is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_root = Path("/filesystemHcog/runs") / f"{ts}_NN_SOLID_1_2"

    if out_root.exists() and not args.resume:
        print(f"ERROR: out_root exists and --resume not set: {out_root}", file=sys.stderr, flush=True)
        return 1

    ctx = RunContext(
        out_root=out_root,
        audit_dir=out_root / "AUDIT",
        outzip_dir=out_root / "OUTZIP",
        data_root=args.data_root,
        features_root_healthy=args.features_root_healthy,
        features_root_mechanism=args.features_root_mechanism,
        mechanism_dataset=args.mechanism_dataset,
        sternberg_datasets=[x.strip() for x in str(args.sternberg_datasets).split(",") if x.strip()],
        event_map=args.event_map,
        mechanism_event_map=args.mechanism_event_map,
        config=args.config,
        datasets_stage_config=args.datasets_stage_config,
        clinical_bids_root=args.clinical_bids_root,
        clinical_severity_csv=args.clinical_severity_csv,
        wall_hours=float(args.wall_hours),
        lawc_n_perm=int(args.lawc_n_perm),
        mechanism_n_perm=int(args.mechanism_n_perm),
        mechanism_seeds=str(args.mechanism_seeds),
        gpu_parallel_procs=int(max(1, args.gpu_parallel_procs)),
        cpu_workers=int(max(1, args.cpu_workers)),
        resume=bool(args.resume),
        start_ts=time.time(),
        deadline_ts=time.time() + float(args.wall_hours) * 3600.0,
        stage_records=[],
        monitor_proc=None,
    )

    ctx.out_root.mkdir(parents=True, exist_ok=True)
    ctx.audit_dir.mkdir(parents=True, exist_ok=True)
    ctx.outzip_dir.mkdir(parents=True, exist_ok=True)

    run_status = "PASS"
    run_error = ""

    stages = [
        _stage_preflight,
        _stage_compile_gate,
        _stage_data_staging,
        _stage_mechanism_deep,
        _stage_lawc_ultradeep,
        _stage_effect_size_pack,
        _stage_clinical_translation,
        _stage_zip_bundle,
    ]

    try:
        for fn in stages:
            if _budget_exhausted(ctx) and fn.__name__ not in {"_stage_zip_bundle", "_stage_clinical_translation"}:
                # Soft budget guard: skip remaining heavy stages, continue to report/zip.
                stage_name = fn.__name__.replace("_stage_", "")
                now = time.time()
                dummy_log = ctx.audit_dir / f"{stage_name}.log"
                dummy_sum = ctx.audit_dir / f"{stage_name}_summary.json"
                _write_text(dummy_log, f"[{_iso_now()}] SKIP budget exhausted\n")
                _record_stage(
                    ctx,
                    stage=stage_name,
                    status="SKIP",
                    started=now,
                    rc=0,
                    log_path=dummy_log,
                    summary_path=dummy_sum,
                    command="budget_guard",
                    error="budget exhausted",
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
    finally:
        _stop_gpu_monitor(ctx)

    report_path = _build_report(ctx, run_status, run_error)

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
    print(f"FINAL_REPORT={report_path}", flush=True)
    print(f"BUNDLE_ZIP={ctx.outzip_dir / 'nn_solid_1_2_bundle.zip'}", flush=True)

    return 0 if run_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
