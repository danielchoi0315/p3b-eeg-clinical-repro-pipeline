#!/usr/bin/env python3
"""PEARL_SOLID2 orchestrator (ds004796 clinical/risk translation)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from aggregate_results import _compute_repo_fingerprint
except Exception:
    _compute_repo_fingerprint = None


@dataclass
class RunContext:
    out_root: Path
    audit_dir: Path
    outzip_dir: Path
    pearl_dir: Path
    data_root: Path
    dataset_id: str
    dataset_root: Path
    config: Path
    event_map: Path
    features_root: Path
    wall_hours: float
    lawc_n_perm: int
    risk_n_perm: int
    model_seeds: str
    gpu_parallel_procs: int
    cpu_workers: int
    resume: bool
    start_ts: float
    deadline_ts: float
    stage_records: List[Dict[str, Any]]
    stage_status: Dict[str, str]
    monitor_proc: Optional[subprocess.Popen]
    runtime_env: Dict[str, str]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tail(path: Path, n: int = 200) -> str:
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
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{_iso_now()}] CMD: {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True, env=env, check=False)
    return int(p.returncode)


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
    ctx.stage_status[stage] = status
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
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=8)
    finally:
        ctx.monitor_proc = None


def _wall_guard(ctx: RunContext, *, seconds_needed: float) -> bool:
    return (ctx.deadline_ts - time.time()) > float(seconds_needed)


def _stage_preflight(ctx: RunContext) -> Dict[str, Any]:
    stage = "preflight"
    started = time.time()
    log = ctx.audit_dir / f"{stage}.log"
    summary = ctx.audit_dir / f"{stage}_summary.json"
    outputs: List[Path] = []
    try:
        ctx.out_root.mkdir(parents=True, exist_ok=True)
        ctx.audit_dir.mkdir(parents=True, exist_ok=True)
        ctx.outzip_dir.mkdir(parents=True, exist_ok=True)
        ctx.pearl_dir.mkdir(parents=True, exist_ok=True)

        py = ctx.audit_dir / "python_version.txt"
        _write_text(py, sys.version + "\n")
        outputs.append(py)

        pipf = ctx.audit_dir / "pip_freeze.txt"
        p = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
        _write_text(pipf, p.stdout if p.returncode == 0 else p.stdout + "\n" + p.stderr)
        outputs.append(pipf)

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
            else {"repo_root": str(REPO_ROOT), "git_head": None, "repo_fingerprint_sha256": None, "file_count_hashed": 0}
        )
        repo_j = ctx.audit_dir / "repo_fingerprint.json"
        _write_json(repo_j, repo_fp)
        outputs.append(repo_j)

        _start_gpu_monitor(ctx)
        return _record_stage(
            ctx,
            stage=stage,
            status="PASS",
            rc=0,
            started=started,
            log_path=log,
            summary_path=summary,
            command="preflight",
            outputs=outputs,
        )
    except Exception as exc:
        _write_text(log, traceback.format_exc())
        _stop_gpu_monitor(ctx)
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log,
            summary_path=summary,
            command="preflight",
            error=str(exc),
        )


def _stage_compile_gate(ctx: RunContext) -> Dict[str, Any]:
    stage = "compile_gate"
    started = time.time()
    log = ctx.audit_dir / f"{stage}.log"
    summary = ctx.audit_dir / f"{stage}_summary.json"
    cmd = ["bash", "-lc", "find . -name '*.py' -print0 | xargs -0 python -m py_compile"]
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log, env=ctx.runtime_env)
    status = "PASS" if rc == 0 else "FAIL"
    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=rc,
        started=started,
        log_path=log,
        summary_path=summary,
        command="find . -name '*.py' -print0 | xargs -0 python -m py_compile",
        error="" if status == "PASS" else "compile gate failed",
    )


def _stage_stage_dataset(ctx: RunContext) -> Dict[str, Any]:
    stage = "stage_ds004796"
    started = time.time()
    log = ctx.audit_dir / f"{stage}.log"
    summary = ctx.audit_dir / f"{stage}_summary.json"
    manifest = ctx.audit_dir / "ds004796_manifest.json"
    cmd = [
        "bash",
        "scripts/pearl_stage_ds004796.sh",
        "--data_root",
        str(ctx.data_root),
        "--dataset_id",
        str(ctx.dataset_id),
        "--out_manifest",
        str(manifest),
    ]
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log, env=ctx.runtime_env)
    status = "PASS" if rc == 0 and manifest.exists() else "FAIL"
    err = ""
    if status != "PASS":
        err = f"dataset staging failed rc={rc}"
    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=rc if status == "PASS" else 1,
        started=started,
        log_path=log,
        summary_path=summary,
        command=" ".join(cmd),
        outputs=[manifest],
        error=err,
    )


def _stage_decode_mapping(ctx: RunContext) -> Dict[str, Any]:
    stage = "decode_mapping"
    started = time.time()
    log = ctx.audit_dir / f"{stage}.log"
    summary = ctx.audit_dir / f"{stage}_summary.json"
    out_dir = ctx.pearl_dir / "mapping_decode"
    out_dir.mkdir(parents=True, exist_ok=True)
    sum_json = out_dir / "mapping_decode_summary.json"
    cmd = [
        sys.executable,
        "scripts/pearl_decode_mapping.py",
        "--dataset_root",
        str(ctx.dataset_root),
        "--out_dir",
        str(out_dir),
        "--event_map_out",
        str(ctx.event_map),
        "--dataset_id",
        str(ctx.dataset_id),
        "--sample_subjects",
        "20",
    ]
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log, env=ctx.runtime_env)
    status = "FAIL"
    err = ""
    if rc == 0 and sum_json.exists():
        payload = json.loads(sum_json.read_text(encoding="utf-8"))
        s = str(payload.get("status", "FAIL")).upper()
        if s == "PASS":
            status = "PASS"
        elif s == "SKIP":
            status = "SKIP"
            err = str(payload.get("reason", "mapping skipped"))
        else:
            status = "FAIL"
            err = str(payload.get("reason", "mapping decode failed"))
    else:
        err = f"mapping decode command failed rc={rc}"

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=0 if status != "FAIL" else 1,
        started=started,
        log_path=log,
        summary_path=summary,
        command=" ".join(cmd),
        outputs=[sum_json, ctx.event_map, out_dir / "CANDIDATE_TABLE.csv", ctx.pearl_dir / "STOP_REASON.md"],
        error=err,
    )


def _stage_extract_features(ctx: RunContext) -> Dict[str, Any]:
    stage = "extract_features"
    started = time.time()
    log = ctx.audit_dir / f"{stage}.log"
    summary = ctx.audit_dir / f"{stage}_summary.json"
    out_sum = ctx.pearl_dir / "features_summary.json"
    trial_csv = ctx.pearl_dir / "trial_table.csv"
    if ctx.stage_status.get("decode_mapping") != "PASS":
        msg = "mapping not PASS; feature extraction skipped"
        _write_json(out_sum, {"status": "SKIP", "reason": msg})
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log,
            summary_path=summary,
            command="pearl_extract_features",
            outputs=[out_sum],
            error=msg,
        )

    cmd = [
        sys.executable,
        "scripts/pearl_extract_features.py",
        "--dataset_root",
        str(ctx.dataset_root),
        "--deriv_root",
        str(ctx.pearl_dir / "derivatives" / ctx.dataset_id),
        "--features_root",
        str(ctx.features_root),
        "--event_map",
        str(ctx.event_map),
        "--config",
        str(ctx.config),
        "--dataset_id",
        str(ctx.dataset_id),
        "--cpu_workers",
        str(ctx.cpu_workers),
        "--out_summary",
        str(out_sum),
    ]
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log, env=ctx.runtime_env)
    status = "PASS" if rc == 0 and out_sum.exists() and trial_csv.exists() else "FAIL"
    err = "" if status == "PASS" else f"feature extraction failed rc={rc}"
    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=rc if status == "PASS" else 1,
        started=started,
        log_path=log,
        summary_path=summary,
        command=" ".join(cmd),
        outputs=[out_sum, trial_csv],
        error=err,
    )


def _stage_lawc_audit(ctx: RunContext) -> Dict[str, Any]:
    stage = "lawc_ds004796"
    started = time.time()
    log = ctx.audit_dir / f"{stage}.log"
    summary = ctx.audit_dir / f"{stage}_summary.json"
    out_dir = ctx.pearl_dir / "lawc_ds004796"
    out_dir.mkdir(parents=True, exist_ok=True)
    if ctx.stage_status.get("extract_features") != "PASS":
        msg = "feature extraction not PASS; Law-C audit skipped"
        _write_json(summary, {"status": "SKIP", "reason": msg})
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log,
            summary_path=summary,
            command="pearl_lawc_audit",
            error=msg,
        )

    cmd = [
        sys.executable,
        "scripts/pearl_lawc_audit.py",
        "--features_root",
        str(ctx.features_root),
        "--out_dir",
        str(out_dir),
        "--dataset_id",
        str(ctx.dataset_id),
        "--n_perm",
        str(ctx.lawc_n_perm),
        "--min_trials",
        "20",
        "--n_control_shuffles",
        "256",
    ]
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log, env=ctx.runtime_env)
    ok = (out_dir / "locked_test_results.json").exists() and (out_dir / "negative_controls.csv").exists()
    status = "PASS" if rc == 0 and ok else "FAIL"
    err = "" if status == "PASS" else f"Law-C audit failed rc={rc}"
    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=rc if status == "PASS" else 1,
        started=started,
        log_path=log,
        summary_path=summary,
        command=" ".join(cmd),
        outputs=[out_dir / "locked_test_results.json", out_dir / "locked_test_results.csv", out_dir / "negative_controls.csv"],
        error=err,
    )


def _stage_normative_risk(ctx: RunContext) -> Dict[str, Any]:
    stage = "normative_risk"
    started = time.time()
    log = ctx.audit_dir / f"{stage}.log"
    summary = ctx.audit_dir / f"{stage}_summary.json"
    out_dir = ctx.pearl_dir
    trial_csv = ctx.pearl_dir / "trial_table.csv"
    if ctx.stage_status.get("extract_features") != "PASS":
        msg = "feature extraction not PASS; normative risk stage skipped"
        _write_json(summary, {"status": "SKIP", "reason": msg})
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log,
            summary_path=summary,
            command="pearl_normative_risk",
            error=msg,
        )

    cmd = [
        sys.executable,
        "scripts/pearl_normative_risk.py",
        "--trial_table_csv",
        str(trial_csv),
        "--participants_tsv",
        str(ctx.dataset_root / "participants.tsv"),
        "--out_dir",
        str(out_dir),
        "--seeds",
        str(ctx.model_seeds),
        "--workers",
        str(max(1, ctx.gpu_parallel_procs)),
        "--cv_folds",
        "5",
        "--n_perm",
        str(ctx.risk_n_perm),
    ]
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log, env=ctx.runtime_env)
    needed = [
        out_dir / "clinical_risk_results.csv",
        out_dir / "deviation_scores.csv",
        out_dir / "FIG_deviation_by_risk_group.png",
        out_dir / "FIG_deviation_vs_memory_scores.png",
        out_dir / "FIG_auc_risk_group.png",
        out_dir / "normative_seed_stability.json",
    ]
    status = "PASS" if rc == 0 and all(p.exists() for p in needed[:2]) else "FAIL"
    err = "" if status == "PASS" else f"normative/risk stage failed rc={rc}"
    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=rc if status == "PASS" else 1,
        started=started,
        log_path=log,
        summary_path=summary,
        command=" ".join(cmd),
        outputs=needed,
        error=err,
    )


def _stage_rest_optional(ctx: RunContext) -> Dict[str, Any]:
    stage = "rest_slowing_optional"
    started = time.time()
    log = ctx.audit_dir / f"{stage}.log"
    summary = ctx.audit_dir / f"{stage}_summary.json"
    out_dir = ctx.pearl_dir / "resting_state"
    out_dir.mkdir(parents=True, exist_ok=True)

    if ctx.stage_status.get("extract_features") != "PASS":
        msg = "feature extraction not PASS; rest stage skipped"
        _write_json(summary, {"status": "SKIP", "reason": msg})
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log,
            summary_path=summary,
            command="pearl_rest_slowing",
            error=msg,
        )

    # Keep budget for final report + zip.
    if not _wall_guard(ctx, seconds_needed=2400):
        msg = "wall-clock budget guard skipped optional rest stage"
        _write_json(summary, {"status": "SKIP", "reason": msg})
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log,
            summary_path=summary,
            command="pearl_rest_slowing",
            error=msg,
        )

    cmd = [
        sys.executable,
        "scripts/pearl_rest_slowing.py",
        "--dataset_root",
        str(ctx.dataset_root),
        "--participants_tsv",
        str(ctx.dataset_root / "participants.tsv"),
        "--out_dir",
        str(out_dir),
        "--max_subjects",
        "32",
        "--n_perm",
        "5000",
    ]
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log, env=ctx.runtime_env)
    rest_summary = out_dir / "rest_summary.json"
    status = "FAIL"
    err = ""
    if rc == 0 and rest_summary.exists():
        payload = json.loads(rest_summary.read_text(encoding="utf-8"))
        s = str(payload.get("status", "SKIP")).upper()
        if s == "PASS":
            status = "PASS"
        elif s == "SKIP":
            status = "SKIP"
            err = str(payload.get("reason", "rest stage skipped"))
        else:
            status = "FAIL"
            err = str(payload.get("reason", "rest stage failed"))
    else:
        err = f"rest stage command failed rc={rc}"
    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=0 if status != "FAIL" else 1,
        started=started,
        log_path=log,
        summary_path=summary,
        command=" ".join(cmd),
        outputs=[out_dir],
        error=err,
    )


def _stage_zip_bundle(ctx: RunContext) -> Dict[str, Any]:
    stage = "zip_bundle"
    started = time.time()
    log = ctx.audit_dir / f"{stage}.log"
    summary = ctx.audit_dir / f"{stage}_summary.json"
    zpath = ctx.outzip_dir / "pearl_solid2_bundle.zip"
    err = ""
    added = 0
    try:
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root in [ctx.audit_dir, ctx.pearl_dir]:
                if not root.exists():
                    continue
                for p in sorted(root.rglob("*")):
                    if p.is_dir():
                        continue
                    rel = p.relative_to(ctx.out_root)
                    zf.write(p, rel.as_posix())
                    added += 1
            if ctx.event_map.exists():
                rel = Path("AUDIT") / "pearl_event_map.yaml"
                zf.write(ctx.event_map, rel.as_posix())
                added += 1
        status = "PASS"
        rc = 0
    except Exception as exc:
        status = "FAIL"
        rc = 1
        err = str(exc)

    _write_json(summary, {"status": status, "zip": str(zpath), "n_files": int(added), "error": err})
    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        rc=rc,
        started=started,
        log_path=log,
        summary_path=summary,
        command="zip bundle",
        outputs=[zpath],
        error=err,
    )


def _build_report(ctx: RunContext, run_status: str, run_error: str) -> Path:
    report = ctx.audit_dir / "PEARL_SOLID2_REPORT.md"
    stage_rows = [
        f"| {r['stage']} | {r['status']} | {r['returncode']} | {r['elapsed_sec']:.1f} | {Path(r['log']).name} | {Path(r['summary']).name} |"
        for r in ctx.stage_records
    ]

    manifest_rows = ["| <missing> | <missing> | <missing> | <missing> | <missing> |"]
    man = ctx.audit_dir / "ds004796_manifest.json"
    if man.exists():
        try:
            payload = json.loads(man.read_text(encoding="utf-8"))
            c = payload.get("counts", {})
            manifest_rows = [
                f"| {payload.get('dataset_id', ctx.dataset_id)} | {payload.get('checked_out_commit')} | "
                f"{c.get('sternberg_events_tsv')} | {c.get('sternberg_eeg_headers')} | {c.get('sourcedata_sternberg_logs')} |"
            ]
        except Exception:
            pass

    map_rows = ["| <missing> | <missing> | <missing> | <missing> |"]
    msum = ctx.pearl_dir / "mapping_decode" / "mapping_decode_summary.json"
    if msum.exists():
        p = json.loads(msum.read_text(encoding="utf-8"))
        map_rows = [
            f"| {ctx.dataset_id} | {p.get('status')} | {p.get('reason', '')} | `{json.dumps(p.get('mapping', {}), sort_keys=True)}` |"
        ]

    lawc_rows = ["| <missing> | <missing> | <missing> | <missing> | <missing> |"]
    lawc_json = ctx.pearl_dir / "lawc_ds004796" / "locked_test_results.json"
    if lawc_json.exists():
        p = json.loads(lawc_json.read_text(encoding="utf-8"))
        lawc_rows = [
            f"| {p.get('dataset_id')} | {p.get('median_rho')} | {p.get('p_value_perm')} | "
            f"{p.get('x_degrade_pass')} | {p.get('y_degrade_pass')} |"
        ]

    risk_rows = ["| <none> | <none> | <none> | <none> | <none> | <none> |"]
    risk_csv = ctx.pearl_dir / "clinical_risk_results.csv"
    if risk_csv.exists():
        try:
            df = pd.read_csv(risk_csv)
            if not df.empty:
                risk_rows = []
                for _, r in df.iterrows():
                    risk_rows.append(
                        f"| {r.get('endpoint')} | {r.get('type')} | {r.get('status')} | {r.get('n', '')} | "
                        f"{r.get('beta', '')} | {r.get('perm_p', '')} / {r.get('perm_q', '')} |"
                    )
        except Exception:
            pass

    lines = [
        "# PEARL SOLID2 REPORT",
        "",
        f"- Output root: `{ctx.out_root}`",
        f"- Run status: `{run_status}`",
        f"- Resume: `{ctx.resume}`",
        f"- Dataset: `{ctx.dataset_id}`",
        f"- Model seeds: `{ctx.model_seeds}`",
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
            "## Dataset staging manifest",
            "| Dataset | Commit | Sternberg events | Sternberg EEG headers | Sternberg logs |",
            "|---|---|---:|---:|---:|",
            *manifest_rows,
            "",
            "## Mapping outcomes",
            "| Dataset | Status | Reason | Mapping |",
            "|---|---|---|---|",
            *map_rows,
            f"- Candidate table: `{ctx.pearl_dir / 'mapping_decode' / 'CANDIDATE_TABLE.csv'}`",
            f"- Mapping summary: `{ctx.pearl_dir / 'mapping_decode' / 'mapping_decode_summary.json'}`",
            "",
            "## Law-C ds004796 (locked endpoint)",
            "| Dataset | Median rho | Perm p | X-control degrade | Y-control degrade |",
            "|---|---:|---:|---|---|",
            *lawc_rows,
            "",
            "## Clinical/risk endpoints",
            "| Endpoint | Type | Status | N | Beta | Perm p / q |",
            "|---|---|---|---:|---:|---|",
            *risk_rows,
            "",
            "## Artefacts",
            f"- Repo fingerprint: `{ctx.audit_dir / 'repo_fingerprint.json'}`",
            f"- GPU monitor CSV: `{ctx.audit_dir / 'nvidia_smi_1hz.csv'}`",
            f"- Event map: `{ctx.event_map}`",
            f"- Feature summary: `{ctx.pearl_dir / 'features_summary.json'}`",
            f"- Trial table: `{ctx.pearl_dir / 'trial_table.csv'}`",
            f"- Law-C outputs: `{ctx.pearl_dir / 'lawc_ds004796'}`",
            f"- Risk outputs: `{ctx.pearl_dir}`",
            f"- Rest exploratory: `{ctx.pearl_dir / 'resting_state'}`",
            f"- Bundle: `{ctx.outzip_dir / 'pearl_solid2_bundle.zip'}`",
        ]
    )
    _write_text(report, "\n".join(lines) + "\n")
    return report


def _apply_targeted_fix(ctx: RunContext, rec: Dict[str, Any]) -> bool:
    txt = _tail(Path(rec["log"]), 300)
    changed = False
    if re.search(r"VLEN strings|embedded NULL|HDF5|Unicode", txt, flags=re.IGNORECASE):
        ctx.runtime_env["HDF5_USE_FILE_LOCKING"] = "FALSE"
        changed = True
    if re.search(r"IndexError|out of bounds|KeyError", txt, flags=re.IGNORECASE):
        ctx.cpu_workers = max(4, ctx.cpu_workers // 2)
        changed = True
    if re.search(r"No module named|ImportError|NameError", txt, flags=re.IGNORECASE):
        changed = True
    if changed:
        fix_log = ctx.audit_dir / "AUTO_FIX_LOG.md"
        with fix_log.open("a", encoding="utf-8") as f:
            f.write(f"## {_iso_now()} stage={rec['stage']}\n")
            f.write("Applied targeted runtime fix heuristics before retry.\n\n")
    return changed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, default=Path("/filesystemHcog/openneuro"))
    ap.add_argument("--dataset_id", type=str, default="ds004796")
    ap.add_argument("--out_root", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--event_map", type=Path, default=Path("configs/pearl_event_map.yaml"))
    ap.add_argument("--features_root", type=Path, default=None)
    ap.add_argument("--wall_hours", type=float, default=10.0)
    ap.add_argument("--lawc_n_perm", type=int, default=50000)
    ap.add_argument("--risk_n_perm", type=int, default=20000)
    ap.add_argument("--model_seeds", type=str, default="0-199")
    ap.add_argument("--gpu_parallel_procs", type=int, default=8)
    ap.add_argument("--cpu_workers", type=int, default=32)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.out_root is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_root = Path("/filesystemHcog/runs") / f"{ts}_PEARL_SOLID2"
    else:
        out_root = args.out_root

    if out_root.exists() and not args.resume:
        print(f"ERROR: out_root exists and --resume is not set: {out_root}", file=sys.stderr)
        return 1

    run_ts = out_root.name.split("_", 1)[0]
    features_root = args.features_root or Path(f"/filesystemHcog/features_cache_PEARL_SOLID2_{run_ts}")

    env = dict(os.environ)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(REPO_ROOT / "src")

    ctx = RunContext(
        out_root=out_root,
        audit_dir=out_root / "AUDIT",
        outzip_dir=out_root / "OUTZIP",
        pearl_dir=out_root / "PEARLPack",
        data_root=args.data_root,
        dataset_id=str(args.dataset_id),
        dataset_root=args.data_root / str(args.dataset_id),
        config=args.config,
        event_map=args.event_map,
        features_root=features_root,
        wall_hours=float(args.wall_hours),
        lawc_n_perm=int(args.lawc_n_perm),
        risk_n_perm=int(args.risk_n_perm),
        model_seeds=str(args.model_seeds),
        gpu_parallel_procs=int(max(1, args.gpu_parallel_procs)),
        cpu_workers=int(max(1, args.cpu_workers)),
        resume=bool(args.resume),
        start_ts=time.time(),
        deadline_ts=time.time() + float(args.wall_hours) * 3600.0,
        stage_records=[],
        stage_status={},
        monitor_proc=None,
        runtime_env=env,
    )
    ctx.out_root.mkdir(parents=True, exist_ok=True)
    ctx.audit_dir.mkdir(parents=True, exist_ok=True)
    ctx.outzip_dir.mkdir(parents=True, exist_ok=True)
    ctx.pearl_dir.mkdir(parents=True, exist_ok=True)

    stages: List[Tuple[str, Any]] = [
        ("preflight", _stage_preflight),
        ("compile_gate", _stage_compile_gate),
        ("stage_ds004796", _stage_stage_dataset),
        ("decode_mapping", _stage_decode_mapping),
        ("extract_features", _stage_extract_features),
        ("lawc_ds004796", _stage_lawc_audit),
        ("normative_risk", _stage_normative_risk),
        ("rest_slowing_optional", _stage_rest_optional),
        ("zip_bundle", _stage_zip_bundle),
    ]

    run_status = "PASS"
    run_error = ""
    try:
        for stage_name, fn in stages:
            # Resume optimization: reuse successful stage records from prior attempts.
            # Keep preflight live so monitor/provenance are refreshed for each invocation.
            if ctx.resume and stage_name != "preflight":
                status_path = ctx.audit_dir / f"{stage_name}.status"
                summary_path = ctx.audit_dir / f"{stage_name}_summary.json"
                if status_path.exists() and summary_path.exists():
                    prev_status = status_path.read_text(encoding="utf-8", errors="ignore").strip().upper()
                    if prev_status in {"PASS", "SKIP"}:
                        try:
                            prev_rec = json.loads(summary_path.read_text(encoding="utf-8"))
                        except Exception:
                            prev_rec = {
                                "stage": stage_name,
                                "status": prev_status,
                                "returncode": 0,
                                "started_at": _iso_now(),
                                "ended_at": _iso_now(),
                                "elapsed_sec": 0.0,
                                "log": str(ctx.audit_dir / f"{stage_name}.log"),
                                "summary": str(summary_path),
                                "command": "resume-skip",
                                "outputs": [],
                                "error": "",
                            }
                        prev_rec["stage"] = stage_name
                        prev_rec["status"] = prev_status
                        ctx.stage_records.append(prev_rec)
                        ctx.stage_status[stage_name] = prev_status
                        continue

            attempt = 1
            while True:
                rec = fn(ctx)
                if rec["status"] != "FAIL":
                    break
                if attempt >= 3:
                    run_status = "FAIL"
                    run_error = rec.get("error", "stage failed")
                    break
                if not _apply_targeted_fix(ctx, rec):
                    run_status = "FAIL"
                    run_error = rec.get("error", "stage failed")
                    break
                attempt += 1
            if run_status == "FAIL":
                break
    except Exception as exc:
        run_status = "FAIL"
        run_error = f"{type(exc).__name__}: {exc}\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    finally:
        _stop_gpu_monitor(ctx)

    report = _build_report(ctx, run_status, run_error)
    _write_json(
        ctx.audit_dir / "run_status.json",
        {
            "status": run_status,
            "error": run_error,
            "out_root": str(ctx.out_root),
            "report": str(report),
            "stage_records": ctx.stage_records,
        },
    )

    print(f"OUT_ROOT={ctx.out_root}")
    print(f"FINAL_REPORT={report}")
    print(f"BUNDLE_ZIP={ctx.outzip_dir / 'pearl_solid2_bundle.zip'}")
    return 0 if run_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
