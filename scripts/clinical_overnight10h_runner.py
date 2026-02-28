#!/usr/bin/env python3
"""Clinical Overnight 10h runner.

Orchestrates:
- compile gate
- clinical data staging (if missing)
- participants->severity build
- auditable clinical mapping decode (events/json/codebook/task scripts)
- clinical extraction + many-seed normative application
- group endpoint inference + figures
- final report + zip
"""

from __future__ import annotations

import argparse
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from common.lawc_audit import bh_fdr  # noqa: E402
from p3b_pipeline.clinical import subject_load_deviation_score  # noqa: E402

try:
    from aggregate_results import _compute_repo_fingerprint
except Exception:
    _compute_repo_fingerprint = None


@dataclass
class RunContext:
    out_root: Path
    audit_dir: Path
    outzip_dir: Path
    clinical_pack_dir: Path

    data_root: Path
    clinical_bids_root: Path
    clinical_severity_csv: Path
    features_root_healthy: Path
    features_root_clinical: Path

    config: Path
    lawc_event_map: Path
    clinical_event_map: Path
    datasets: List[str]
    model_seeds: List[int]

    wall_hours: float
    rt_n_perm: int
    gpu_parallel_procs: int
    cpu_workers: int
    resume: bool

    start_ts: float
    deadline_ts: float

    stage_records: List[Dict[str, Any]]
    monitor_proc: Optional[subprocess.Popen]
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
    txt = path.read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()
    return "\n".join(lines[-max(1, int(n)) :])


def _run_cmd(
    cmd: List[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    check: bool = False,
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
    if check and rc != 0:
        raise RuntimeError(f"command failed rc={rc}: {' '.join(cmd)}")
    return rc


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


def _discover_clinical_datasets(clinical_root: Path) -> List[Path]:
    if not clinical_root.exists():
        return []
    out: List[Path] = []
    for p in sorted(clinical_root.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if (p / "participants.tsv").exists():
            out.append(p)
            continue
        if any(p.rglob("*_events.tsv")) or any(p.rglob("*_events.tsv.gz")):
            out.append(p)
    return out


_EEG_FILE_RE = re.compile(r"_eeg\.(edf|bdf|set|vhdr|eeg|fif|gdf|fdt)(\.gz)?$", re.IGNORECASE)


def _count_event_files(dataset_root: Path) -> int:
    return int(len(list(dataset_root.rglob("*_events.tsv"))) + len(list(dataset_root.rglob("*_events.tsv.gz"))))


def _count_eeg_payload_files(dataset_root: Path) -> int:
    n = 0
    for p in dataset_root.rglob("*"):
        if not p.is_file():
            continue
        if _EEG_FILE_RE.search(p.name):
            n += 1
    return int(n)


def _dataset_payload_ready(dataset_root: Path) -> bool:
    if not (dataset_root / "participants.tsv").exists():
        return False
    if _count_event_files(dataset_root) <= 0:
        return False
    if _count_eeg_payload_files(dataset_root) <= 0:
        return False
    return True


def _extract_subject_key_tokens(subject_key: str) -> Tuple[str, str]:
    s = str(subject_key)
    m_sub = re.search(r"sub-([^:]+)", s)
    m_ses = re.search(r"ses-([^:]+)", s)
    subj = m_sub.group(1) if m_sub else s
    ses = m_ses.group(1) if m_ses else ""
    return subj, ses


def _safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _seed_done(module_root: Path, seed: int) -> bool:
    path = module_root / f"seed_{seed}" / "reports" / "normative" / f"seed_{seed}" / "normative_metrics.json"
    return path.exists()


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
    boots = []
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


def _encode_sex(series: pd.Series) -> np.ndarray:
    vals = series.fillna("").astype(str).str.strip().str.lower()
    out = np.full(len(vals), np.nan, dtype=float)
    for i, v in enumerate(vals):
        if v in {"m", "male", "1"}:
            out[i] = 1.0
        elif v in {"f", "female", "0"}:
            out[i] = 0.0
    return out


def _linear_effect_with_covariates(df: pd.DataFrame, score_col: str) -> Tuple[float, float]:
    import statsmodels.api as sm

    work = df.copy()
    work["group"] = pd.to_numeric(work["group"], errors="coerce")
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    if "age" in work.columns:
        work["age"] = pd.to_numeric(work["age"], errors="coerce")
    else:
        work["age"] = np.nan
    if "sex" in work.columns:
        work["sex_num"] = _encode_sex(work["sex"])
    else:
        work["sex_num"] = np.nan

    cols = [score_col, "group", "age", "sex_num"]
    fit = work[cols].dropna().copy()
    if len(fit) < 8 or fit["group"].nunique() < 2:
        return float("nan"), float("nan")

    y = fit[score_col].astype(float)
    X = sm.add_constant(fit[["group", "age", "sex_num"]].astype(float), has_constant="add")
    model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    res = model.fit()
    beta = float(res.params.get("group", np.nan))
    return beta, float(len(fit))


def _perm_p_group_effect(df: pd.DataFrame, score_col: str, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(int(seed))
    obs, n_fit = _linear_effect_with_covariates(df, score_col)
    if not np.isfinite(obs):
        return float("nan")

    null = np.full(int(n_perm), np.nan, dtype=float)
    arr_group = pd.to_numeric(df["group"], errors="coerce").to_numpy(dtype=float)
    for i in range(int(n_perm)):
        perm = arr_group.copy()
        rng.shuffle(perm)
        dfp = df.copy()
        dfp["group"] = perm
        b, _ = _linear_effect_with_covariates(dfp, score_col)
        null[i] = b

    finite = null[np.isfinite(null)]
    if finite.size == 0:
        return float("nan")
    return float((1.0 + np.sum(np.abs(finite) >= abs(obs))) / (1.0 + finite.size))


def _event_files(dataset_root: Path) -> List[Path]:
    return sorted(list(dataset_root.rglob("*_events.tsv")) + list(dataset_root.rglob("*_events.tsv.gz")))


def _write_stop_reason_dataset(path: Path, dataset_id: str, reason: str, details: Dict[str, Any]) -> None:
    lines = [
        f"# STOP_REASON {dataset_id}",
        "",
        "## Why skipped",
        reason,
        "",
        "## Diagnostics",
        "```json",
        json.dumps(details, indent=2),
        "```",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _infer_mapping_for_dataset(dataset_root: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], str]:
    ds = dataset_root.name
    efiles = _event_files(dataset_root)
    diag: Dict[str, Any] = {"dataset_id": ds, "n_event_files": len(efiles), "candidates": []}
    if not efiles:
        return None, diag, "no *_events.tsv files found"

    samples: List[pd.DataFrame] = []
    for fp in efiles[:20]:
        try:
            df = pd.read_csv(fp, sep="\t")
        except Exception:
            continue
        df["__source__"] = str(fp)
        samples.append(df)
    if not samples:
        return None, diag, "failed to read sampled events files"

    sdf = pd.concat(samples, axis=0, ignore_index=True)

    load_cols = [
        "memory_load",
        "set_size",
        "setsize",
        "load",
        "n_items",
        "memory_cond",
        "value",
        "trial_type",
        "condition",
    ]

    load_options: List[Dict[str, Any]] = []
    for c in load_cols:
        if c not in sdf.columns:
            continue
        if c == "trial_type":
            extracted = sdf[c].astype(str).str.extract(r"(\d+)", expand=False)
            v = pd.to_numeric(extracted, errors="coerce")
        elif c == "value":
            v = pd.to_numeric(sdf[c], errors="coerce")
            # avoid event code columns with huge spread if likely marker codes
            if v.notna().mean() > 0:
                u = pd.Series(v.dropna().unique())
                if u.nunique() > 20:
                    continue
        else:
            v = pd.to_numeric(sdf[c], errors="coerce")
        finite_rate = float(v.notna().mean())
        n_unique = int(v.dropna().nunique())
        if finite_rate >= 0.5 and 2 <= n_unique <= 12:
            score = finite_rate * math.log1p(n_unique)
            load_options.append({"load_column": c, "finite_rate": finite_rate, "n_unique": n_unique, "score": score})

    if not load_options:
        return None, diag, "no unambiguous load column candidates (need finite_rate>=0.5 and 2-12 unique levels)"

    load_options = sorted(load_options, key=lambda x: x["score"], reverse=True)
    best_load = load_options[0]
    diag["load_options"] = load_options

    # Event filter candidates that are explicitly probe/test-like.
    string_cols = [c for c in ["trial_type", "task_role", "event_type", "condition", "value"] if c in sdf.columns]
    kw = ["probe", "test", "target", "recall", "memory", "nback"]

    event_candidates: List[Dict[str, Any]] = []
    for c in string_cols:
        s = sdf[c].astype(str)
        for token in kw:
            mask = s.str.contains(token, case=False, regex=False, na=False)
            n = int(mask.sum())
            if n < 50:
                continue
            event_filter = f"{c}.str.contains('{token}', case=False, na=False)"
            sub = sdf.loc[mask].copy()

            lc = best_load["load_column"]
            if lc == "trial_type":
                lv = pd.to_numeric(sub[lc].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
            else:
                lv = pd.to_numeric(sub[lc], errors="coerce")
            load_rate = float(lv.notna().mean())
            load_levels = int(lv.dropna().nunique())
            if load_rate < 0.6 or load_levels < 2:
                continue

            score = n * load_rate * math.log1p(load_levels)
            event_candidates.append(
                {
                    "event_filter": event_filter,
                    "column": c,
                    "token": token,
                    "n_selected": n,
                    "load_rate": load_rate,
                    "load_levels": load_levels,
                    "score": score,
                }
            )

    if not event_candidates:
        return None, diag, "no probe/test-like event_filter candidate met load support criteria"

    event_candidates = sorted(event_candidates, key=lambda x: x["score"], reverse=True)
    diag["event_candidates"] = event_candidates

    best_event = event_candidates[0]
    ambiguous = False
    if len(event_candidates) > 1:
        second = event_candidates[1]
        if second["score"] >= 0.9 * best_event["score"] and second["event_filter"] != best_event["event_filter"]:
            ambiguous = True

    if ambiguous:
        return None, diag, "ambiguous event_filter candidates with similar support; manual mapping required"

    mapping = {
        "event_filter": best_event["event_filter"],
        "load_column": best_load["load_column"],
    }

    # If load column is text, add regex parse help.
    if best_load["load_column"] in {"trial_type", "condition"}:
        mapping["load_regex"] = r"(\d+)"

    # Optional RT column if available and non-missing.
    for rtc in ["response_time", "reaction_time", "rt", "RT"]:
        if rtc in sdf.columns:
            rtv = pd.to_numeric(sdf[rtc], errors="coerce")
            if float(rtv.notna().mean()) >= 0.3:
                mapping["rt_column"] = rtc
                break

    return mapping, diag, ""


def _ensure_compile_gate(ctx: RunContext) -> Dict[str, Any]:
    stage = "compile_gate"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"
    cmd = ["bash", "-lc", "find . -name '*.py' -print0 | xargs -0 python -m py_compile"]
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path)
    status = "PASS" if rc == 0 else "FAIL"
    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        started=started,
        rc=rc,
        log_path=log_path,
        summary_path=summary_path,
        command="find . -name '*.py' -print0 | xargs -0 python -m py_compile",
        error="" if status == "PASS" else "compile gate failed",
    )


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
        ctx.clinical_pack_dir.mkdir(parents=True, exist_ok=True)

        pyv = ctx.audit_dir / "python_version.txt"
        _write_text(pyv, sys.version + "\n")
        outputs.append(pyv)

        pip_freeze = ctx.audit_dir / "pip_freeze.txt"
        r = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
        _write_text(pip_freeze, r.stdout if r.returncode == 0 else r.stdout + "\n" + r.stderr)
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
        repo_path = ctx.audit_dir / "repo_fingerprint.json"
        _write_json(repo_path, repo_fp)
        outputs.append(repo_path)

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


def _stage_ensure_clinical_data(ctx: RunContext) -> Dict[str, Any]:
    stage = "ensure_clinical_data"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    ds_dirs_all = _discover_clinical_datasets(ctx.clinical_bids_root)
    requested = set(ctx.datasets)
    ds_dirs = [d for d in ds_dirs_all if d.name in requested]
    ready_dirs = [d for d in ds_dirs if _dataset_payload_ready(d)]
    hashes_json = ctx.audit_dir / "clinical_dataset_hashes.json"

    used_existing = len(ready_dirs) == len(requested) and len(requested) > 0
    rc = 0
    cmd_desc = "use existing clinical_bids"

    if not used_existing:
        cmd = [
            "bash",
            "scripts/clinical_stage_openneuro.sh",
            "--clinical_bids_root",
            str(ctx.clinical_bids_root),
            "--audit_json",
            str(hashes_json),
            "--datasets",
            ",".join(ctx.datasets),
        ]
        cmd_desc = " ".join(cmd)
        rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path)
        ds_dirs_all = _discover_clinical_datasets(ctx.clinical_bids_root)
        ds_dirs = [d for d in ds_dirs_all if d.name in requested]
        ready_dirs = [d for d in ds_dirs if _dataset_payload_ready(d)]

    status = "PASS"
    error = ""
    if rc != 0:
        status = "FAIL"
        error = "clinical staging command failed"
    elif not ds_dirs:
        status = "FAIL"
        error = f"requested dataset folders not found in {ctx.clinical_bids_root}: {sorted(requested)}"
    elif len(ready_dirs) != len(requested):
        missing = sorted(requested - {d.name for d in ready_dirs})
        status = "FAIL"
        error = f"requested datasets missing ready payload (participants+events+eeg): {missing}"

    # When using existing clinical root, still snapshot dataset hashes.
    if status == "PASS" and used_existing:
        payload = {
            "timestamp_utc": _iso_now(),
            "clinical_bids_root": str(ctx.clinical_bids_root),
            "datasets": [],
            "source": "existing",
        }
        for d in ds_dirs:
            commit = "<unavailable>"
            try:
                r = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
                if r.returncode == 0:
                    commit = r.stdout.strip()
            except Exception:
                pass
            n_events = _count_event_files(d)
            n_eeg = _count_eeg_payload_files(d)
            payload_ready = bool(_dataset_payload_ready(d))
            payload["datasets"].append(
                {
                    "dataset_id": d.name,
                    "path": str(d),
                    "checked_out_commit": commit,
                    "n_event_files": int(n_events),
                    "n_eeg_files": int(n_eeg),
                    "payload_ready": payload_ready,
                }
            )
        _write_json(hashes_json, payload)

    _write_json(
        summary_path,
        {
            "status": status,
            "returncode": rc,
            "used_existing": used_existing,
            "clinical_bids_root": str(ctx.clinical_bids_root),
            "datasets_requested": list(ctx.datasets),
            "n_datasets": len(ds_dirs),
            "n_ready_payload_datasets": len(ready_dirs),
            "dataset_dirs": [str(d) for d in ds_dirs],
            "ready_dataset_dirs": [str(d) for d in ready_dirs],
            "hashes_json": str(hashes_json),
            "error": error,
        },
    )

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        started=started,
        rc=rc if status == "PASS" else 1,
        log_path=log_path,
        summary_path=summary_path,
        command=cmd_desc,
        outputs=[hashes_json],
        error=error,
    )


def _stage_build_severity(ctx: RunContext) -> Dict[str, Any]:
    stage = "build_severity"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    summary_json = ctx.audit_dir / "clinical_severity_build.json"
    cmd = [
        sys.executable,
        "scripts/build_clinical_severity_from_participants.py",
        "--clinical_bids_root",
        str(ctx.clinical_bids_root),
        "--out_csv",
        str(ctx.clinical_severity_csv),
        "--summary_json",
        str(summary_json),
    ]
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path)

    status = "PASS"
    err = ""
    if rc != 0:
        status = "FAIL"
        err = "severity builder failed"
    elif not ctx.clinical_severity_csv.exists():
        status = "FAIL"
        err = "clinical_severity.csv missing after builder"
    else:
        try:
            df = pd.read_csv(ctx.clinical_severity_csv)
        except Exception as exc:
            status = "FAIL"
            err = f"failed reading clinical_severity.csv: {exc}"
            df = pd.DataFrame()
        if status == "PASS" and df.empty:
            status = "FAIL"
            err = "clinical_severity.csv empty"

    _write_json(
        summary_path,
        {
            "status": status,
            "returncode": rc,
            "severity_csv": str(ctx.clinical_severity_csv),
            "summary_json": str(summary_json),
            "error": err,
        },
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
        outputs=[ctx.clinical_severity_csv, summary_json],
        error=err,
    )


def _stage_decode_mapping(ctx: RunContext) -> Dict[str, Any]:
    stage = "decode_mapping"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    decode_dir = ctx.clinical_pack_dir / "mapping_decode"
    decode_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/clinical_decode_mapping.py",
        "--clinical_bids_root",
        str(ctx.clinical_bids_root),
        "--datasets",
        ",".join(ctx.datasets),
        "--out_dir",
        str(decode_dir),
        "--out_event_map",
        str(ctx.clinical_event_map),
        "--base_event_map",
        str(ctx.lawc_event_map),
        "--sample_subjects",
        "20",
        "--random_seed",
        "0",
    ]
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, env=ctx.runtime_env)

    summary_json = decode_dir / "mapping_decode_summary.json"
    candidate_csv = decode_dir / "CANDIDATE_TABLE.csv"
    rows: List[Dict[str, Any]] = []
    status = "FAIL"
    reason = ""
    if rc != 0:
        reason = "clinical_decode_mapping.py failed"
    elif not summary_json.exists():
        reason = "mapping_decode_summary.json missing"
    else:
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        rows = list(payload.get("rows", []))
        status = "PASS" if any(r.get("status") == "PASS" for r in rows) else "SKIP"
        reason = ""

    # Normalize to legacy mapping_status path consumed by extraction stage.
    mapping_status_json = ctx.clinical_pack_dir / "mapping_status.json"
    _write_json(
        mapping_status_json,
        {
            "status": status if status != "FAIL" else "FAIL",
            "clinical_event_map": str(ctx.clinical_event_map),
            "rows": rows,
            "mapping_decode_summary_json": str(summary_json),
            "candidate_table_csv": str(candidate_csv),
        },
    )

    _write_json(
        summary_path,
        {
            "status": status if status != "FAIL" else "FAIL",
            "returncode": rc,
            "reason": reason,
            "mapping_decode_summary_json": str(summary_json),
            "candidate_table_csv": str(candidate_csv),
            "mapping_status_json": str(mapping_status_json),
            "rows": rows,
        },
    )

    if status == "FAIL":
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            started=started,
            rc=1,
            log_path=log_path,
            summary_path=summary_path,
            command=" ".join(cmd),
            outputs=[summary_json, candidate_csv, ctx.clinical_event_map, mapping_status_json],
            error=reason,
        )

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        started=started,
        rc=0,
        log_path=log_path,
        summary_path=summary_path,
        command=" ".join(cmd),
        outputs=[summary_json, candidate_csv, ctx.clinical_event_map, mapping_status_json],
        error="",
    )


def _run_seed_process(
    *,
    dataset_id: str,
    seed: int,
    features_root: Path,
    module_root: Path,
    config: Path,
    severity_csv: Path,
    env: Dict[str, str],
) -> subprocess.Popen:
    cmd = [
        "bash",
        "scripts/run_module.sh",
        "--module",
        "04",
        "--features_root",
        str(features_root),
        "--out_root",
        str(module_root),
        "--config",
        str(config),
        "--seeds",
        str(seed),
        "--healthy_cohort",
        "healthy",
        "--clinical_cohort",
        "clinical",
        "--healthy_dataset_ids",
        dataset_id,
        "--severity_csv",
        str(severity_csv),
    ]
    p = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, text=True)
    return p


def _load_subject_group_for_dataset(severity_df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    df = severity_df.copy()
    df = df[df["dataset_id"].astype(str) == str(dataset_id)].copy()
    if df.empty:
        return df
    sid = df["subject_id"].astype(str).str.replace(r"^sub-", "", regex=True).str.strip()
    # BIDS subject labels are often zero-padded (e.g., 001) while participants/group
    # tables may store integers (e.g., 1). Normalize numeric IDs for extractor matching.
    df["subject_id"] = sid.map(lambda x: x.zfill(3) if x.isdigit() else x)
    df["group"] = pd.to_numeric(df["group"], errors="coerce")
    return df


def _prepare_module04_severity_csv(features_root: Path, dataset_id: str, group_df: pd.DataFrame, out_csv: Path) -> Path:
    import h5py

    rows = []
    ds_root = features_root / dataset_id
    for fp in sorted(ds_root.rglob("*.h5")):
        with h5py.File(fp, "r") as h:
            sk = ""
            if "subject_key" in h:
                v = np.asarray(h["subject_key"]).astype(str)
                sk = str(v[0]) if v.size else ""
            if not sk:
                sk = str(h.attrs.get("subject_key", ""))
            if not sk:
                continue
            subj, _ses = _extract_subject_key_tokens(sk)
            rows.append({"subject": sk, "subject_id": subj})

    if not rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["subject", "severity"]).to_csv(out_csv, index=False)
        return out_csv

    m = pd.DataFrame(rows).drop_duplicates(subset=["subject"], keep="first")
    g = group_df[["subject_id", "group"]].dropna().drop_duplicates(subset=["subject_id"], keep="first")
    m = m.merge(g, on="subject_id", how="left")
    m = m.rename(columns={"group": "severity"})
    m[["subject", "severity"]].to_csv(out_csv, index=False)
    return out_csv


def _collect_trial_preds(pred_csv: Path, cohort_label: str, group_label: int, dataset_id: str, seed: int) -> pd.DataFrame:
    if not pred_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(pred_csv)
    if df.empty:
        return df
    req = {"subject_key", "memory_load", "z"}
    if not req.issubset(df.columns):
        return pd.DataFrame()

    use = df.copy()
    use["subject"] = use["subject_key"].astype(str)
    subj = subject_load_deviation_score(use)
    if subj.empty:
        return subj

    subj["dataset_id"] = dataset_id
    subj["seed"] = int(seed)
    subj["group"] = int(group_label)
    subj["cohort_label"] = cohort_label
    subj["subject_key"] = subj["subject"].astype(str)
    subj["subject_id"] = subj["subject_key"].map(lambda x: _extract_subject_key_tokens(x)[0])
    subj["session"] = subj["subject_key"].map(lambda x: _extract_subject_key_tokens(x)[1])
    return subj


def _plot_deviation_by_group(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    labels = ["control", "case"]
    colors = ["#1b4965", "#ca3c25"]
    for g in [0, 1]:
        vals = pd.to_numeric(df.loc[df["group"] == g, "z_hi_minus_lo"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        x = np.full(vals.shape, g + 1, dtype=float)
        jit = np.linspace(-0.12, 0.12, num=len(vals), dtype=float) if len(vals) > 1 else np.asarray([0.0])
        ax.scatter(x + jit, vals, s=14, alpha=0.5, color=colors[g])
        ax.hlines(float(np.median(vals)), g + 0.8, g + 1.2, color="black", linewidth=2)
    ax.set_xticks([1, 2], labels)
    ax.set_ylabel("Subject load deviation score (z_hi_minus_lo)")
    ax.set_title("Deviation by Group")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_auc(res_df: pd.DataFrame, out_path: Path) -> None:
    if res_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ds = res_df["dataset_id"].astype(str).tolist()
    auc = pd.to_numeric(res_df["auc"], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(res_df["auc_ci95_lo"], errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(res_df["auc_ci95_hi"], errors="coerce").to_numpy(dtype=float)
    x = np.arange(len(ds), dtype=float)
    ax.bar(x, auc, color="#22577a", alpha=0.8)
    yerr = np.vstack([auc - lo, hi - auc])
    yerr = np.where(np.isfinite(yerr), yerr, 0.0)
    ax.errorbar(x, auc, yerr=yerr, fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(x, ds)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title("Case vs Control Separation")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_recovery(df: pd.DataFrame, out_path: Path) -> bool:
    work = df.copy()
    work = work[work["session"].astype(str).str.len() > 0].copy()
    if work.empty:
        return False
    if work["session"].nunique() < 2:
        return False

    agg = work.groupby(["group", "session"], as_index=False).agg(mean_dev=("z_hi_minus_lo", "mean"), n=("subject_key", "nunique"))
    if agg.empty:
        return False

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for g, lab, col in [(0, "control", "#1b4965"), (1, "case", "#ca3c25")]:
        sub = agg[agg["group"] == g].sort_values("session")
        if sub.empty:
            continue
        ax.plot(sub["session"].astype(str), sub["mean_dev"].astype(float), marker="o", linewidth=2.0, color=col, label=lab)
    ax.set_xlabel("Session")
    ax.set_ylabel("Mean load deviation")
    ax.set_title("Recovery trajectory by group")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _analyze_dataset_group_results(
    dataset_id: str,
    module_root: Path,
    group_df: pd.DataFrame,
    out_dir: Path,
    n_perm: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    seeds = sorted([int(p.name.split("_", 1)[1]) for p in module_root.glob("seed_*") if p.is_dir() and p.name.split("_", 1)[1].isdigit()])
    dev_rows = []

    for s in seeds:
        rep = module_root / f"seed_{s}" / "reports" / "normative" / f"seed_{s}"
        hcsv = rep / "healthy_trial_predictions.csv"
        ccsv = rep / "clinical_trial_predictions.csv"
        d0 = _collect_trial_preds(hcsv, "healthy", 0, dataset_id, s)
        d1 = _collect_trial_preds(ccsv, "clinical", 1, dataset_id, s)
        if not d0.empty:
            dev_rows.append(d0)
        if not d1.empty:
            dev_rows.append(d1)

    if not dev_rows:
        return pd.DataFrame(), pd.DataFrame(), {"status": "SKIP", "reason": "no trial prediction outputs available"}

    dev = pd.concat(dev_rows, axis=0, ignore_index=True)
    ginfo = group_df[["subject_id", "group", "age", "sex"]].drop_duplicates(subset=["subject_id"], keep="first")
    dev = dev.merge(ginfo, on="subject_id", how="left", suffixes=("", "_sev"))
    dev["group"] = pd.to_numeric(dev["group"], errors="coerce").fillna(pd.to_numeric(dev.get("group_sev"), errors="coerce"))
    if "group_sev" in dev.columns:
        dev = dev.drop(columns=["group_sev"]) 

    # Aggregate across seeds per subject for final inference.
    subj = (
        dev.groupby(["dataset_id", "subject_key", "subject_id", "session", "cohort_label"], as_index=False)
        .agg(
            z_mean=("z_mean", "mean"),
            z_hi_minus_lo=("z_hi_minus_lo", "mean"),
            n_trials=("n_trials", "mean"),
            group=("group", "first"),
            age=("age", "first"),
            sex=("sex", "first"),
        )
    )
    subj = subj[np.isfinite(pd.to_numeric(subj["group"], errors="coerce"))].copy()
    subj["group"] = pd.to_numeric(subj["group"], errors="coerce")

    if subj.empty or subj["group"].nunique() < 2:
        return dev, pd.DataFrame(), {"status": "SKIP", "reason": "insufficient group-labeled subjects after merge"}

    auc, auc_ci = _bootstrap_auc(
        y_true=subj["group"].to_numpy(dtype=int),
        y_score=subj["z_hi_minus_lo"].to_numpy(dtype=float),
        n_boot=2000,
        seed=seed,
    )

    beta_group, n_fit = _linear_effect_with_covariates(subj, "z_hi_minus_lo")
    p_group = _perm_p_group_effect(subj, "z_hi_minus_lo", n_perm=n_perm, seed=seed + 17)

    res = pd.DataFrame(
        [
            {
                "dataset_id": dataset_id,
                "n_subjects_total": int(subj["subject_key"].nunique()),
                "n_controls": int((subj["group"] == 0).sum()),
                "n_cases": int((subj["group"] == 1).sum()),
                "auc": auc,
                "auc_ci95_lo": auc_ci[0],
                "auc_ci95_hi": auc_ci[1],
                "group_beta": beta_group,
                "group_perm_p": p_group,
                "group_perm_q": float("nan"),
                "status": "PASS",
                "reason": "",
            }
        ]
    )

    return dev, res, {"status": "PASS", "reason": ""}


def _stage_extract_and_model(ctx: RunContext) -> Dict[str, Any]:
    stage = "clinical_extract_model"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    map_path = ctx.clinical_pack_dir / "mapping_status.json"
    if not map_path.exists():
        _write_json(summary_path, {"status": "SKIP", "reason": "mapping_status.json missing"})
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            started=started,
            rc=0,
            log_path=log_path,
            summary_path=summary_path,
            command="clinical extract/model",
            error="mapping_status missing",
        )

    mapping_status = json.loads(map_path.read_text(encoding="utf-8"))
    rows = mapping_status.get("rows", [])
    pass_datasets = [r["dataset_id"] for r in rows if r.get("status") == "PASS"]
    skip_rows = [r for r in rows if r.get("status") != "PASS"]

    if not pass_datasets:
        _write_json(summary_path, {"status": "SKIP", "reason": "no datasets passed mapping", "rows": rows})
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            started=started,
            rc=0,
            log_path=log_path,
            summary_path=summary_path,
            command="clinical extract/model",
            error="no datasets passed mapping",
        )

    if not ctx.clinical_severity_csv.exists():
        _write_json(summary_path, {"status": "FAIL", "reason": "clinical_severity.csv missing"})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            started=started,
            rc=1,
            log_path=log_path,
            summary_path=summary_path,
            command="clinical extract/model",
            error="clinical_severity.csv missing",
        )

    sev = pd.read_csv(ctx.clinical_severity_csv)
    if "dataset_id" not in sev.columns or "subject_id" not in sev.columns or "group" not in sev.columns:
        err = "clinical_severity.csv missing required columns dataset_id/subject_id/group"
        _write_json(summary_path, {"status": "FAIL", "reason": err})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            started=started,
            rc=1,
            log_path=log_path,
            summary_path=summary_path,
            command="clinical extract/model",
            error=err,
        )

    env = dict(ctx.runtime_env)
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["NUMEXPR_NUM_THREADS"] = "2"
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(REPO_ROOT / "src")

    dataset_outcomes: List[Dict[str, Any]] = []
    all_results: List[pd.DataFrame] = []
    all_dev: List[pd.DataFrame] = []

    for ds in pass_datasets:
        if time.time() > ctx.deadline_ts - 1200:
            hint = ctx.audit_dir / "RESUME_HINT.md"
            _write_text(
                hint,
                "\n".join(
                    [
                        "# RESUME_HINT",
                        "",
                        "Wall-clock guard prevented launching new long stages.",
                        f"Resume command: `python scripts/clinical_overnight10h_runner.py --out_root {ctx.out_root} --resume --datasets {','.join(ctx.datasets)} --wall_hours {ctx.wall_hours}`",
                    ]
                )
                + "\n",
            )
            dataset_outcomes.append({"dataset_id": ds, "status": "SKIP", "reason": "wall-clock budget guard"})
            continue

        ds_root = ctx.clinical_bids_root / ds
        ds_deriv = ctx.clinical_pack_dir / "derivatives" / ds
        ds_model = ctx.clinical_pack_dir / "modeling" / ds
        ds_model.mkdir(parents=True, exist_ok=True)

        sdf = _load_subject_group_for_dataset(sev, ds)
        controls = sorted(set(sdf.loc[sdf["group"] == 0, "subject_id"].astype(str).tolist()))
        cases = sorted(set(sdf.loc[sdf["group"] == 1, "subject_id"].astype(str).tolist()))
        if not controls or not cases:
            reason = f"insufficient controls/cases from clinical_severity.csv (controls={len(controls)}, cases={len(cases)})"
            stop = ctx.clinical_pack_dir / f"STOP_REASON_{ds}.md"
            _write_stop_reason_dataset(stop, ds, reason, {"controls": controls[:20], "cases": cases[:20]})
            dataset_outcomes.append({"dataset_id": ds, "status": "SKIP", "reason": reason, "stop_reason": str(stop)})
            continue

        # 1) preprocess
        pre_cmd = [
            sys.executable,
            "01_preprocess_CPU.py",
            "--bids_root",
            str(ds_root),
            "--deriv_root",
            str(ds_deriv),
            "--config",
            str(ctx.config),
            "--workers",
            str(max(1, min(8, ctx.cpu_workers // 2))),
            "--mne_n_jobs",
            "1",
        ]
        rc = _run_cmd(pre_cmd, cwd=REPO_ROOT, log_path=log_path, env=env)
        if rc != 0:
            reason = f"preprocess failed rc={rc}"
            stop = ctx.clinical_pack_dir / f"STOP_REASON_{ds}.md"
            _write_stop_reason_dataset(stop, ds, reason, {"cmd": pre_cmd, "log_tail": _tail(log_path)})
            dataset_outcomes.append({"dataset_id": ds, "status": "SKIP", "reason": reason, "stop_reason": str(stop)})
            continue

        # 2) extract controls as healthy
        ext_ctrl = [
            sys.executable,
            "02_extract_features_CPU.py",
            "--bids_root",
            str(ds_root),
            "--deriv_root",
            str(ds_deriv),
            "--features_root",
            str(ctx.features_root_clinical),
            "--config",
            str(ctx.config),
            "--cohort",
            "healthy",
            "--dataset_id",
            ds,
            "--lawc_event_map",
            str(ctx.clinical_event_map),
            "--workers",
            str(max(1, min(8, ctx.cpu_workers // 2))),
            "--subjects",
            *controls,
        ]
        rc = _run_cmd(ext_ctrl, cwd=REPO_ROOT, log_path=log_path, env=env)
        if rc != 0:
            reason = f"control extraction failed rc={rc}"
            stop = ctx.clinical_pack_dir / f"STOP_REASON_{ds}.md"
            _write_stop_reason_dataset(stop, ds, reason, {"cmd": ext_ctrl, "log_tail": _tail(log_path)})
            dataset_outcomes.append({"dataset_id": ds, "status": "SKIP", "reason": reason, "stop_reason": str(stop)})
            continue

        # 3) extract cases as clinical
        ext_case = [
            sys.executable,
            "02_extract_features_CPU.py",
            "--bids_root",
            str(ds_root),
            "--deriv_root",
            str(ds_deriv),
            "--features_root",
            str(ctx.features_root_clinical),
            "--config",
            str(ctx.config),
            "--cohort",
            "clinical",
            "--dataset_id",
            ds,
            "--lawc_event_map",
            str(ctx.clinical_event_map),
            "--workers",
            str(max(1, min(8, ctx.cpu_workers // 2))),
            "--subjects",
            *cases,
        ]
        rc = _run_cmd(ext_case, cwd=REPO_ROOT, log_path=log_path, env=env)
        if rc != 0:
            reason = f"case extraction failed rc={rc}"
            stop = ctx.clinical_pack_dir / f"STOP_REASON_{ds}.md"
            _write_stop_reason_dataset(stop, ds, reason, {"cmd": ext_case, "log_tail": _tail(log_path)})
            dataset_outcomes.append({"dataset_id": ds, "status": "SKIP", "reason": reason, "stop_reason": str(stop)})
            continue

        module_root = ds_model / "module04_manyseed"
        module_root.mkdir(parents=True, exist_ok=True)

        severity_m04 = ds_model / "severity_for_module04.csv"
        _prepare_module04_severity_csv(ctx.features_root_clinical, ds, sdf, severity_m04)

        seeds = list(ctx.model_seeds)
        running: Dict[int, subprocess.Popen] = {}
        finished: List[int] = []
        failed: List[int] = []
        queue = list(seeds)

        grace = max(600.0, min(1800.0, 0.04 * ctx.wall_hours * 3600.0))

        while queue or running:
            now = time.time()
            time_left = ctx.deadline_ts - now

            while queue and len(running) < ctx.gpu_parallel_procs and time_left > grace:
                s = queue.pop(0)
                if ctx.resume and _seed_done(module_root, s):
                    finished.append(s)
                    continue
                p = _run_seed_process(
                    dataset_id=ds,
                    seed=s,
                    features_root=ctx.features_root_clinical,
                    module_root=module_root,
                    config=ctx.config,
                    severity_csv=severity_m04,
                    env=env,
                )
                running[s] = p
                time.sleep(0.5)
                time_left = ctx.deadline_ts - time.time()

            if not running:
                if queue and time_left <= grace:
                    break
                if not queue:
                    break

            done = []
            for s, p in list(running.items()):
                rc = p.poll()
                if rc is None:
                    continue
                done.append(s)
                if rc == 0 and _seed_done(module_root, s):
                    finished.append(s)
                else:
                    failed.append(s)

            for s in done:
                running.pop(s, None)

            if running:
                time.sleep(1.5)

        finished = sorted(set(finished))

        if not finished:
            reason = f"module04 produced no completed seeds for {ds}; failed={failed}"
            stop = ctx.clinical_pack_dir / f"STOP_REASON_{ds}.md"
            _write_stop_reason_dataset(stop, ds, reason, {"failed_seeds": failed})
            dataset_outcomes.append({"dataset_id": ds, "status": "SKIP", "reason": reason, "stop_reason": str(stop)})
            continue

        # aggregate seeds
        agg_cmd = [
            sys.executable,
            "aggregate_results.py",
            "--out_root",
            str(module_root),
            "--seeds",
            ",".join(str(s) for s in finished),
        ]
        rc = _run_cmd(agg_cmd, cwd=REPO_ROOT, log_path=log_path, env=env)
        if rc == 0 and (module_root / "aggregate_results.json").exists():
            shutil.copy2(module_root / "aggregate_results.json", module_root / "aggregate_results_manyseed.json")

        dev, res, meta = _analyze_dataset_group_results(
            dataset_id=ds,
            module_root=module_root,
            group_df=sdf,
            out_dir=ds_model,
            n_perm=ctx.rt_n_perm,
            seed=1234,
        )
        if meta.get("status") != "PASS":
            reason = str(meta.get("reason", "analysis failed"))
            stop = ctx.clinical_pack_dir / f"STOP_REASON_{ds}.md"
            _write_stop_reason_dataset(stop, ds, reason, {"analysis_meta": meta})
            dataset_outcomes.append({"dataset_id": ds, "status": "SKIP", "reason": reason, "stop_reason": str(stop)})
            continue

        dev.to_csv(ds_model / "deviation_scores.csv", index=False)
        res.to_csv(ds_model / "clinical_group_results.csv", index=False)

        dataset_outcomes.append(
            {
                "dataset_id": ds,
                "status": "PASS",
                "reason": "",
                "completed_seeds": finished,
                "n_completed_seeds": len(finished),
                "n_failed_seeds": len(failed),
                "module_root": str(module_root),
                "deviation_scores": str(ds_model / "deviation_scores.csv"),
                "group_results": str(ds_model / "clinical_group_results.csv"),
            }
        )
        all_dev.append(dev)
        all_results.append(res)

    # Consolidate all passed datasets into required outputs.
    if all_dev:
        all_dev_df = pd.concat(all_dev, axis=0, ignore_index=True)
        all_dev_df.to_csv(ctx.clinical_pack_dir / "deviation_scores.csv", index=False)
    else:
        all_dev_df = pd.DataFrame()
        pd.DataFrame().to_csv(ctx.clinical_pack_dir / "deviation_scores.csv", index=False)

    if all_results:
        all_res_df = pd.concat(all_results, axis=0, ignore_index=True)
        # BH-FDR across dataset group-effect tests.
        pvals = pd.to_numeric(all_res_df["group_perm_p"], errors="coerce").to_numpy(dtype=float)
        qvals = bh_fdr([float(x) if np.isfinite(x) else 1.0 for x in pvals.tolist()]) if len(pvals) else []
        all_res_df["group_perm_q"] = qvals
        all_res_df.to_csv(ctx.clinical_pack_dir / "clinical_group_results.csv", index=False)
    else:
        all_res_df = pd.DataFrame()
        pd.DataFrame().to_csv(ctx.clinical_pack_dir / "clinical_group_results.csv", index=False)

    # Figures
    fig_dev = ctx.clinical_pack_dir / "FIG_deviation_by_group.png"
    fig_auc = ctx.clinical_pack_dir / "FIG_auc.png"
    fig_rec = ctx.clinical_pack_dir / "FIG_recovery_trajectory.png"

    if not all_dev_df.empty:
        _plot_deviation_by_group(all_dev_df, fig_dev)
        wrote_recovery = _plot_recovery(all_dev_df, fig_rec)
    else:
        wrote_recovery = False
    if not all_res_df.empty:
        _plot_auc(all_res_df, fig_auc)

    stage_status = "PASS" if any(r.get("status") == "PASS" for r in dataset_outcomes) else "SKIP"

    _write_json(
        summary_path,
        {
            "status": stage_status,
            "dataset_outcomes": dataset_outcomes,
            "n_pass": int(sum(1 for r in dataset_outcomes if r.get("status") == "PASS")),
            "n_skip": int(sum(1 for r in dataset_outcomes if r.get("status") == "SKIP")),
            "outputs": {
                "deviation_scores": str(ctx.clinical_pack_dir / "deviation_scores.csv"),
                "clinical_group_results": str(ctx.clinical_pack_dir / "clinical_group_results.csv"),
                "fig_deviation_by_group": str(fig_dev) if fig_dev.exists() else "",
                "fig_auc": str(fig_auc) if fig_auc.exists() else "",
                "fig_recovery_trajectory": str(fig_rec) if wrote_recovery and fig_rec.exists() else "",
            },
        },
    )

    return _record_stage(
        ctx,
        stage=stage,
        status=stage_status,
        started=started,
        rc=0,
        log_path=log_path,
        summary_path=summary_path,
        command="clinical extract + module04 + stats",
        outputs=[
            ctx.clinical_pack_dir / "deviation_scores.csv",
            ctx.clinical_pack_dir / "clinical_group_results.csv",
            fig_dev,
            fig_auc,
            fig_rec,
        ],
    )


def _stage_zip_bundle(ctx: RunContext) -> Dict[str, Any]:
    stage = "zip_bundle"
    started = time.time()
    log_path = ctx.audit_dir / f"{stage}.log"
    summary_path = ctx.audit_dir / f"{stage}_summary.json"

    zpath = ctx.outzip_dir / "clinical_overnight10h_bundle.zip"
    include = [ctx.audit_dir, ctx.clinical_pack_dir]

    added: List[str] = []
    err = ""
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
            if ctx.clinical_event_map.exists():
                rel = Path("AUDIT") / "clinical_event_map_autogen.yaml"
                zf.write(ctx.clinical_event_map, rel.as_posix())
                added.append(rel.as_posix())
        status = "PASS"
        rc = 0
    except Exception as exc:
        status = "FAIL"
        rc = 1
        err = str(exc)

    _write_json(summary_path, {"status": status, "zip": str(zpath), "n_files": len(added), "error": err})

    return _record_stage(
        ctx,
        stage=stage,
        status=status,
        started=started,
        rc=rc,
        log_path=log_path,
        summary_path=summary_path,
        command="zip",
        outputs=[zpath] if zpath.exists() else [],
        error=err,
    )


def _build_report(ctx: RunContext, run_status: str, run_error: str) -> Path:
    report = ctx.audit_dir / "CLINICAL_OVERNIGHT10H_REPORT.md"

    stage_rows = [
        f"| {r['stage']} | {r['status']} | {r['returncode']} | {r['elapsed_sec']:.1f} | {Path(r['log']).name} | {Path(r['summary']).name} |"
        for r in ctx.stage_records
    ]

    hash_json = ctx.audit_dir / "clinical_dataset_hashes.json"
    hash_rows: List[str] = ["| <none> | <none> | <none> | <none> |"]
    if hash_json.exists():
        try:
            payload = json.loads(hash_json.read_text(encoding="utf-8"))
            hash_rows = []
            for d in payload.get("datasets", []):
                hash_rows.append(
                    f"| {d.get('dataset_id')} | {d.get('checked_out_commit')} | {d.get('n_event_files')} | {d.get('n_eeg_files', '<na>')} |"
                )
        except Exception:
            hash_rows = ["| parse_error | parse_error | parse_error | parse_error |"]

    map_json = ctx.clinical_pack_dir / "mapping_status.json"
    map_rows = ["| <none> | <none> | <none> |"]
    if map_json.exists():
        try:
            payload = json.loads(map_json.read_text(encoding="utf-8"))
            map_rows = []
            for r in payload.get("rows", []):
                mapping = r.get("mapping", {})
                map_rows.append(
                    f"| {r.get('dataset_id')} | {r.get('status')} | {r.get('reason', '')} | `{json.dumps(mapping, sort_keys=True)}` |"
                )
        except Exception:
            map_rows = ["| parse_error | parse_error | parse_error | parse_error |"]

    res_csv = ctx.clinical_pack_dir / "clinical_group_results.csv"
    res_rows = ["| <none> | <none> | <none> | <none> | <none> | <none> |"]
    if res_csv.exists():
        try:
            df = pd.read_csv(res_csv)
            if not df.empty:
                res_rows = []
                for _, r in df.iterrows():
                    res_rows.append(
                        f"| {r.get('dataset_id')} | {r.get('n_controls')} | {r.get('n_cases')} | {r.get('auc')} [{r.get('auc_ci95_lo')}, {r.get('auc_ci95_hi')}] | {r.get('group_beta')} | {r.get('group_perm_p')} / {r.get('group_perm_q')} |"
                    )
        except Exception:
            res_rows = ["| parse_error | parse_error | parse_error | parse_error | parse_error | parse_error |"]

    decode_summary = ctx.clinical_pack_dir / "mapping_decode" / "mapping_decode_summary.json"
    candidate_table = ctx.clinical_pack_dir / "mapping_decode" / "CANDIDATE_TABLE.csv"

    lines = [
        "# CLINICAL OVERNIGHT 10H REPORT",
        "",
        f"- Output root: `{ctx.out_root}`",
        f"- Run status: `{run_status}`",
        f"- Resume: `{ctx.resume}`",
        f"- Datasets requested: `{','.join(ctx.datasets)}`",
        f"- Model seeds: `{','.join(str(x) for x in ctx.model_seeds)}`",
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
            "## Clinical dataset hashes",
            "| Dataset | Commit | Event files | EEG files |",
            "|---|---|---:|---:|",
            *hash_rows,
            "",
            "## Mapping outcomes",
            "| Dataset | Status | Reason | Mapping |",
            "|---|---|---|---|",
            *map_rows,
            "",
            "## Mapping Decode Artifacts",
            f"- Mapping decode summary: `{decode_summary}`",
            f"- Candidate table: `{candidate_table}`",
            "",
            "## Group endpoint results",
            "| Dataset | N controls | N cases | AUC [CI95] | Group beta | Perm p / q |",
            "|---|---:|---:|---|---:|---|",
            *res_rows,
            "",
            "## Outputs",
            f"- Deviation scores: `{ctx.clinical_pack_dir / 'deviation_scores.csv'}`",
            f"- Group results: `{ctx.clinical_pack_dir / 'clinical_group_results.csv'}`",
            f"- Figure deviation: `{ctx.clinical_pack_dir / 'FIG_deviation_by_group.png'}`",
            f"- Figure AUC: `{ctx.clinical_pack_dir / 'FIG_auc.png'}`",
            f"- Figure recovery: `{ctx.clinical_pack_dir / 'FIG_recovery_trajectory.png'}`",
            f"- Event map: `{ctx.clinical_event_map}`",
            f"- Severity CSV: `{ctx.clinical_severity_csv}`",
            f"- Bundle: `{ctx.outzip_dir / 'clinical_overnight10h_bundle.zip'}`",
        ]
    )

    _write_text(report, "\n".join(lines) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, default=Path("/filesystemHcog/openneuro"))
    ap.add_argument("--out_root", type=Path, default=None)
    ap.add_argument("--clinical_bids_root", type=Path, default=Path("/filesystemHcog/clinical_bids"))
    ap.add_argument("--clinical_severity_csv", type=Path, default=Path("/filesystemHcog/clinical_bids/clinical_severity.csv"))
    ap.add_argument("--features_root_healthy", type=Path, default=Path("/filesystemHcog/features_cache_FIX2_20260222_061927"))
    ap.add_argument("--features_root_clinical", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--lawc_event_map", type=Path, default=Path("configs/lawc_event_map.yaml"))
    ap.add_argument("--clinical_event_map", type=Path, default=Path("configs/clinical_event_map_autogen.yaml"))
    ap.add_argument("--datasets", type=str, default="ds003523,ds005114")
    ap.add_argument("--model_seeds", type=str, default="0-199")
    ap.add_argument("--wall_hours", type=float, default=10.0)
    ap.add_argument("--rt_n_perm", type=int, default=20000)
    ap.add_argument("--gpu_parallel_procs", type=int, default=12)
    ap.add_argument("--cpu_workers", type=int, default=32)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def _apply_targeted_fix(ctx: RunContext, rec: Dict[str, Any]) -> bool:
    log_txt = _tail(Path(rec["log"]), n=300)
    changed = False

    if re.search(r"VLEN strings|embedded NULL|HDF5|Unicode", log_txt, flags=re.IGNORECASE):
        ctx.runtime_env["HDF5_USE_FILE_LOCKING"] = "FALSE"
        changed = True

    if re.search(r"IndexError|out of bounds|cannot do a non-empty take", log_txt, flags=re.IGNORECASE):
        ctx.cpu_workers = max(4, ctx.cpu_workers // 2)
        changed = True

    if re.search(r"No module named|ImportError|NameError", log_txt, flags=re.IGNORECASE):
        # Re-run compile pass next attempt with current tree; no code changes here.
        changed = True

    if changed:
        fix_note = ctx.audit_dir / "AUTO_FIX_LOG.md"
        with fix_note.open("a", encoding="utf-8") as f:
            f.write(f"## {_iso_now()} {rec['stage']}\n")
            f.write("Applied targeted runtime fix heuristics before retry.\n\n")
    return changed


def main() -> int:
    args = parse_args()

    if args.out_root is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_root = Path("/filesystemHcog/runs") / f"{ts}_CLINICAL_OVERNIGHT10H"
    else:
        out_root = args.out_root

    datasets = _split_csv(args.datasets)
    if not datasets:
        print("ERROR: --datasets resolved to empty set", file=sys.stderr, flush=True)
        return 1
    model_seeds = _parse_seeds(args.model_seeds)
    if not model_seeds:
        print("ERROR: --model_seeds resolved to empty set", file=sys.stderr, flush=True)
        return 1

    run_ts = out_root.name.split("_", 1)[0]
    features_root_clinical = args.features_root_clinical
    if features_root_clinical is None:
        features_root_clinical = Path(f"/filesystemHcog/features_cache_CLINICAL_OVERNIGHT10H_{run_ts}")

    if out_root.exists() and not args.resume:
        print(f"ERROR: out_root exists and --resume not set: {out_root}", file=sys.stderr, flush=True)
        return 1

    runtime_env = os.environ.copy()
    runtime_env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{runtime_env.get('PYTHONPATH', '')}" if runtime_env.get("PYTHONPATH") else str(REPO_ROOT / "src")

    ctx = RunContext(
        out_root=out_root,
        audit_dir=out_root / "AUDIT",
        outzip_dir=out_root / "OUTZIP",
        clinical_pack_dir=out_root / "ClinicalPack",
        data_root=args.data_root,
        clinical_bids_root=args.clinical_bids_root,
        clinical_severity_csv=args.clinical_severity_csv,
        features_root_healthy=args.features_root_healthy,
        features_root_clinical=features_root_clinical,
        config=args.config,
        lawc_event_map=args.lawc_event_map,
        clinical_event_map=args.clinical_event_map,
        datasets=datasets,
        model_seeds=model_seeds,
        wall_hours=float(args.wall_hours),
        rt_n_perm=int(args.rt_n_perm),
        gpu_parallel_procs=int(max(1, args.gpu_parallel_procs)),
        cpu_workers=int(max(1, args.cpu_workers)),
        resume=bool(args.resume),
        start_ts=time.time(),
        deadline_ts=time.time() + float(args.wall_hours) * 3600.0,
        stage_records=[],
        monitor_proc=None,
        runtime_env=runtime_env,
    )

    ctx.out_root.mkdir(parents=True, exist_ok=True)
    ctx.audit_dir.mkdir(parents=True, exist_ok=True)
    ctx.outzip_dir.mkdir(parents=True, exist_ok=True)
    ctx.clinical_pack_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        _stage_preflight,
        _ensure_compile_gate,
        _stage_ensure_clinical_data,
        _stage_build_severity,
        _stage_decode_mapping,
        _stage_extract_and_model,
        _stage_zip_bundle,
    ]

    run_status = "PASS"
    run_error = ""

    try:
        for fn in stages:
            attempt = 1
            max_attempts = 3
            while True:
                rec = fn(ctx)
                if rec["status"] != "FAIL":
                    break
                if attempt >= max_attempts:
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

    report_path = _build_report(ctx, run_status, run_error)

    _write_json(
        ctx.audit_dir / "run_status.json",
        {
            "status": run_status,
            "error": run_error,
            "out_root": str(ctx.out_root),
            "report": str(report_path),
            "stage_records": ctx.stage_records,
        },
    )

    print(f"OUT_ROOT={ctx.out_root}", flush=True)
    print(f"FINAL_REPORT={report_path}", flush=True)
    print(f"BUNDLE_ZIP={ctx.outzip_dir / 'clinical_overnight10h_bundle.zip'}", flush=True)

    return 0 if run_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
