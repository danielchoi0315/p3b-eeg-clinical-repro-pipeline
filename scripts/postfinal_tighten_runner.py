#!/usr/bin/env python3
"""Post-final tightening runner.

Scope:
- Start from canonical NN_FINAL_MASTER_V1 outputs
- Re-run ds004584 clinical stage only (coverage tightening)
- Attempt one-shot ds004752 scalp EEG repair
- Export results-only tarball for archival/download

Fail-closed behavior:
- No silent skips: STOP_REASON markdown for any skipped dataset stage
- Core Law-C/mechanism are not recomputed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_auc_score, roc_curve


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_RUN_DEFAULT = Path("/filesystemHcog/runs/20260223_185511_NN_FINAL_MASTER_V1")
CANONICAL_ZIP_NAME = "NN_FINAL_MASTER_V1_SUBMISSION_PACKET.zip"

STAGES: List[str] = [
    "preflight",
    "ds004584_inspect",
    "ds004584_retrieve_missing",
    "ds004584_rerun_pdrest_endpoints",
    "ds004752_one_shot_repair",
    "bundle_zip",
    "tarball_export",
]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_sanitize(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _json_sanitize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_json_sanitize(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_json_sanitize(x) for x in v.tolist()]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        fv = float(v)
        return fv if np.isfinite(fv) else None
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    return v


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_sanitize(payload), indent=2, allow_nan=False), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tail(path: Path, n: int = 200) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max(1, int(n)) :])


def _parse_bool(raw: Any) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _stable_int_from_text(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    allow_fail: bool = False,
    env: Optional[Dict[str, str]] = None,
    timeout_sec: Optional[int] = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"[{_iso_now()}] CMD: {' '.join(cmd)}\n")
        lf.flush()
        p = None
        try:
            p = subprocess.Popen(
                list(cmd),
                cwd=str(cwd),
                env=env,
                stdout=lf,
                stderr=lf,
                text=True,
                preexec_fn=os.setsid,
            )
            rc = int(p.wait(timeout=timeout_sec))
        except subprocess.TimeoutExpired:
            rc = 124
            lf.write(f"[{_iso_now()}] ERROR: timeout after {timeout_sec}s\n")
            if p is not None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    pass
        except Exception as exc:
            rc = 1
            lf.write(f"[{_iso_now()}] ERROR: {exc}\n")
    if rc != 0 and not allow_fail:
        raise RuntimeError(f"command failed rc={rc}: {' '.join(cmd)}")
    return rc


def _write_stop_reason(path: Path, title: str, reason: str, diagnostics: Optional[Dict[str, Any]] = None) -> None:
    lines = [f"# STOP_REASON {title}", "", "## Why", reason]
    if diagnostics is not None:
        lines += ["", "## Diagnostics", "```json", json.dumps(_json_sanitize(diagnostics), indent=2), "```"]
    _write_text(path, "\n".join(lines) + "\n")


def _bh_qvals(pvals: Sequence[float]) -> List[float]:
    arr = np.asarray([float(x) if np.isfinite(float(x)) else 1.0 for x in pvals], dtype=float)
    n = int(arr.size)
    if n == 0:
        return []
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    q = arr * float(n) / ranks
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q_sorted, 0.0, 1.0)
    return [float(x) for x in out]


def _bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray, *, n_boot: int, seed: int) -> Tuple[float, Tuple[float, float]]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if y.size < 4 or np.unique(y).size < 2:
        return float("nan"), (float("nan"), float("nan"))
    obs = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    n = int(y.size)
    for _ in range(int(max(1, n_boot))):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if np.unique(yb).size < 2:
            continue
        vals.append(float(roc_auc_score(yb, s[idx])))
    if not vals:
        return obs, (float("nan"), float("nan"))
    return obs, (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


def _perm_p_auc(y_true: np.ndarray, y_score: np.ndarray, *, n_perm: int, seed: int) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if y.size < 4 or np.unique(y).size < 2:
        return float("nan")
    obs = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(int(max(1, n_perm))):
        yp = rng.permutation(y)
        if np.unique(yp).size < 2:
            continue
        null.append(float(roc_auc_score(yp, s)))
    if not null:
        return float("nan")
    nv = np.asarray(null, dtype=float)
    p = (1.0 + np.sum(np.abs(nv - 0.5) >= abs(obs - 0.5))) / (1.0 + nv.size)
    return float(p)


def _fit_logit_beta(X: np.ndarray, y: np.ndarray) -> float:
    if X.size == 0 or y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    try:
        clf = LogisticRegression(solver="liblinear", penalty="l2", C=1.0, max_iter=300)
        clf.fit(X, y)
        coef = np.asarray(clf.coef_, dtype=float)
        return float(coef[0, 0]) if coef.ndim == 2 and coef.shape[1] >= 1 else float("nan")
    except Exception:
        return float("nan")


def _bootstrap_beta_ci(X: np.ndarray, y: np.ndarray, *, n_boot: int, seed: int) -> Tuple[float, float]:
    if y.size <= 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    n = int(y.size)
    for _ in range(int(max(1, n_boot))):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if np.unique(yb).size < 2:
            continue
        b = _fit_logit_beta(X[idx], yb)
        if np.isfinite(b):
            vals.append(float(b))
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _perm_p_beta(X: np.ndarray, y: np.ndarray, obs_beta: float, *, n_perm: int, seed: int) -> float:
    if X.size == 0 or y.size == 0 or np.unique(y).size < 2 or not np.isfinite(obs_beta):
        return float("nan")
    rng = np.random.default_rng(seed)
    null: List[float] = []
    for _ in range(int(max(1, n_perm))):
        yp = rng.permutation(y)
        if np.unique(yp).size < 2:
            continue
        b = _fit_logit_beta(X, yp)
        if np.isfinite(b):
            null.append(float(b))
    if not null:
        return float("nan")
    nv = np.asarray(null, dtype=float)
    p = (1.0 + np.sum(np.abs(nv) >= abs(obs_beta))) / (1.0 + nv.size)
    return float(p)


def _read_raw_any(path: Path):
    suf = path.suffix.lower()
    if suf == ".vhdr":
        return mne.io.read_raw_brainvision(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".set":
        return mne.io.read_raw_eeglab(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".edf":
        return mne.io.read_raw_edf(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".bdf":
        return mne.io.read_raw_bdf(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".fif":
        return mne.io.read_raw_fif(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".gdf":
        return mne.io.read_raw_gdf(path.as_posix(), preload=True, verbose="ERROR")
    raise RuntimeError(f"unsupported EEG suffix: {suf}")


def _find_best_rest_file(ds_root: Path, subject_id: str) -> Tuple[Optional[Path], str]:
    pats = [
        f"sub-{subject_id}/**/*_eeg.vhdr",
        f"sub-{subject_id}/**/*_eeg.set",
        f"sub-{subject_id}/**/*_eeg.edf",
        f"sub-{subject_id}/**/*_eeg.bdf",
        f"sub-{subject_id}/**/*_eeg.fif",
        f"sub-{subject_id}/**/*_eeg.gdf",
    ]
    cand: List[Path] = []
    for pat in pats:
        cand.extend(ds_root.glob(pat))
    cand = sorted(set(cand))
    if not cand:
        return None, "no_eeg_file"
    for p in cand:
        if p.suffix.lower() == ".vhdr":
            if p.exists() and p.with_suffix(".eeg").exists() and p.with_suffix(".vmrk").exists():
                return p, "ok"
        elif p.suffix.lower() == ".set":
            if p.exists() and p.with_suffix(".fdt").exists():
                return p, "ok"
        else:
            if p.exists():
                return p, "ok"
    return None, "missing_sidecar_or_payload"


def _compute_rest_feature_row(raw, *, max_seconds: float = 120.0) -> Dict[str, Any]:
    raw.pick_types(eeg=True, eog=False, misc=False, stim=False)
    if len(raw.ch_names) == 0:
        raise RuntimeError("no EEG channels")
    sf = float(raw.info["sfreq"])
    if sf > 256.0:
        raw.resample(256, verbose="ERROR")
        sf = float(raw.info["sfreq"])
    dur_s = float(raw.n_times / max(sf, 1e-9))
    if dur_s > max_seconds:
        raw.crop(tmin=0.0, tmax=float(max_seconds), include_tmax=False)
        dur_s = float(raw.n_times / max(sf, 1e-9))
    data = raw.get_data()
    if data.size == 0:
        raise RuntimeError("empty data")

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
    slope = float("nan")
    if int(msl.sum()) >= 6:
        x = np.log10(f[msl])
        y = np.log10(pxx[msl])
        slope = float(np.polyfit(x, y, deg=1)[0])

    return {
        "theta_alpha_ratio": float(theta / max(alpha, 1e-12)) if np.isfinite(theta) and np.isfinite(alpha) else float("nan"),
        "rel_alpha": float(alpha / max(total, 1e-12)) if np.isfinite(alpha) and np.isfinite(total) else float("nan"),
        "spectral_slope": float(slope),
        "n_channels": int(data.shape[0]),
        "n_samples": int(data.shape[1]),
        "sfreq": float(sf),
        "duration_s": float(dur_s),
    }


def _infer_ds004584_groups(part: pd.DataFrame) -> Tuple[Optional[pd.Series], str, Dict[str, Any]]:
    cols = list(part.columns)
    candidates = [c for c in cols if re.search(r"group|diag|dx|type|status", c, flags=re.IGNORECASE)]
    if "GROUP" in cols and "GROUP" not in candidates:
        candidates.insert(0, "GROUP")
    diag: Dict[str, Any] = {"candidate_columns": candidates}
    for c in candidates:
        s = part[c].astype(str).str.strip()
        low = s.str.lower()
        mapped = pd.Series(["UNK"] * len(s), index=s.index)
        mapped[low.str.contains(r"\bpd\b|parkinson", na=False, regex=True)] = "PD"
        mapped[low.str.contains(r"control|healthy|\bcn\b|\bhc\b|\bctl\b", na=False, regex=True)] = "CN"
        if c.upper() == "TYPE":
            num = pd.to_numeric(s, errors="coerce")
            if np.isfinite(num).sum() >= 20 and pd.Series(num).dropna().nunique() == 2:
                lo = float(np.nanmin(num))
                hi = float(np.nanmax(num))
                mapped[np.isfinite(num) & (num == hi)] = "PD"
                mapped[np.isfinite(num) & (num == lo)] = "CN"
        n_pd = int((mapped == "PD").sum())
        n_cn = int((mapped == "CN").sum())
        if n_pd >= 20 and n_cn >= 20:
            diag["selected_column"] = c
            diag["n_pd"] = n_pd
            diag["n_cn"] = n_cn
            return mapped, c, diag
    return None, "", diag


def _encode_sex(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower().str.strip()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[s.str.startswith("m")] = 1.0
    out[s.str.startswith("f")] = 0.0
    return out


def _build_design_matrix(df: pd.DataFrame, feature_col: str) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    x0 = pd.to_numeric(df[feature_col], errors="coerce")
    y = pd.to_numeric(df["label"], errors="coerce")
    work = pd.DataFrame({"feature": x0, "label": y})
    used_cov: List[str] = []

    if "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce")
        if int(np.isfinite(age).sum()) >= 10 and pd.Series(age[np.isfinite(age)]).nunique() > 1:
            age = age.fillna(float(np.nanmedian(age)))
            work["age"] = age
            used_cov.append("age")

    if "sex" in df.columns:
        sx = _encode_sex(df["sex"])
        if int(np.isfinite(sx).sum()) >= 10 and pd.Series(sx[np.isfinite(sx)]).nunique() > 1:
            sx = sx.fillna(float(np.nanmedian(sx)))
            work["sex_num"] = sx
            used_cov.append("sex_num")

    fit = work[["feature"] + used_cov + ["label"]].dropna().copy()
    if fit.empty:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=int), used_cov, 0
    X = fit[["feature"] + used_cov].to_numpy(dtype=float)
    yv = fit["label"].to_numpy(dtype=int)
    return X, yv, used_cov, int(len(fit))


def _compute_deviation(df: pd.DataFrame, controls_mask: pd.Series) -> pd.DataFrame:
    out = df.copy()
    base = ["theta_alpha_ratio", "rel_alpha", "spectral_slope"]
    for f in base:
        x = pd.to_numeric(out[f], errors="coerce")
        ref = x[controls_mask]
        mu = float(np.nanmean(ref)) if np.isfinite(ref).any() else 0.0
        sd = float(np.nanstd(ref)) if np.isfinite(ref).any() else 1.0
        if not np.isfinite(sd) or sd <= 1e-6:
            sd = 1.0
        out[f"dev_z_{f}"] = (x - mu) / sd
    out["composite_deviation"] = np.nanmean(
        np.column_stack(
            [
                pd.to_numeric(out.get("dev_z_theta_alpha_ratio"), errors="coerce"),
                pd.to_numeric(out.get("dev_z_rel_alpha"), errors="coerce"),
                pd.to_numeric(out.get("dev_z_spectral_slope"), errors="coerce"),
            ]
        ),
        axis=1,
    )
    return out


def _plot_roc_calibration(y_true: np.ndarray, scores: np.ndarray, title: str, roc_path: Path, cal_path: Path) -> None:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    m = np.isfinite(s)
    y = y[m]
    s = s[m]
    if y.size < 4 or np.unique(y).size < 2:
        return

    fpr, tpr, _ = roc_curve(y, s)
    auc_v = float(sk_auc(fpr, tpr))
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#0072B2", lw=2, label=f"AUC={auc_v:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(roc_path, dpi=140)
    plt.close()

    pmin, pmax = float(np.nanmin(s)), float(np.nanmax(s))
    sn = (s - pmin) / max(pmax - pmin, 1e-9)
    prob_true, prob_pred = calibration_curve(y, sn, n_bins=8, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, "o-", color="#009E73", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title(title + " Calibration")
    plt.tight_layout()
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cal_path, dpi=140)
    plt.close()


def _file_inventory(root: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not root.exists():
        return out
    for ext in ["vhdr", "set", "edf", "bdf", "fif", "gdf", "tsv", "json"]:
        out[f"eeg_{ext}"] = len(list(root.rglob(f"*_eeg.{ext}")))
        out[f"ieeg_{ext}"] = len(list(root.rglob(f"*_ieeg.{ext}")))
    out["events_tsv"] = len(list(root.rglob("*_events.tsv")))
    return out


def _find_matching_signal(events_tsv: Path, modality: str) -> Optional[Path]:
    stem = events_tsv.name.replace("_events.tsv", f"_{modality}")
    parent = events_tsv.parent
    for ext in [".vhdr", ".set", ".edf", ".bdf", ".fif", ".gdf"]:
        p = parent / f"{stem}{ext}"
        if p.exists():
            if ext == ".set" and not p.with_suffix(".fdt").exists():
                continue
            return p
    prefix = events_tsv.name.replace("_events.tsv", "")
    for ext in [".vhdr", ".set", ".edf", ".bdf", ".fif", ".gdf"]:
        for p in sorted(parent.glob(prefix + f"_{modality}{ext}")):
            if ext == ".set" and not p.with_suffix(".fdt").exists():
                continue
            return p
    return None


def _extract_event_amp(raw, onsets: np.ndarray, *, p3_win: Tuple[float, float], pick_ch_name: Optional[str]) -> float:
    sf = float(raw.info["sfreq"])
    if sf <= 0:
        return float("nan")
    if pick_ch_name and pick_ch_name in raw.ch_names:
        picks = [raw.ch_names.index(pick_ch_name)]
    else:
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if picks is None or len(picks) == 0:
            picks = np.arange(len(raw.ch_names))
    events: List[List[int]] = []
    for o in np.asarray(onsets, dtype=float):
        s = int(round(o * sf))
        if s + int(round(0.8 * sf)) >= int(raw.n_times):
            continue
        if s - int(round(0.2 * sf)) < 0:
            continue
        events.append([s, 0, 1])
    if len(events) < 3:
        return float("nan")
    ev = np.asarray(events, dtype=int)
    epochs = mne.Epochs(
        raw,
        ev,
        event_id={"stim": 1},
        tmin=-0.2,
        tmax=0.8,
        baseline=(-0.2, 0.0),
        preload=True,
        picks=picks,
        reject_by_annotation=False,
        detrend=None,
        verbose="ERROR",
    )
    if len(epochs) < 3:
        return float("nan")
    data = epochs.get_data()
    tm = epochs.times
    m = (tm >= float(p3_win[0])) & (tm <= float(p3_win[1]))
    if int(m.sum()) < 2:
        return float("nan")
    return float(np.nanmean(data[:, :, m]))


@dataclass
class Ctx:
    out_root: Path
    audit: Path
    outzip: Path
    tarballs: Path
    pack_pdrest: Path
    pack_bio: Path
    data_root: Path
    canonical_root: Path
    resume: bool
    stage_records: List[Dict[str, Any]]
    stage_status: Dict[str, str]
    stage_extra: Dict[str, Dict[str, Any]]
    partial_reasons: List[str]


def _record_stage(
    ctx: Ctx,
    *,
    stage: str,
    status: str,
    rc: int,
    started: float,
    log_path: Path,
    summary_path: Path,
    command: str,
    outputs: Optional[List[Path]] = None,
    error: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rec = {
        "stage": stage,
        "status": status,
        "returncode": int(rc),
        "elapsed_sec": float(time.time() - started),
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "ended_at": _iso_now(),
        "command": command,
        "log": str(log_path),
        "summary": str(summary_path),
        "error": str(error),
        "outputs": [str(p) for p in (outputs or [])],
    }
    if extra:
        rec.update(_json_sanitize(extra))
    _write_json(summary_path, rec)
    _write_text(ctx.audit / f"{stage}.status", status + "\n")
    ctx.stage_records.append(rec)
    ctx.stage_status[stage] = status
    ctx.stage_extra[stage] = extra or {}
    return rec


def _load_resumed_stage(ctx: Ctx, stage: str) -> Optional[Dict[str, Any]]:
    st = ctx.audit / f"{stage}.status"
    sm = ctx.audit / f"{stage}_summary.json"
    if not st.exists() or not sm.exists():
        return None
    try:
        rec = _read_json(sm)
    except Exception:
        return None
    rec["status"] = st.read_text(encoding="utf-8").strip()
    ctx.stage_records.append(rec)
    ctx.stage_status[stage] = rec.get("status", "")
    ctx.stage_extra[stage] = rec
    return rec


def _build_repo_fingerprint(repo_root: Path, out_path: Path) -> None:
    try:
        p = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if p.returncode == 0:
            _write_json(out_path, {"git_head": p.stdout.strip(), "repo_root": str(repo_root), "mode": "git"})
            return
    except Exception:
        pass
    files = []
    for p in sorted(repo_root.rglob("*.py")):
        if ".git" in p.parts:
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            while True:
                b = f.read(1024 * 1024)
                if not b:
                    break
                h.update(b)
        files.append({"path": p.relative_to(repo_root).as_posix(), "sha256": h.hexdigest(), "size": int(p.stat().st_size)})
    _write_json(out_path, {"git_head": None, "repo_root": str(repo_root), "mode": "sha256_manifest", "files": files})


def _write_dataset_hashes(data_root: Path, out_path: Path) -> None:
    rows = []
    for ds in ["ds004584", "ds007020", "ds004752", "ds007262"]:
        ds_root = data_root / ds
        head = None
        if ds_root.exists():
            try:
                p = subprocess.run(["git", "-C", str(ds_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
                if p.returncode == 0:
                    head = p.stdout.strip()
            except Exception:
                head = None
        rows.append({"dataset_id": ds, "commit": head, "dataset_root": str(ds_root)})
    _write_json(out_path, {"datasets": rows})


def _stage_preflight(ctx: Ctx) -> Dict[str, Any]:
    stage = "preflight"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"

    _build_repo_fingerprint(REPO_ROOT, ctx.audit / "repo_fingerprint.json")
    _write_dataset_hashes(ctx.data_root, ctx.audit / "dataset_hashes.json")

    try:
        pf = subprocess.run([sys.executable, "-m", "pip", "freeze"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
        _write_text(ctx.audit / "pip_freeze.txt", pf.stdout)
    except Exception:
        _write_text(ctx.audit / "pip_freeze.txt", "")

    canonical_zip = ctx.canonical_root / "OUTZIP" / CANONICAL_ZIP_NAME
    can_ok = canonical_zip.exists()
    data_ok = (ctx.data_root / "ds004584").exists() and (ctx.data_root / "ds004752").exists()
    _write_json(
        ctx.audit / "preflight_env.json",
        {
            "repo_root": str(REPO_ROOT),
            "out_root": str(ctx.out_root),
            "canonical_root": str(ctx.canonical_root),
            "canonical_zip": str(canonical_zip),
            "data_root": str(ctx.data_root),
            "canonical_exists": bool(can_ok),
            "data_exists": bool(data_ok),
        },
    )
    if not can_ok or not data_ok:
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="preflight",
            outputs=[ctx.audit / "preflight_env.json"],
            error="missing canonical zip or required datasets",
        )

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="preflight",
        outputs=[
            ctx.audit / "preflight_env.json",
            ctx.audit / "repo_fingerprint.json",
            ctx.audit / "dataset_hashes.json",
            ctx.audit / "pip_freeze.txt",
        ],
    )


def _stage_ds004584_inspect(ctx: Ctx) -> Dict[str, Any]:
    stage = "ds004584_inspect"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"

    ds_root = ctx.data_root / "ds004584"
    part_path = ds_root / "participants.tsv"
    can_exc = ctx.canonical_root / "PACK_CLINICAL_PDREST_MASTER" / "EXCLUSIONS.csv"
    can_feat = ctx.canonical_root / "PACK_CLINICAL_PDREST_MASTER" / "pdrest_features.csv"

    if not part_path.exists():
        stop = ctx.audit / "STOP_REASON_ds004584_inspect.md"
        _write_stop_reason(stop, stage, "participants.tsv missing for ds004584", diagnostics={"path": str(part_path)})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="inspect ds004584",
            outputs=[stop],
            error="participants.tsv missing",
        )

    part = pd.read_csv(part_path, sep="\t")
    expected_n = int(len(part))
    vhdr_files = sorted(ds_root.rglob("*_eeg.vhdr"))
    vhdr_count = int(len(vhdr_files))
    set_files = sorted(ds_root.rglob("*_eeg.set"))
    set_ready = [p for p in set_files if p.exists() and p.with_suffix(".fdt").exists()]
    set_ready_count = int(len(set_ready))

    n_used_before = 0
    if can_feat.exists():
        try:
            n_used_before = int(pd.read_csv(can_feat)["subject_id"].nunique())
        except Exception:
            n_used_before = 0

    before_reason_counts: Dict[str, int] = {}
    before_missing_subjects: List[str] = []
    if can_exc.exists():
        try:
            exc_df = pd.read_csv(can_exc)
            if "reason" in exc_df.columns:
                before_reason_counts = {str(k): int(v) for k, v in exc_df["reason"].value_counts(dropna=False).to_dict().items()}
            missing_mask = exc_df.get("reason", pd.Series("", index=exc_df.index)).astype(str).str.contains(
                r"missing|no_eeg|payload|sidecar|absent", case=False, na=False, regex=True
            )
            before_missing_subjects = exc_df.loc[missing_mask, "subject_id"].astype(str).tolist()[:200] if "subject_id" in exc_df.columns else []
        except Exception:
            pass

    need_retrieve = bool(vhdr_count < expected_n or len(before_missing_subjects) > 0)

    md_lines = [
        "# ds004584 Coverage Before Tightening",
        "",
        f"- Expected participants (participants.tsv rows): `{expected_n}`",
        f"- Available vhdr headers: `{vhdr_count}`",
        f"- Available EEGLAB set+fdt pairs: `{set_ready_count}`",
        f"- Canonical N_used before: `{n_used_before}`",
        f"- Retrieval needed: `{need_retrieve}`",
        "",
        "## Canonical exclusion taxonomy (before)",
    ]
    if before_reason_counts:
        for k, v in before_reason_counts.items():
            md_lines.append(f"- `{k}`: {v}")
    else:
        md_lines.append("- <none/unknown>")
    _write_text(ctx.audit / "ds004584_coverage_before.md", "\n".join(md_lines) + "\n")

    extra = {
        "expected_n": expected_n,
        "vhdr_count_before": vhdr_count,
        "set_ready_count_before": set_ready_count,
        "n_used_before": n_used_before,
        "before_reason_counts": before_reason_counts,
        "retrieve_needed": need_retrieve,
    }
    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="inspect ds004584",
        outputs=[ctx.audit / "ds004584_coverage_before.md"],
        extra=extra,
    )


def _stage_ds004584_retrieve_missing(ctx: Ctx) -> Dict[str, Any]:
    stage = "ds004584_retrieve_missing"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"
    ds_root = ctx.data_root / "ds004584"

    inspect = ctx.stage_extra.get("ds004584_inspect", {})
    need_retrieve = bool(inspect.get("retrieve_needed", True))

    rc_datalad = None
    rc_annex = None
    stop_path = ctx.audit / "STOP_REASON_ds004584_retrieval.md"

    if need_retrieve:
        if shutil.which("datalad") is not None:
            rc_datalad = _run_cmd(
                ["bash", "-lc", "shopt -s globstar nullglob; datalad get -r -J 8 sub-*/**/eeg/*"],
                cwd=ds_root,
                log_path=log_path,
                allow_fail=True,
                timeout_sec=2400,
            )
        if (ds_root / ".git").exists() and shutil.which("git") is not None:
            rc_annex = _run_cmd(
                ["bash", "-lc", "shopt -s globstar nullglob; git annex get -J 8 -- sub-*/**/eeg/*"],
                cwd=ds_root,
                log_path=log_path,
                allow_fail=True,
                timeout_sec=2400,
            )

    part_path = ds_root / "participants.tsv"
    expected_n = int(len(pd.read_csv(part_path, sep="\t"))) if part_path.exists() else 0
    vhdr_count = int(len(list(ds_root.rglob("*_eeg.vhdr"))))
    set_ready_count = int(len([p for p in ds_root.rglob("*_eeg.set") if p.exists() and p.with_suffix(".fdt").exists()]))

    if need_retrieve and (rc_datalad not in {0, None}) and (rc_annex not in {0, None}):
        _write_stop_reason(
            stop_path,
            stage,
            "dataset retrieval commands failed (datalad and git-annex).",
            diagnostics={
                "rc_datalad": rc_datalad,
                "rc_annex": rc_annex,
                "vhdr_count_after": vhdr_count,
                "set_ready_count_after": set_ready_count,
                "expected_n": expected_n,
                "log_tail": _tail(log_path, 200),
            },
        )

    md_lines = [
        "# ds004584 Coverage After Retrieval Attempt",
        "",
        f"- Expected participants: `{expected_n}`",
        f"- Available vhdr headers: `{vhdr_count}`",
        f"- Available EEGLAB set+fdt pairs: `{set_ready_count}`",
        f"- Retrieval attempted: `{need_retrieve}`",
        f"- datalad rc: `{rc_datalad}`",
        f"- git-annex rc: `{rc_annex}`",
    ]
    if stop_path.exists():
        md_lines += ["", f"- STOP reason: `{stop_path}`"]
    _write_text(ctx.audit / "ds004584_coverage_after_retrieval.md", "\n".join(md_lines) + "\n")

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="retrieve missing ds004584 files",
        outputs=[ctx.audit / "ds004584_coverage_after_retrieval.md"] + ([stop_path] if stop_path.exists() else []),
        extra={
            "expected_n": expected_n,
            "vhdr_count_after": vhdr_count,
            "set_ready_count_after": set_ready_count,
            "retrieval_attempted": bool(need_retrieve),
            "rc_datalad": rc_datalad,
            "rc_annex": rc_annex,
        },
    )


def _stage_ds004584_rerun_pdrest_endpoints(ctx: Ctx) -> Dict[str, Any]:
    stage = "ds004584_rerun_pdrest_endpoints"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"
    out_dir = ctx.pack_pdrest
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_root = ctx.data_root / "ds004584"
    part_path = ds_root / "participants.tsv"
    stop = out_dir / "STOP_REASON_ds004584.md"
    if not part_path.exists():
        _write_stop_reason(stop, stage, "participants.tsv missing", diagnostics={"path": str(part_path)})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="rerun ds004584 clinical endpoints",
            outputs=[stop],
            error="participants.tsv missing",
        )

    part = pd.read_csv(part_path, sep="\t")
    part["subject_id"] = part["participant_id"].astype(str).str.replace("sub-", "", regex=False)

    labels, label_col, diag = _infer_ds004584_groups(part)
    if labels is None:
        _write_stop_reason(stop, stage, "ambiguous PD/control labels", diagnostics=diag)
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="rerun ds004584 clinical endpoints",
            outputs=[stop],
            error="label inference failed",
        )

    part["group"] = labels
    part = part[part["group"].isin(["PD", "CN"])].copy()
    part["age"] = pd.to_numeric(part.get("AGE", part.get("Age", np.nan)), errors="coerce")
    part["sex"] = part.get("GENDER", part.get("Sex", "")).astype(str)

    feat_rows: List[Dict[str, Any]] = []
    ex_rows: List[Dict[str, Any]] = []
    for sid in part["subject_id"].astype(str).tolist():
        fpath, why = _find_best_rest_file(ds_root, sid)
        if fpath is None:
            ex_rows.append({"subject_id": sid, "reason": why, "rest_file": ""})
            continue
        try:
            raw = _read_raw_any(fpath)
            feat = _compute_rest_feature_row(raw, max_seconds=120.0)
            feat["subject_id"] = sid
            feat["rest_file"] = str(fpath)
            feat_rows.append(feat)
        except Exception as exc:
            ex_rows.append({"subject_id": sid, "reason": f"read_error:{exc}", "rest_file": str(fpath)})
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(f"[{_iso_now()}] subject={sid} read_error {exc}\n")

    feat_df = pd.DataFrame(feat_rows)
    ex_df = pd.DataFrame(ex_rows) if ex_rows else pd.DataFrame(columns=["subject_id", "reason", "rest_file"])
    ex_path = out_dir / "EXCLUSIONS.csv"
    ex_df.to_csv(ex_path, index=False)

    if feat_df.empty:
        _write_stop_reason(stop, stage, "no readable resting EEG features", diagnostics={"n_exclusions": int(len(ex_df))})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="rerun ds004584 clinical endpoints",
            outputs=[stop, ex_path],
            error="no features",
        )

    feat_df = feat_df.merge(part[["subject_id", "group", "age", "sex"]], on="subject_id", how="left")
    dev_df = _compute_deviation(feat_df, controls_mask=(feat_df["group"] == "CN"))
    feat_out = out_dir / "features.csv"
    dev_df.to_csv(feat_out, index=False)

    n_perm = 20000
    n_boot = 2000
    rows: List[Dict[str, Any]] = []
    features = ["dev_z_theta_alpha_ratio", "dev_z_rel_alpha", "dev_z_spectral_slope", "composite_deviation"]
    for feat in features:
        sub = dev_df[["subject_id", "group", "age", "sex", feat]].copy()
        sub[feat] = pd.to_numeric(sub[feat], errors="coerce")
        sub = sub[np.isfinite(sub[feat])].copy()
        if sub.empty:
            continue
        sub["label"] = (sub["group"].astype(str) == "PD").astype(int)
        y = sub["label"].to_numpy(dtype=int)
        score = sub[feat].to_numpy(dtype=float)
        if np.unique(y).size < 2:
            continue

        auc_obs, auc_ci = _bootstrap_auc(y, score, n_boot=n_boot, seed=10000 + _stable_int_from_text(feat) % 100000)
        p_auc = _perm_p_auc(y, score, n_perm=n_perm, seed=11000 + _stable_int_from_text(feat) % 100000)

        X, yy, covs, nfit = _build_design_matrix(sub.rename(columns={feat: "feature"}), "feature")
        beta = _fit_logit_beta(X, yy)
        b_lo, b_hi = _bootstrap_beta_ci(X, yy, n_boot=n_boot, seed=12000 + _stable_int_from_text(feat) % 100000)
        p_beta = _perm_p_beta(X, yy, beta, n_perm=n_perm, seed=13000 + _stable_int_from_text(feat) % 100000)

        rows.append(
            {
                "dataset_id": "ds004584",
                "endpoint": "AUC_PD_vs_CN",
                "feature": feat,
                "type": "auc",
                "n": int(len(sub)),
                "estimate": float(auc_obs),
                "ci95_lo": float(auc_ci[0]),
                "ci95_hi": float(auc_ci[1]),
                "perm_p": float(p_auc),
                "n_perm_done": int(n_perm),
                "n_boot_done": int(n_boot),
            }
        )
        rows.append(
            {
                "dataset_id": "ds004584",
                "endpoint": "LogitBeta_PD_vs_CN",
                "feature": feat,
                "type": "logit_beta",
                "n": int(nfit),
                "estimate": float(beta),
                "ci95_lo": float(b_lo),
                "ci95_hi": float(b_hi),
                "perm_p": float(p_beta),
                "covariates": ";".join(covs),
                "n_perm_done": int(n_perm),
                "n_boot_done": int(n_boot),
            }
        )

    end_df = pd.DataFrame(rows)
    if end_df.empty:
        _write_stop_reason(stop, stage, "no endpoints after feature extraction", diagnostics={"n_subjects_features": int(len(dev_df))})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="rerun ds004584 clinical endpoints",
            outputs=[stop, ex_path, feat_out],
            error="no endpoints",
        )

    end_df["q_within_ds004584"] = _bh_qvals(pd.to_numeric(end_df["perm_p"], errors="coerce").fillna(1.0).to_numpy(dtype=float).tolist())
    end_path = out_dir / "endpoints.csv"
    end_df.to_csv(end_path, index=False)

    primary_feat = "composite_deviation"
    subp = dev_df[dev_df["group"].isin(["PD", "CN"])].copy()
    yy = (subp["group"].astype(str) == "PD").astype(int).to_numpy(dtype=int)
    ss = pd.to_numeric(subp[primary_feat], errors="coerce").to_numpy(dtype=float)
    roc_path = out_dir / "FIG_pdrest_primary_auc_roc.png"
    cal_path = out_dir / "FIG_pdrest_calibration.png"
    _plot_roc_calibration(yy, ss, "ds004584 PD vs CN", roc_path, cal_path)

    # Coverage note: before/after usage and exclusion taxonomy.
    can_feat = ctx.canonical_root / "PACK_CLINICAL_PDREST_MASTER" / "pdrest_features.csv"
    can_exc = ctx.canonical_root / "PACK_CLINICAL_PDREST_MASTER" / "EXCLUSIONS.csv"
    n_before = int(pd.read_csv(can_feat)["subject_id"].nunique()) if can_feat.exists() else 0
    n_after = int(dev_df["subject_id"].nunique())
    before_tax = {}
    if can_exc.exists():
        tmp = pd.read_csv(can_exc)
        if "reason" in tmp.columns:
            before_tax = {str(k): int(v) for k, v in tmp["reason"].value_counts(dropna=False).to_dict().items()}
    after_tax = {str(k): int(v) for k, v in ex_df["reason"].value_counts(dropna=False).to_dict().items()} if not ex_df.empty else {}

    inspect = ctx.stage_extra.get("ds004584_inspect", {})
    retrieve = ctx.stage_extra.get("ds004584_retrieve_missing", {})
    note_lines = [
        "# COVERAGE NOTE (ds004584)",
        "",
        f"- Expected cohort size (participants.tsv): `{inspect.get('expected_n')}`",
        f"- vhdr count before: `{inspect.get('vhdr_count_before')}`",
        f"- vhdr count after retrieval attempt: `{retrieve.get('vhdr_count_after')}`",
        f"- set+fdt pairs before: `{inspect.get('set_ready_count_before')}`",
        f"- set+fdt pairs after retrieval attempt: `{retrieve.get('set_ready_count_after')}`",
        f"- N_used before (canonical): `{n_before}`",
        f"- N_used after (tighten): `{n_after}`",
        f"- Delta N_used: `{n_after - n_before}`",
        "",
        "## Exclusion taxonomy before",
    ]
    if before_tax:
        for k, v in before_tax.items():
            note_lines.append(f"- `{k}`: {v}")
    else:
        note_lines.append("- <none/unknown>")
    note_lines += ["", "## Exclusion taxonomy after"]
    if after_tax:
        for k, v in after_tax.items():
            note_lines.append(f"- `{k}`: {v}")
    else:
        note_lines.append("- <none>")
    _write_text(ctx.audit / "COVERAGE_NOTE.md", "\n".join(note_lines) + "\n")

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="rerun ds004584 clinical endpoints",
        outputs=[feat_out, end_path, ex_path, roc_path, cal_path, ctx.audit / "COVERAGE_NOTE.md"],
        extra={
            "n_subjects_used": int(n_after),
            "n_before": int(n_before),
            "n_after": int(n_after),
            "delta_n": int(n_after - n_before),
            "n_perm_done": int(n_perm),
            "n_boot_done": int(n_boot),
            "group_column": label_col,
        },
    )


def _stage_ds004752_one_shot_repair(ctx: Ctx) -> Dict[str, Any]:
    stage = "ds004752_one_shot_repair"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"
    out_dir = ctx.pack_bio
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_root = ctx.data_root / "ds004752"
    stop_path = out_dir / "STOP_REASON_ds004752.md"
    can_stop = ctx.canonical_root / "PACK_BIO_CROSSMODALITY" / "BIO_D_cross_modality" / "STOP_REASON_ds004752.md"

    inventory = _file_inventory(ds_root)
    _write_json(out_dir / "file_inventory_ds004752.json", inventory)

    can_stop_excerpt = ""
    if can_stop.exists():
        can_stop_excerpt = _tail(can_stop, 80)
        _write_text(out_dir / "canonical_stop_excerpt.txt", can_stop_excerpt + "\n")

    try:
        events = sorted(ds_root.rglob("*_task-verbalWM*_events.tsv"))
        rows: List[Dict[str, Any]] = []
        fails: List[Dict[str, Any]] = []

        for ep in events:
            if "/eeg/" not in ep.as_posix():
                continue
            sid_m = re.search(r"sub-([A-Za-z0-9]+)", ep.as_posix())
            sid = sid_m.group(1) if sid_m else "unknown"
            try:
                df = pd.read_csv(ep, sep="\t")
                if "onset" not in df.columns:
                    fails.append({"subject_id": sid, "events_file": str(ep), "reason": "missing onset"})
                    continue
                load_col = None
                for c in ["SetSize", "setsize", "load", "difficulty", "condition"]:
                    if c in df.columns:
                        load_col = c
                        break
                if load_col is None:
                    fails.append({"subject_id": sid, "events_file": str(ep), "reason": "missing load column"})
                    continue
                onset = pd.to_numeric(df["onset"], errors="coerce")
                load = pd.to_numeric(df[load_col], errors="coerce")
                ok = np.isfinite(onset.to_numpy(dtype=float)) & np.isfinite(load.to_numpy(dtype=float))
                if int(np.sum(ok)) < 8:
                    fails.append({"subject_id": sid, "events_file": str(ep), "reason": "insufficient valid events"})
                    continue
                onset_v = onset.to_numpy(dtype=float)[ok]
                load_v = load.to_numpy(dtype=float)[ok]
                med = float(np.nanmedian(load_v))
                low = onset_v[load_v < med]
                high = onset_v[load_v >= med]
                if len(low) < 3 or len(high) < 3:
                    fails.append({"subject_id": sid, "events_file": str(ep), "reason": "not enough low/high events"})
                    continue

                sig = _find_matching_signal(ep, "eeg")
                if sig is None:
                    fails.append({"subject_id": sid, "events_file": str(ep), "reason": "missing EEG signal"})
                    continue
                raw = _read_raw_any(sig)
                raw.pick_types(eeg=True, eog=False, misc=False, stim=False)
                if len(raw.ch_names) == 0:
                    fails.append({"subject_id": sid, "events_file": str(ep), "reason": "no EEG channels"})
                    continue

                prefer = ["Pz", "PZ", "P3", "P4", "CPz", "CPZ", "POz", "POZ", "Oz", "OZ", "Cz", "CZ"]
                pick_name = None
                for ch in prefer:
                    if ch in raw.ch_names:
                        pick_name = ch
                        break
                if pick_name is None:
                    parietal_like = [c for c in raw.ch_names if re.search(r"^P|^CP|^PO", str(c), flags=re.IGNORECASE)]
                    pick_name = parietal_like[0] if parietal_like else raw.ch_names[0]

                amp_low = _extract_event_amp(raw, low, p3_win=(0.35, 0.60), pick_ch_name=pick_name)
                amp_high = _extract_event_amp(raw, high, p3_win=(0.35, 0.60), pick_ch_name=pick_name)
                sig_hl = float(amp_high - amp_low) if np.isfinite(amp_high) and np.isfinite(amp_low) else float("nan")
                if not np.isfinite(sig_hl):
                    fails.append({"subject_id": sid, "events_file": str(ep), "reason": "non-finite signature"})
                    continue
                rows.append(
                    {
                        "subject_id": sid,
                        "events_file": str(ep),
                        "signal_file": str(sig),
                        "channel_used": pick_name,
                        "load_column": load_col,
                        "n_low": int(len(low)),
                        "n_high": int(len(high)),
                        "amp_low": float(amp_low),
                        "amp_high": float(amp_high),
                        "signature_high_minus_low": float(sig_hl),
                    }
                )
            except Exception as exc:
                fails.append({"subject_id": sid, "events_file": str(ep), "reason": f"reader_error:{exc}"})
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(f"[{_iso_now()}] ds004752 subject={sid} error={exc}\n")

        sig_df = pd.DataFrame(rows)
        fail_df = pd.DataFrame(fails) if fails else pd.DataFrame(columns=["subject_id", "events_file", "reason"])
        sig_path = out_dir / "crossmodality_eeg_signatures.csv"
        fail_path = out_dir / "crossmodality_eeg_failures.csv"
        sig_df.to_csv(sig_path, index=False)
        fail_df.to_csv(fail_path, index=False)

        if sig_df.empty or int(sig_df["subject_id"].nunique()) < 8:
            _write_stop_reason(
                stop_path,
                stage,
                "scalp-EEG-only extraction did not yield enough valid signatures.",
                diagnostics={
                    "n_rows": int(len(sig_df)),
                    "n_subjects": int(sig_df["subject_id"].nunique()) if not sig_df.empty else 0,
                    "file_inventory": inventory,
                    "canonical_stop_excerpt_tail": can_stop_excerpt,
                    "next_steps": [
                        "Verify events semantics for probe/load in ds004752 sidecars.",
                        "Add reader-specific preprocessing for each EEG format in dataset.",
                        "Relax minimum event thresholds only with explicit scientific justification.",
                    ],
                    "log_tail": _tail(log_path, 200),
                },
            )
            return _record_stage(
                ctx,
                stage=stage,
                status="SKIP",
                rc=0,
                started=started,
                log_path=log_path,
                summary_path=sum_path,
                command="one-shot ds004752 repair",
                outputs=[sig_path, fail_path, stop_path, out_dir / "file_inventory_ds004752.json"],
                error="insufficient valid ds004752 signatures",
                extra={"n_rows": int(len(sig_df)), "n_subjects": int(sig_df["subject_id"].nunique()) if not sig_df.empty else 0},
            )

        # Minimal figure
        fig = out_dir / "FIG_ds004752_eeg_signature_hist.png"
        plt.figure(figsize=(6, 5))
        plt.hist(sig_df["signature_high_minus_low"].to_numpy(dtype=float), bins=12, color="#0072B2", alpha=0.85)
        plt.axvline(float(np.nanmean(sig_df["signature_high_minus_low"])), color="k", ls="--", lw=1)
        plt.xlabel("High-Low ERP mean (0.35-0.60 s)")
        plt.ylabel("Count")
        plt.title("ds004752 scalp EEG one-shot signatures")
        plt.tight_layout()
        plt.savefig(fig, dpi=140)
        plt.close()

        _write_json(
            out_dir / "crossmodality_summary.json",
            {
                "status": "PASS",
                "n_rows": int(len(sig_df)),
                "n_subjects": int(sig_df["subject_id"].nunique()),
                "mean_signature": float(np.nanmean(sig_df["signature_high_minus_low"])),
                "median_signature": float(np.nanmedian(sig_df["signature_high_minus_low"])),
                "note": "Scalp-only exploratory repair; iEEG convergence not required for this one-shot pass.",
            },
        )

        return _record_stage(
            ctx,
            stage=stage,
            status="PASS",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="one-shot ds004752 repair",
            outputs=[sig_path, fail_path, fig, out_dir / "crossmodality_summary.json", out_dir / "file_inventory_ds004752.json"],
            extra={"n_rows": int(len(sig_df)), "n_subjects": int(sig_df["subject_id"].nunique())},
        )
    except Exception as exc:
        _write_stop_reason(
            stop_path,
            stage,
            f"ds004752 one-shot repair failed: {exc}",
            diagnostics={
                "file_inventory": inventory,
                "canonical_stop_excerpt_tail": can_stop_excerpt,
                "log_tail": _tail(log_path, 200),
            },
        )
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="one-shot ds004752 repair",
            outputs=[stop_path, out_dir / "file_inventory_ds004752.json"],
            error=str(exc),
        )


def _stage_bundle_zip(ctx: Ctx) -> Dict[str, Any]:
    stage = "bundle_zip"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"
    out_zip = ctx.outzip / "POSTFINAL_TIGHTEN_PACKET.zip"
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    canonical_zip = ctx.canonical_root / "OUTZIP" / CANONICAL_ZIP_NAME

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for base in [ctx.audit, ctx.pack_pdrest, ctx.pack_bio]:
            if not base.exists():
                continue
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(ctx.out_root)))
        if canonical_zip.exists():
            zf.write(canonical_zip, arcname=f"CANONICAL/{CANONICAL_ZIP_NAME}")

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="bundle tighten zip",
        outputs=[out_zip],
        extra={"zip_size_bytes": int(out_zip.stat().st_size if out_zip.exists() else 0)},
    )


def _stage_tarball_export(ctx: Ctx) -> Dict[str, Any]:
    stage = "tarball_export"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"

    tar_path = ctx.tarballs / "results_only.tar.gz"
    sha_path = ctx.tarballs / "results_only.tar.gz.sha256"
    ctx.tarballs.mkdir(parents=True, exist_ok=True)

    canon_name = ctx.canonical_root.name
    tight_name = ctx.out_root.name

    # Exclude current tarball path to avoid recursive inclusion while writing.
    cmd = [
        "tar",
        "--exclude",
        f"{tight_name}/TARBALLS/results_only.tar.gz",
        "-czf",
        str(tar_path),
        "-C",
        "/filesystemHcog/runs",
        canon_name,
        tight_name,
    ]
    _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, allow_fail=False, timeout_sec=7200)

    h = hashlib.sha256()
    with tar_path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    digest = h.hexdigest()
    _write_text(sha_path, f"{digest}  {tar_path.name}\n")

    scp_cmd = f"scp <user>@<host>:\"{tar_path}\" ."
    _write_text(ctx.audit / "DOWNLOAD_SCP.txt", scp_cmd + "\n")

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="export results-only tarball",
        outputs=[tar_path, sha_path, ctx.audit / "DOWNLOAD_SCP.txt"],
        extra={"tar_sha256": digest, "scp": scp_cmd},
    )


def _final_run_status(ctx: Ctx, run_status: str, run_error: str) -> None:
    payload = {
        "status": run_status,
        "error": run_error,
        "out_root": str(ctx.out_root),
        "stages": ctx.stage_records,
        "partial_reasons": ctx.partial_reasons,
    }
    _write_json(ctx.audit / "run_status.json", payload)


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-final tightening runner")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="/filesystemHcog/openneuro")
    ap.add_argument("--canonical_root", type=str, default=str(CANONICAL_RUN_DEFAULT))
    ap.add_argument("--resume", type=str, default="false")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    if out_root.exists() and not _parse_bool(args.resume):
        existing = list(out_root.iterdir())
        if existing:
            print(f"ERROR: out_root exists and is non-empty: {out_root}", file=sys.stderr, flush=True)
            return 2

    audit = out_root / "AUDIT"
    outzip = out_root / "OUTZIP"
    tarballs = out_root / "TARBALLS"
    pack_pdrest = out_root / "PACK_CLINICAL_PDREST_TIGHTEN"
    pack_bio = out_root / "PACK_BIO_CROSSMODALITY_TIGHTEN"

    for p in [out_root, audit, outzip, tarballs, pack_pdrest, pack_bio]:
        p.mkdir(parents=True, exist_ok=True)

    ctx = Ctx(
        out_root=out_root,
        audit=audit,
        outzip=outzip,
        tarballs=tarballs,
        pack_pdrest=pack_pdrest,
        pack_bio=pack_bio,
        data_root=Path(args.data_root).resolve(),
        canonical_root=Path(args.canonical_root).resolve(),
        resume=_parse_bool(args.resume),
        stage_records=[],
        stage_status={},
        stage_extra={},
        partial_reasons=[],
    )

    stage_funcs = {
        "preflight": _stage_preflight,
        "ds004584_inspect": _stage_ds004584_inspect,
        "ds004584_retrieve_missing": _stage_ds004584_retrieve_missing,
        "ds004584_rerun_pdrest_endpoints": _stage_ds004584_rerun_pdrest_endpoints,
        "ds004752_one_shot_repair": _stage_ds004752_one_shot_repair,
        "bundle_zip": _stage_bundle_zip,
        "tarball_export": _stage_tarball_export,
    }

    run_status = "PASS"
    run_error = ""
    try:
        for stage in STAGES:
            if ctx.resume and stage != "preflight":
                resumed = _load_resumed_stage(ctx, stage)
                if resumed is not None and str(resumed.get("status", "")) in {"PASS", "SKIP"}:
                    continue
            rec = stage_funcs[stage](ctx)
            if rec.get("status") == "FAIL":
                run_status = "FAIL"
                run_error = str(rec.get("error", "stage failed"))
                break
            if rec.get("status") == "SKIP":
                ctx.partial_reasons.append(f"{stage} skipped: {rec.get('error', '')}")
    except Exception as exc:
        run_status = "FAIL"
        run_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    # Objective gate
    if run_status != "FAIL":
        pd_end = ctx.pack_pdrest / "endpoints.csv"
        coverage_note = ctx.audit / "COVERAGE_NOTE.md"
        tar_ok = (ctx.tarballs / "results_only.tar.gz").exists()
        if not pd_end.exists() or not coverage_note.exists() or not tar_ok:
            run_status = "FAIL"
            run_error = "required outputs missing (ds004584 endpoints/coverage note/tarball)"

    _final_run_status(ctx, run_status, run_error)

    print(f"OUT_ROOT={ctx.out_root}", flush=True)
    print(f"TARBALL={ctx.tarballs / 'results_only.tar.gz'}", flush=True)
    print("SCP=scp <user>@<host>:" + f"\"{ctx.tarballs / 'results_only.tar.gz'}\"" + " .", flush=True)
    return 0 if run_status in {"PASS", "PARTIAL_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

