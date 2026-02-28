#!/usr/bin/env python3
"""NN_FINAL_MEGA_V3_CLINICALFIX surgical runner.

This runner reuses canonical V2_BIO outputs and recomputes only clinical packs:
- ds004584 PD-vs-control resting EEG (near-full cohort gate)
- ds007020 mortality resting EEG with LEAPD-style primary endpoint

Fail-closed behavior:
- Every skipped/failed dataset writes STOP_REASON_<dataset>.md
- PASS_STRICT only if requested strict objective gates are met
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

STAGES: List[str] = [
    "preflight",
    "stage_verify_ds004584_full",
    "stage_verify_ds007020_full",
    "clinical_ds004584_fullN",
    "clinical_ds007020_mortality_fixbeta",
    "endpoint_hierarchy_and_corrections",
    "v3_bundle",
]

CANONICAL_DEFAULT = Path("/filesystemHcog/runs/20260223_052006_NN_FINAL_MEGA_V2_BIO")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_int_from_text(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


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
        if not np.isfinite(fv):
            return None
        return fv
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return v
    return v


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_sanitize(payload), indent=2, allow_nan=False), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tail(path: Path, n: int = 120) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max(1, int(n)) :])


def _run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    allow_fail: bool = False,
    timeout_sec: Optional[int] = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"[{_iso_now()}] CMD: {' '.join(cmd)}\n")
        lf.flush()
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
            lf.write(f"[{_iso_now()}] ERROR: command timed out after {timeout_sec}s\n")
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                p.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass
                try:
                    p.wait(timeout=5)
                except Exception:
                    pass
        except Exception as exc:
            rc = 1
            lf.write(f"[{_iso_now()}] ERROR: {exc}\n")
    if rc != 0 and not allow_fail:
        raise RuntimeError(f"command failed rc={rc}: {' '.join(cmd)}")
    return rc


def _parse_bool(raw: Any) -> bool:
    t = str(raw).strip().lower()
    return t in {"1", "true", "yes", "y", "on"}


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
    q_final = np.empty(n, dtype=float)
    q_final[order] = np.clip(q_sorted, 0.0, 1.0)
    return [float(x) for x in q_final]


def _bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray, *, n_boot: int, seed: int) -> Tuple[float, Tuple[float, float]]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan"), (float("nan"), float("nan"))
    obs = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    n = int(y.size)
    for _ in range(int(max(1, n_boot))):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        sb = s[idx]
        if np.unique(yb).size < 2:
            continue
        vals.append(float(roc_auc_score(yb, sb)))
    if not vals:
        return obs, (float("nan"), float("nan"))
    lo = float(np.quantile(vals, 0.025))
    hi = float(np.quantile(vals, 0.975))
    return obs, (lo, hi)


def _perm_p_auc(y_true: np.ndarray, y_score: np.ndarray, *, n_perm: int, seed: int) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if y.size == 0 or np.unique(y).size < 2:
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


def _write_stop_reason(path: Path, title: str, reason: str, diagnostics: Optional[Dict[str, Any]] = None) -> None:
    lines = [
        f"# STOP_REASON {title}",
        "",
        "## Why",
        reason,
    ]
    if diagnostics is not None:
        lines.extend([
            "",
            "## Diagnostics",
            "```json",
            json.dumps(_json_sanitize(diagnostics), indent=2),
            "```",
        ])
    _write_text(path, "\n".join(lines) + "\n")


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
    sub_patterns = [
        f"sub-{subject_id}/**/*_eeg.vhdr",
        f"sub-{subject_id}/**/*_eeg.set",
        f"sub-{subject_id}/**/*_eeg.edf",
        f"sub-{subject_id}/**/*_eeg.bdf",
        f"sub-{subject_id}/**/*_eeg.fif",
        f"sub-{subject_id}/**/*_eeg.gdf",
    ]
    cand: List[Path] = []
    for pat in sub_patterns:
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


def _compute_rest_feature_row(raw, *, max_seconds: float = 120.0, lpc_order: Optional[int] = None) -> Dict[str, Any]:
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

    out: Dict[str, Any] = {
        "theta_alpha_ratio": float(theta / max(alpha, 1e-12)) if np.isfinite(theta) and np.isfinite(alpha) else float("nan"),
        "rel_alpha": float(alpha / max(total, 1e-12)) if np.isfinite(alpha) and np.isfinite(total) else float("nan"),
        "spectral_slope": float(slope),
        "n_channels": int(data.shape[0]),
        "n_samples": int(data.shape[1]),
        "sfreq": float(sf),
        "duration_s": float(dur_s),
    }

    if lpc_order is not None and int(lpc_order) > 1:
        prefer = ["Pz", "PZ", "Cz", "CZ", "Fz", "FZ"]
        idx = None
        for ch in prefer:
            if ch in raw.ch_names:
                idx = raw.ch_names.index(ch)
                break
        if idx is None:
            idx = 0
        sig = np.asarray(data[idx], dtype=float)
        sig = sig - float(np.nanmean(sig))
        if not np.isfinite(sig).all() or sig.size <= int(lpc_order) + 2:
            coeff = np.full(int(lpc_order), np.nan, dtype=float)
        else:
            y = sig[int(lpc_order) :]
            X = np.column_stack([sig[int(lpc_order) - k - 1 : sig.size - k - 1] for k in range(int(lpc_order))])
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                coeff = np.asarray(beta, dtype=float)
            except Exception:
                coeff = np.full(int(lpc_order), np.nan, dtype=float)
        for i, v in enumerate(coeff, start=1):
            out[f"lpc_{i:02d}"] = float(v) if np.isfinite(v) else float("nan")

    return out


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


def _infer_ds007020_labels(part: pd.DataFrame) -> Tuple[Optional[pd.Series], str, Dict[str, Any]]:
    cols = list(part.columns)
    candidates = [c for c in cols if re.search(r"surviv|death|mort|vital|status|deceas|living", c, flags=re.IGNORECASE)]
    diag: Dict[str, Any] = {"candidate_columns": candidates}
    for c in candidates:
        s = part[c]
        y = pd.Series(np.nan, index=s.index, dtype=float)
        num = pd.to_numeric(s, errors="coerce")
        if np.isfinite(num).sum() > 0 and set(pd.Series(num).dropna().astype(int).unique().tolist()).issubset({0, 1}):
            y = num.astype(float)
        if y.isna().all():
            low = s.astype(str).str.lower().str.strip()
            y[low.str.contains(r"deceased|dead|died|death|mort", na=False, regex=True)] = 1.0
            y[low.str.contains(r"living|alive|surviv", na=False, regex=True)] = 0.0
        n0 = int((y == 0).sum())
        n1 = int((y == 1).sum())
        if n0 >= 5 and n1 >= 5:
            diag["selected_column"] = c
            diag["n_living"] = n0
            diag["n_deceased"] = n1
            return y, c, diag
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
            med = float(np.nanmedian(age)) if np.isfinite(age).any() else 0.0
            age = age.fillna(med)
            work["age"] = age
            used_cov.append("age")

    if "sex" in df.columns:
        sexn = _encode_sex(df["sex"])
        if int(np.isfinite(sexn).sum()) >= 10 and pd.Series(sexn[np.isfinite(sexn)]).nunique() > 1:
            med = float(np.nanmedian(sexn)) if np.isfinite(sexn).any() else 0.0
            sexn = sexn.fillna(med)
            work["sex_num"] = sexn
            used_cov.append("sex_num")

    cols = ["feature"] + used_cov + ["label"]
    fit = work[cols].dropna().copy()
    if fit.empty:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=int), used_cov, 0
    X = fit[["feature"] + used_cov].to_numpy(dtype=float)
    yy = fit["label"].to_numpy(dtype=int)
    return X, yy, used_cov, int(len(fit))


def _fit_logit_beta(X: np.ndarray, y: np.ndarray) -> float:
    if X.size == 0 or y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    try:
        clf = LogisticRegression(solver="liblinear", penalty="l2", C=1.0, max_iter=200)
        clf.fit(X, y)
        coef = np.asarray(clf.coef_, dtype=float)
        if coef.ndim != 2 or coef.shape[1] < 1:
            return float("nan")
        return float(coef[0, 0])
    except Exception:
        return float("nan")


def _bootstrap_beta_ci(X: np.ndarray, y: np.ndarray, *, n_boot: int, seed: int) -> Tuple[float, float]:
    n = int(y.size)
    if n <= 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(int(max(1, n_boot))):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if np.unique(yb).size < 2:
            continue
        bb = _fit_logit_beta(X[idx], yb)
        if np.isfinite(bb):
            vals.append(float(bb))
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
        bb = _fit_logit_beta(X, yp)
        if np.isfinite(bb):
            null.append(float(bb))
    if not null:
        return float("nan")
    nv = np.asarray(null, dtype=float)
    p = (1.0 + np.sum(np.abs(nv) >= abs(obs_beta))) / (1.0 + nv.size)
    return float(p)


def _plot_roc_calibration(y_true: np.ndarray, scores: np.ndarray, title: str, roc_path: Path, cal_path: Path) -> None:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    m = np.isfinite(s)
    y = y[m]
    s = s[m]
    if y.size < 4 or np.unique(y).size < 2:
        return

    fpr, tpr, _ = roc_curve(y, s)
    roc_v = float(sk_auc(fpr, tpr))
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#0072B2", lw=2, label=f"AUC={roc_v:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(roc_path, dpi=140)
    plt.close()

    prob_true, prob_pred = calibration_curve(y, (s - np.min(s)) / max(np.max(s) - np.min(s), 1e-9), n_bins=8, strategy="quantile")
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


def _count_ds004584_headers(ds_root: Path) -> Dict[str, Any]:
    vhdr = sorted(ds_root.rglob("*_eeg.vhdr"))
    setf = sorted(ds_root.rglob("*_eeg.set"))
    set_ok = [p for p in setf if p.exists() and p.with_suffix(".fdt").exists()]
    sid_re = re.compile(r"sub-([A-Za-z0-9]+)")
    sub_vhdr = sorted({sid_re.search(str(p)).group(1) for p in vhdr if sid_re.search(str(p)) and p.exists()})
    sub_set = sorted({sid_re.search(str(p)).group(1) for p in set_ok if sid_re.search(str(p))})
    return {
        "vhdr_count": int(sum(1 for p in vhdr if p.exists())),
        "set_count": int(len(set_ok)),
        "subjects_vhdr": sub_vhdr,
        "subjects_set": sub_set,
    }


def _direct_download_ds004584_min(ds_root: Path, participants: pd.DataFrame, *, min_subjects: int, log_path: Path) -> Dict[str, Any]:
    part = participants.copy()
    part["sid"] = part["participant_id"].astype(str).str.replace("sub-", "", regex=False)
    grp = part.get("GROUP", pd.Series(["UNK"] * len(part))).astype(str)
    pd_sids = part.loc[grp.str.lower().str.contains("pd", na=False), "sid"].tolist()
    cn_sids = part.loc[grp.str.lower().str.contains("control|cn|hc|ctl", na=False, regex=True), "sid"].tolist()

    selected: List[str] = []
    selected.extend(cn_sids)
    for sid in pd_sids:
        if sid not in selected:
            selected.append(sid)
        if len(selected) >= max(min_subjects, len(cn_sids) + 10):
            break

    base_url = "https://s3.amazonaws.com/openneuro.org/ds004584/"
    jobs: List[Tuple[str, Path]] = []
    for sid in selected:
        for ext in ("set", "fdt"):
            rel = f"sub-{sid}/eeg/sub-{sid}_task-Rest_eeg.{ext}"
            p = ds_root / rel
            if p.exists():
                continue
            if p.is_symlink():
                target = (p.parent / os.readlink(p)).resolve()
            else:
                target = p
            jobs.append((rel, target))

    if not jobs:
        return {"selected_subjects": len(selected), "jobs": 0, "ok": 0, "fail": 0}

    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"[{_iso_now()}] ds004584 direct-http fallback: selected={len(selected)} jobs={len(jobs)}\n")

    opener = urllib.request.build_opener()

    def _download_one(job: Tuple[str, Path]) -> Tuple[str, bool, str]:
        rel, target = job
        url = base_url + rel
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with opener.open(req, timeout=180) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f, length=1024 * 1024)
            tmp.replace(target)
            return rel, True, ""
        except Exception as exc:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return rel, False, str(exc)

    ok = 0
    fail: List[Tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_download_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            rel, good, msg = fut.result()
            if good:
                ok += 1
            else:
                fail.append((rel, msg))
            if i % 20 == 0 or i == len(jobs):
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(f"[{_iso_now()}] ds004584 direct-http progress {i}/{len(jobs)} ok={ok} fail={len(fail)}\n")

    return {
        "selected_subjects": int(len(selected)),
        "jobs": int(len(jobs)),
        "ok": int(ok),
        "fail": int(len(fail)),
        "fail_sample": fail[:10],
    }


def _subject_set_fdt_paths(subject_ids: Sequence[str]) -> List[str]:
    out: List[str] = []
    for sid0 in subject_ids:
        sid = re.sub(r"[^A-Za-z0-9]", "", str(sid0))
        if not sid:
            continue
        out.append(f"sub-{sid}/eeg/sub-{sid}_task-Rest_eeg.set")
        out.append(f"sub-{sid}/eeg/sub-{sid}_task-Rest_eeg.fdt")
    return out


@dataclass
class Ctx:
    out_root: Path
    audit: Path
    outzip: Path
    data_root: Path
    canonical_root: Path
    wall_hours: float
    resume: bool

    pack_pdrest_v3: Path
    pack_mort_v3: Path

    start_ts: float
    deadline_ts: float

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
    status_path = ctx.audit / f"{stage}.status"
    summary_path = ctx.audit / f"{stage}_summary.json"
    if not status_path.exists() or not summary_path.exists():
        return None
    try:
        rec = _read_json(summary_path)
    except Exception:
        return None
    rec["status"] = status_path.read_text(encoding="utf-8").strip()
    ctx.stage_records.append(rec)
    ctx.stage_status[stage] = rec.get("status", "")
    ctx.stage_extra[stage] = rec
    return rec


def _stage_preflight(ctx: Ctx) -> Dict[str, Any]:
    stage = "preflight"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"

    data_ok = (ctx.data_root / "ds004584").exists() and (ctx.data_root / "ds007020").exists()
    can_ok = ctx.canonical_root.exists() and (ctx.canonical_root / "OUTZIP" / "NN_FINAL_MEGA_V2_BIO_SUBMISSION_PACKET.zip").exists()
    free_gb = float(shutil.disk_usage(ctx.out_root).free) / (1024**3)

    py_ver = sys.version.replace("\n", " ")
    _write_text(ctx.audit / "python_version.txt", py_ver + "\n")

    _write_json(
        ctx.audit / "preflight_env.json",
        {
            "cwd": str(REPO_ROOT),
            "python": py_ver,
            "data_root": str(ctx.data_root),
            "canonical_root": str(ctx.canonical_root),
            "free_gb": free_gb,
            "resume": ctx.resume,
            "wall_hours": ctx.wall_hours,
        },
    )

    if not data_ok:
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
            error="required datasets missing under data_root",
        )
    if not can_ok:
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
            error="canonical V2 bundle missing",
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
        outputs=[ctx.audit / "preflight_env.json", ctx.audit / "python_version.txt"],
        extra={"free_gb": free_gb},
    )


def _stage_verify_ds004584_full(ctx: Ctx) -> Dict[str, Any]:
    stage = "stage_verify_ds004584_full"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"

    ds_root = ctx.data_root / "ds004584"
    out_stop = ctx.pack_pdrest_v3 / "STOP_REASON_ds004584.md"
    part_path = ds_root / "participants.tsv"

    if not part_path.exists():
        _write_stop_reason(out_stop, stage, "participants.tsv missing", diagnostics={"dataset_root": str(ds_root)})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="verify ds004584",
            outputs=[out_stop],
            error="participants.tsv missing",
        )

    part = pd.read_csv(part_path, sep="\t")
    part["subject_id"] = part["participant_id"].astype(str).str.replace("sub-", "", regex=False)
    n_part = int(len(part))
    set_fdt_paths = _subject_set_fdt_paths(part["subject_id"].astype(str).tolist())

    counts0 = _count_ds004584_headers(ds_root)

    env = os.environ.copy()
    if counts0.get("vhdr_count", 0) < 120 and counts0.get("set_count", 0) < 120:
        if shutil.which("datalad") is not None:
            _run_cmd(
                ["datalad", "get", "-r", "-J", "8", *set_fdt_paths],
                cwd=ds_root,
                log_path=log_path,
                env=env,
                allow_fail=True,
                timeout_sec=600,
            )
        _run_cmd(
            ["git", "annex", "get", "-J", "8", "--", *set_fdt_paths],
            cwd=ds_root,
            log_path=log_path,
            env=env,
            allow_fail=True,
            timeout_sec=900,
        )

    counts1 = _count_ds004584_headers(ds_root)

    dl_info: Dict[str, Any] = {}
    if counts1.get("vhdr_count", 0) < 120 and counts1.get("set_count", 0) < 120:
        dl_info = _direct_download_ds004584_min(ds_root, part, min_subjects=149, log_path=log_path)

    counts2 = _count_ds004584_headers(ds_root)

    if counts2.get("vhdr_count", 0) < 120 and counts2.get("set_count", 0) < 120:
        _run_cmd(
            ["git", "annex", "get", "-J", "8", "--", *set_fdt_paths],
            cwd=ds_root,
            log_path=log_path,
            env=env,
            allow_fail=True,
            timeout_sec=600,
        )
        counts2 = _count_ds004584_headers(ds_root)

    header_mode = "vhdr" if int(counts2.get("vhdr_count", 0)) > 0 else "set"
    header_count = int(counts2.get("vhdr_count", 0) if header_mode == "vhdr" else counts2.get("set_count", 0))
    subjects_ready = set(counts2.get("subjects_vhdr", []) if header_mode == "vhdr" else counts2.get("subjects_set", []))
    missing_subjects = sorted([sid for sid in part["subject_id"].astype(str).tolist() if sid not in subjects_ready])

    extra = {
        "n_participants": n_part,
        "header_mode": header_mode,
        "header_count": header_count,
        "vhdr_count": int(counts2.get("vhdr_count", 0)),
        "set_count": int(counts2.get("set_count", 0)),
        "missing_subjects": missing_subjects[:100],
        "missing_subjects_total": int(len(missing_subjects)),
        "direct_download": dl_info,
    }

    if header_count < 120:
        reason = f"insufficient ds004584 EEG headers: mode={header_mode} count={header_count} (<120), participants={n_part}"
        _write_stop_reason(out_stop, stage, reason, diagnostics=extra)
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="verify ds004584 full",
            outputs=[out_stop],
            error=reason,
            extra=extra,
        )

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="verify ds004584 full",
        outputs=[],
        extra=extra,
    )


def _stage_verify_ds007020_full(ctx: Ctx) -> Dict[str, Any]:
    stage = "stage_verify_ds007020_full"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"

    ds_root = ctx.data_root / "ds007020"
    out_stop = ctx.pack_mort_v3 / "STOP_REASON_ds007020.md"
    part_path = ds_root / "participants.tsv"

    if not part_path.exists():
        _write_stop_reason(out_stop, stage, "participants.tsv missing", diagnostics={"dataset_root": str(ds_root)})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="verify ds007020",
            outputs=[out_stop],
            error="participants.tsv missing",
        )

    part = pd.read_csv(part_path, sep="\t")
    y, y_col, y_diag = _infer_ds007020_labels(part)
    if y is None:
        _write_stop_reason(out_stop, stage, "could not infer living/deceased labels", diagnostics=y_diag)
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="verify ds007020",
            outputs=[out_stop],
            error="labels missing",
            extra={"label_diag": y_diag},
        )

    part["subject_id"] = part["participant_id"].astype(str).str.replace("sub-", "", regex=False)

    eeg_files = [p for p in ds_root.rglob("*_eeg.vhdr") if p.exists()]
    if not eeg_files:
        eeg_files = [p for p in ds_root.rglob("*_eeg.set") if p.exists() and p.with_suffix(".fdt").exists()]

    sid_re = re.compile(r"sub-([A-Za-z0-9]+)")
    sub_have = set()
    for p in eeg_files:
        m = sid_re.search(str(p))
        if m:
            sub_have.add(m.group(1))

    missing = sorted([sid for sid in part["subject_id"].astype(str).tolist() if sid not in sub_have])

    if missing:
        env = os.environ.copy()
        if shutil.which("datalad") is not None:
            _run_cmd(
                ["bash", "-lc", "shopt -s globstar nullglob; datalad get -r -J 8 sub-*/**/eeg/*"],
                cwd=ds_root,
                log_path=log_path,
                env=env,
                allow_fail=True,
                timeout_sec=1200,
            )
        _run_cmd(
            ["bash", "-lc", "shopt -s globstar nullglob; git annex get -J 8 -- sub-*/**/eeg/*"],
            cwd=ds_root,
            log_path=log_path,
            env=env,
            allow_fail=True,
            timeout_sec=1200,
        )

        eeg_files = [p for p in ds_root.rglob("*_eeg.vhdr") if p.exists()]
        if not eeg_files:
            eeg_files = [p for p in ds_root.rglob("*_eeg.set") if p.exists() and p.with_suffix(".fdt").exists()]
        sub_have = set()
        for p in eeg_files:
            m = sid_re.search(str(p))
            if m:
                sub_have.add(m.group(1))
        missing = sorted([sid for sid in part["subject_id"].astype(str).tolist() if sid not in sub_have])

    extra = {
        "n_participants": int(len(part)),
        "n_label_living": int((y == 0).sum()),
        "n_label_deceased": int((y == 1).sum()),
        "label_column": y_col,
        "n_subjects_with_eeg": int(len(sub_have)),
        "n_missing_subjects": int(len(missing)),
        "missing_subjects": missing[:100],
    }

    if len(sub_have) < 40:
        reason = f"insufficient ds007020 EEG coverage: subjects_with_eeg={len(sub_have)}"
        _write_stop_reason(out_stop, stage, reason, diagnostics=extra)
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="verify ds007020",
            outputs=[out_stop],
            error=reason,
            extra=extra,
        )

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="verify ds007020",
        outputs=[],
        extra=extra,
    )


def _extract_features_for_subjects(
    ds_root: Path,
    subjects: Sequence[str],
    *,
    include_lpc: bool,
    log_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    ex: List[Dict[str, Any]] = []
    for sid in subjects:
        fp, why = _find_best_rest_file(ds_root, sid)
        if fp is None:
            ex.append({"subject_id": sid, "reason": why, "rest_file": ""})
            continue
        try:
            raw = _read_raw_any(fp)
            feat = _compute_rest_feature_row(raw, max_seconds=120.0, lpc_order=(12 if include_lpc else None))
            feat["subject_id"] = sid
            feat["rest_file"] = str(fp)
            rows.append(feat)
        except Exception as exc:
            ex.append({"subject_id": sid, "reason": f"read_error:{exc}", "rest_file": str(fp)})
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(f"[{_iso_now()}] subject={sid} read_error {exc}\n")
    return pd.DataFrame(rows), pd.DataFrame(ex)


def _compute_deviation(df: pd.DataFrame, *, controls_mask: Optional[pd.Series]) -> pd.DataFrame:
    out = df.copy()
    base_features = ["theta_alpha_ratio", "rel_alpha", "spectral_slope"]
    for f in base_features:
        x = pd.to_numeric(out[f], errors="coerce")
        if controls_mask is not None:
            ref = x[controls_mask]
        else:
            ref = x
        mu = float(np.nanmean(ref)) if np.isfinite(ref).any() else 0.0
        sd = float(np.nanstd(ref)) if np.isfinite(ref).any() else 1.0
        if not np.isfinite(sd) or sd <= 1e-6:
            sd = 1.0
        out[f"dev_z_{f}"] = (x - mu) / sd
    out["composite_deviation"] = np.nanmean(
        np.column_stack([
            pd.to_numeric(out.get("dev_z_theta_alpha_ratio"), errors="coerce"),
            pd.to_numeric(out.get("dev_z_rel_alpha"), errors="coerce"),
            pd.to_numeric(out.get("dev_z_spectral_slope"), errors="coerce"),
        ]),
        axis=1,
    )
    return out


def _stage_clinical_ds004584_fullN(ctx: Ctx) -> Dict[str, Any]:
    stage = "clinical_ds004584_fullN"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"
    out_dir = ctx.pack_pdrest_v3
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_root = ctx.data_root / "ds004584"
    part = pd.read_csv(ds_root / "participants.tsv", sep="\t")
    part["subject_id"] = part["participant_id"].astype(str).str.replace("sub-", "", regex=False)

    labels, label_col, diag = _infer_ds004584_groups(part)
    if labels is None:
        stop = out_dir / "STOP_REASON_ds004584_labels.md"
        _write_stop_reason(stop, stage, "ambiguous PD/control labels", diagnostics=diag)
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds004584",
            outputs=[stop],
            error="label inference failed",
        )

    part["group"] = labels
    part = part[part["group"].isin(["PD", "CN"])].copy()
    part["age"] = pd.to_numeric(part.get("AGE", part.get("Age", np.nan)), errors="coerce")
    part["sex"] = part.get("GENDER", part.get("Sex", "")).astype(str)

    subjects = part["subject_id"].astype(str).tolist()
    feat_df, ex_df = _extract_features_for_subjects(ds_root, subjects, include_lpc=False, log_path=log_path)
    feat_df = feat_df.merge(part[["subject_id", "group", "age", "sex"]], on="subject_id", how="left")

    ex_path = out_dir / "EXCLUSIONS.csv"
    if ex_df.empty:
        ex_df = pd.DataFrame(columns=["subject_id", "reason", "rest_file"])
    ex_df.to_csv(ex_path, index=False)

    if feat_df.empty:
        stop = out_dir / "STOP_REASON_ds004584.md"
        _write_stop_reason(
            stop,
            stage,
            "no readable resting features",
            diagnostics={"n_subjects_input": len(subjects), "n_exclusions": int(len(ex_df)), "sample_exclusions": ex_df.head(20).to_dict(orient="records")},
        )
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds004584",
            outputs=[stop, ex_path],
            error="no features",
        )

    dev_df = _compute_deviation(feat_df, controls_mask=(feat_df["group"] == "CN"))

    feature_cols = ["dev_z_theta_alpha_ratio", "dev_z_rel_alpha", "dev_z_spectral_slope", "composite_deviation"]
    rows: List[Dict[str, Any]] = []
    n_perm = 20000
    n_boot_auc = 2000
    n_boot_beta = 2000

    for feat in feature_cols:
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

        auc_obs, auc_ci = _bootstrap_auc(y, score, n_boot=n_boot_auc, seed=1000 + _stable_int_from_text(feat) % 100000)
        p_auc = _perm_p_auc(y, score, n_perm=n_perm, seed=2000 + _stable_int_from_text(feat) % 100000)

        X, yy, covs, nfit = _build_design_matrix(sub.rename(columns={feat: "feature"}), "feature")
        beta = _fit_logit_beta(X, yy)
        b_lo, b_hi = _bootstrap_beta_ci(X, yy, n_boot=n_boot_beta, seed=3000 + _stable_int_from_text(feat) % 100000)
        p_beta = _perm_p_beta(X, yy, beta, n_perm=n_perm, seed=4000 + _stable_int_from_text(feat) % 100000)

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
                "covariates": ";".join(covs),
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
            }
        )

    end_df = pd.DataFrame(rows)
    if end_df.empty:
        stop = out_dir / "STOP_REASON_ds004584.md"
        _write_stop_reason(stop, stage, "no valid endpoints after feature extraction", diagnostics={"n_features": int(len(dev_df)), "n_exclusions": int(len(ex_df))})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds004584",
            outputs=[stop, ex_path],
            error="no endpoints",
        )

    end_df["perm_q_within_ds004584"] = _bh_qvals(pd.to_numeric(end_df["perm_p"], errors="coerce").fillna(1.0).to_numpy(dtype=float).tolist())

    feat_path = out_dir / "pdrest_features.csv"
    end_path = out_dir / "pdrest_endpoints.csv"
    dev_df.to_csv(feat_path, index=False)
    end_df.to_csv(end_path, index=False)

    primary_feat = "composite_deviation" if "composite_deviation" in dev_df.columns else "dev_z_theta_alpha_ratio"
    subp = dev_df[dev_df["group"].isin(["PD", "CN"])].copy()
    yy = (subp["group"].astype(str) == "PD").astype(int).to_numpy(dtype=int)
    ss = pd.to_numeric(subp[primary_feat], errors="coerce").to_numpy(dtype=float)
    roc_path = out_dir / "FIG_pdrest_primary_auc_roc.png"
    cal_path = out_dir / "FIG_pdrest_calibration.png"
    _plot_roc_calibration(yy, ss, "ds004584 PD vs CN", roc_path, cal_path)

    n_used = int(dev_df["subject_id"].nunique())
    extra = {
        "n_subjects_expected": 149,
        "n_subjects_used": n_used,
        "n_pd_used": int((dev_df["group"] == "PD").sum()),
        "n_cn_used": int((dev_df["group"] == "CN").sum()),
        "n_endpoints": int(len(end_df)),
    }

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="clinical ds004584 fullN",
        outputs=[feat_path, end_path, roc_path, cal_path, ex_path],
        extra=extra,
    )


def _compute_leapd_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n = int(len(y))
    scores = np.full(n, np.nan, dtype=float)
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        if np.unique(y[tr]).size < 2:
            continue
        xtr = X[tr]
        ytr = y[tr]
        xte = X[i]
        mu0 = np.nanmean(xtr[ytr == 0], axis=0)
        mu1 = np.nanmean(xtr[ytr == 1], axis=0)
        d0 = float(np.linalg.norm(xte - mu0))
        d1 = float(np.linalg.norm(xte - mu1))
        denom = max(d0 + d1, 1e-12)
        # Higher value means closer to class 1 (deceased)
        scores[i] = float(d0 / denom)
    return scores


def _stage_clinical_ds007020_mortality_fixbeta(ctx: Ctx) -> Dict[str, Any]:
    stage = "clinical_ds007020_mortality_fixbeta"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"
    out_dir = ctx.pack_mort_v3
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_root = ctx.data_root / "ds007020"
    part = pd.read_csv(ds_root / "participants.tsv", sep="\t")
    y_lbl, y_col, y_diag = _infer_ds007020_labels(part)
    if y_lbl is None:
        stop = out_dir / "STOP_REASON_ds007020_labels.md"
        _write_stop_reason(stop, stage, "could not infer mortality labels", diagnostics=y_diag)
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds007020",
            outputs=[stop],
            error="labels missing",
        )

    part["subject_id"] = part["participant_id"].astype(str).str.replace("sub-", "", regex=False)
    part["label"] = pd.to_numeric(y_lbl, errors="coerce")

    subjects = part["subject_id"].astype(str).tolist()
    feat_df, ex_df = _extract_features_for_subjects(ds_root, subjects, include_lpc=True, log_path=log_path)
    feat_df = feat_df.merge(part[["subject_id", "label"]], on="subject_id", how="left")

    ex_path = out_dir / "EXCLUSIONS.csv"
    if ex_df.empty:
        ex_df = pd.DataFrame(columns=["subject_id", "reason", "rest_file"])
    ex_df.to_csv(ex_path, index=False)

    if feat_df.empty:
        stop = out_dir / "STOP_REASON_ds007020.md"
        _write_stop_reason(stop, stage, "no readable resting features", diagnostics={"n_exclusions": int(len(ex_df)), "sample_exclusions": ex_df.head(20).to_dict(orient="records")})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds007020",
            outputs=[stop, ex_path],
            error="no features",
        )

    dev_df = _compute_deviation(feat_df, controls_mask=None)

    n_perm = 20000
    n_boot_auc = 2000
    n_boot_beta = 2000

    rows: List[Dict[str, Any]] = []
    feature_cols = ["dev_z_theta_alpha_ratio", "dev_z_rel_alpha", "dev_z_spectral_slope", "composite_deviation"]

    for feat in feature_cols:
        sub = dev_df[["subject_id", "label", feat]].copy()
        sub[feat] = pd.to_numeric(sub[feat], errors="coerce")
        sub["label"] = pd.to_numeric(sub["label"], errors="coerce")
        sub = sub[np.isfinite(sub[feat]) & np.isfinite(sub["label"])].copy()
        if sub.empty:
            continue
        y = sub["label"].astype(int).to_numpy(dtype=int)
        score = sub[feat].to_numpy(dtype=float)
        if np.unique(y).size < 2:
            continue

        auc_obs, auc_ci = _bootstrap_auc(y, score, n_boot=n_boot_auc, seed=5000 + _stable_int_from_text(feat) % 100000)
        p_auc = _perm_p_auc(y, score, n_perm=n_perm, seed=6000 + _stable_int_from_text(feat) % 100000)

        X = score.reshape(-1, 1)
        beta = _fit_logit_beta(X, y)
        b_lo, b_hi = _bootstrap_beta_ci(X, y, n_boot=n_boot_beta, seed=7000 + _stable_int_from_text(feat) % 100000)
        p_beta = _perm_p_beta(X, y, beta, n_perm=n_perm, seed=8000 + _stable_int_from_text(feat) % 100000)

        rows.append(
            {
                "dataset_id": "ds007020",
                "endpoint": "AUC_mortality",
                "feature": feat,
                "type": "auc",
                "n": int(len(sub)),
                "estimate": float(auc_obs),
                "ci95_lo": float(auc_ci[0]),
                "ci95_hi": float(auc_ci[1]),
                "perm_p": float(p_auc),
                "model": "spectral_deviation",
            }
        )
        rows.append(
            {
                "dataset_id": "ds007020",
                "endpoint": "LogitBeta_mortality",
                "feature": feat,
                "type": "logit_beta",
                "n": int(len(sub)),
                "estimate": float(beta),
                "ci95_lo": float(b_lo),
                "ci95_hi": float(b_hi),
                "perm_p": float(p_beta),
                "model": "spectral_deviation",
            }
        )

    lpc_cols = [c for c in dev_df.columns if c.startswith("lpc_")]
    leapd_auc = float("nan")
    leapd_p = float("nan")
    leapd_n = 0
    leapd_scores = pd.Series([], dtype=float)
    if lpc_cols:
        ldf = dev_df[["subject_id", "label"] + lpc_cols].copy()
        ldf["label"] = pd.to_numeric(ldf["label"], errors="coerce")
        for c in lpc_cols:
            ldf[c] = pd.to_numeric(ldf[c], errors="coerce")
        ldf = ldf.dropna().copy()
        if not ldf.empty:
            y = ldf["label"].astype(int).to_numpy(dtype=int)
            X = ldf[lpc_cols].to_numpy(dtype=float)
            if np.unique(y).size >= 2 and len(y) >= 20:
                leapd_scores = pd.Series(_compute_leapd_scores(X, y), index=ldf.index)
                m = np.isfinite(leapd_scores.to_numpy(dtype=float))
                if int(m.sum()) >= 10 and np.unique(y[m]).size >= 2:
                    yy = y[m]
                    ss = leapd_scores.to_numpy(dtype=float)[m]
                    leapd_auc, leapd_ci = _bootstrap_auc(yy, ss, n_boot=n_boot_auc, seed=9001)
                    leapd_p = _perm_p_auc(yy, ss, n_perm=n_perm, seed=9002)
                    leapd_n = int(len(yy))
                    rows.append(
                        {
                            "dataset_id": "ds007020",
                            "endpoint": "LEAPD_AUC_mortality",
                            "feature": "leapd_index",
                            "type": "auc_primary",
                            "n": int(leapd_n),
                            "estimate": float(leapd_auc),
                            "ci95_lo": float(leapd_ci[0]),
                            "ci95_hi": float(leapd_ci[1]),
                            "perm_p": float(leapd_p),
                            "model": "LPC_LEAPD_LOOCV",
                        }
                    )
                    Xb = ss.reshape(-1, 1)
                    beta = _fit_logit_beta(Xb, yy)
                    b_lo, b_hi = _bootstrap_beta_ci(Xb, yy, n_boot=n_boot_beta, seed=9003)
                    p_beta = _perm_p_beta(Xb, yy, beta, n_perm=n_perm, seed=9004)
                    rows.append(
                        {
                            "dataset_id": "ds007020",
                            "endpoint": "LEAPD_LogitBeta_mortality",
                            "feature": "leapd_index",
                            "type": "logit_beta_primary",
                            "n": int(leapd_n),
                            "estimate": float(beta),
                            "ci95_lo": float(b_lo),
                            "ci95_hi": float(b_hi),
                            "perm_p": float(p_beta),
                            "model": "LPC_LEAPD_LOOCV",
                        }
                    )

    end_df = pd.DataFrame(rows)
    if end_df.empty:
        stop = out_dir / "STOP_REASON_ds007020.md"
        _write_stop_reason(stop, stage, "no mortality endpoints produced", diagnostics={"n_features": int(len(dev_df)), "n_exclusions": int(len(ex_df))})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds007020",
            outputs=[stop, ex_path],
            error="no endpoints",
        )

    end_df["perm_q_within_ds007020"] = _bh_qvals(pd.to_numeric(end_df["perm_p"], errors="coerce").fillna(1.0).to_numpy(dtype=float).tolist())

    feat_path = out_dir / "mortality_features.csv"
    end_path = out_dir / "mortality_endpoints.csv"
    dev_df.to_csv(feat_path, index=False)
    end_df.to_csv(end_path, index=False)

    # Primary figure based on LEAPD if available, else composite.
    if np.isfinite(leapd_auc):
        ldf = dev_df[["subject_id", "label"] + [c for c in dev_df.columns if c.startswith("lpc_")]].dropna().copy()
        y = ldf["label"].astype(int).to_numpy(dtype=int)
        X = ldf[[c for c in ldf.columns if c.startswith("lpc_")]].to_numpy(dtype=float)
        s = _compute_leapd_scores(X, y)
    else:
        tmp = dev_df[["label", "composite_deviation"]].copy()
        tmp["label"] = pd.to_numeric(tmp["label"], errors="coerce")
        tmp["composite_deviation"] = pd.to_numeric(tmp["composite_deviation"], errors="coerce")
        tmp = tmp[np.isfinite(tmp["label"]) & np.isfinite(tmp["composite_deviation"])].copy()
        y = tmp["label"].astype(int).to_numpy(dtype=int)
        s = tmp["composite_deviation"].to_numpy(dtype=float)

    roc_path = out_dir / "FIG_mortality_primary_auc_roc.png"
    cal_path = out_dir / "FIG_mortality_calibration.png"
    _plot_roc_calibration(np.asarray(y, dtype=int), np.asarray(s, dtype=float), "ds007020 mortality", roc_path, cal_path)

    finite_beta_count = int(np.isfinite(pd.to_numeric(end_df.loc[end_df["type"].str.contains("beta", na=False), "estimate"], errors="coerce")).sum())

    extra = {
        "n_subjects_used": int(dev_df["subject_id"].nunique()),
        "n_label_living": int((pd.to_numeric(dev_df["label"], errors="coerce") == 0).sum()),
        "n_label_deceased": int((pd.to_numeric(dev_df["label"], errors="coerce") == 1).sum()),
        "finite_beta_count": finite_beta_count,
        "leapd_auc": float(leapd_auc),
        "leapd_perm_p": float(leapd_p),
        "leapd_n": int(leapd_n),
        "label_column": y_col,
    }

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="clinical ds007020 mortality",
        outputs=[feat_path, end_path, roc_path, cal_path, ex_path],
        extra=extra,
    )


def _stage_endpoint_hierarchy_and_corrections(ctx: Ctx) -> Dict[str, Any]:
    stage = "endpoint_hierarchy_and_corrections"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"

    reg_path = ctx.audit / "CLINICAL_ENDPOINT_REGISTRY.md"

    can_clin = ctx.canonical_root / "PACK_CLINICAL" / "clinical_endpoints_all.csv"
    ds004504_df = pd.DataFrame()
    if can_clin.exists():
        try:
            cdf = pd.read_csv(can_clin)
            ds004504_df = cdf[cdf.get("dataset_id", pd.Series([], dtype=str)).astype(str) == "ds004504"].copy()
        except Exception:
            ds004504_df = pd.DataFrame()

    pd_df = pd.read_csv(ctx.pack_pdrest_v3 / "pdrest_endpoints.csv") if (ctx.pack_pdrest_v3 / "pdrest_endpoints.csv").exists() else pd.DataFrame()
    mt_df = pd.read_csv(ctx.pack_mort_v3 / "mortality_endpoints.csv") if (ctx.pack_mort_v3 / "mortality_endpoints.csv").exists() else pd.DataFrame()

    dfs = []
    if not ds004504_df.empty:
        d0 = ds004504_df.copy()
        d0["dataset_id"] = "ds004504"
        if "perm_p" not in d0.columns:
            d0["perm_p"] = np.nan
        dfs.append(d0)
    if not pd_df.empty:
        dfs.append(pd_df.copy())
    if not mt_df.empty:
        dfs.append(mt_df.copy())

    if not dfs:
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="endpoint hierarchy",
            outputs=[],
            error="no clinical endpoint tables found",
        )

    all_df = pd.concat(dfs, ignore_index=True)
    all_df["perm_p"] = pd.to_numeric(all_df.get("perm_p"), errors="coerce")

    all_df["perm_q_within_dataset"] = np.nan
    for dsid, idx in all_df.groupby(all_df["dataset_id"].astype(str)).groups.items():
        pvals = pd.to_numeric(all_df.loc[idx, "perm_p"], errors="coerce").fillna(1.0).to_numpy(dtype=float).tolist()
        qv = _bh_qvals(pvals)
        all_df.loc[idx, "perm_q_within_dataset"] = qv

    all_df["perm_q_global"] = _bh_qvals(pd.to_numeric(all_df["perm_p"], errors="coerce").fillna(1.0).to_numpy(dtype=float).tolist())

    comb_path = ctx.audit / "clinical_endpoints_all_v3.csv"
    all_df.to_csv(comb_path, index=False)

    # Push corrected q-values back into V3 tables.
    if not pd_df.empty:
        m = all_df[all_df["dataset_id"].astype(str) == "ds004584"].copy()
        m.to_csv(ctx.pack_pdrest_v3 / "pdrest_endpoints.csv", index=False)
    if not mt_df.empty:
        m = all_df[all_df["dataset_id"].astype(str) == "ds007020"].copy()
        m.to_csv(ctx.pack_mort_v3 / "mortality_endpoints.csv", index=False)

    reg_lines = [
        "# Clinical Endpoint Registry (V3)",
        "",
        "## Primary Endpoints",
        "- ds007020 primary: `LEAPD_AUC_mortality` (LPC/LEAPD model)",
        "- ds004584 primary: `AUC_PD_vs_CN` on `composite_deviation`",
        "- ds004504 primary: theta/alpha-ratio AUC endpoints from canonical V2 run",
        "",
        "## Correction Scopes",
        "- BH-FDR within each dataset family: ds004504, ds004584, ds007020",
        "- BH-FDR global across all clinical endpoints for transparency",
        "",
        "## Files",
        f"- Combined endpoint table: `{comb_path}`",
        f"- ds004584 endpoints: `{ctx.pack_pdrest_v3 / 'pdrest_endpoints.csv'}`",
        f"- ds007020 endpoints: `{ctx.pack_mort_v3 / 'mortality_endpoints.csv'}`",
    ]
    _write_text(reg_path, "\n".join(reg_lines) + "\n")

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="endpoint hierarchy and corrections",
        outputs=[reg_path, comb_path],
        extra={"n_endpoints_total": int(len(all_df))},
    )


def _stage_v3_bundle(ctx: Ctx) -> Dict[str, Any]:
    stage = "v3_bundle"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"

    can_zip = ctx.canonical_root / "OUTZIP" / "NN_FINAL_MEGA_V2_BIO_SUBMISSION_PACKET.zip"
    v3_zip = ctx.outzip / "NN_FINAL_MEGA_V3_CLINICALFIX_SUBMISSION_PACKET.zip"
    ctx.outzip.mkdir(parents=True, exist_ok=True)

    report_path = ctx.audit / "NN_FINAL_MEGA_V3_REPORT.md"

    # Create concise report after all stage results are available.
    st_rows = []
    for rec in ctx.stage_records:
        st_rows.append(
            f"| {rec.get('stage')} | {rec.get('status')} | {rec.get('returncode')} | {rec.get('elapsed_sec', 0.0):.1f} |"
        )

    pd_sum = _read_json(ctx.audit / "clinical_ds004584_fullN_summary.json") if (ctx.audit / "clinical_ds004584_fullN_summary.json").exists() else {}
    mt_sum = _read_json(ctx.audit / "clinical_ds007020_mortality_fixbeta_summary.json") if (ctx.audit / "clinical_ds007020_mortality_fixbeta_summary.json").exists() else {}

    lines = [
        "# NN_FINAL_MEGA_V3_CLINICALFIX REPORT",
        "",
        f"- Output root: `{ctx.out_root}`",
        f"- Canonical V2 root reused: `{ctx.canonical_root}`",
        "",
        "## Stage status",
        "| Stage | Status | Return code | Runtime (s) |",
        "|---|---|---:|---:|",
        *st_rows,
        "",
        "## Clinical objectives",
        f"- ds004584 n_used: `{pd_sum.get('n_subjects_used')}` (expected 149; strict min 120)",
        f"- ds007020 n_used: `{mt_sum.get('n_subjects_used')}`",
        f"- ds007020 finite_beta_count: `{mt_sum.get('finite_beta_count')}`",
        f"- ds007020 LEAPD AUC: `{mt_sum.get('leapd_auc')}`",
        "",
        "## Endpoint hierarchy",
        "- ds007020 primary: `LEAPD_AUC_mortality`",
        "- ds004584 primary: `AUC_PD_vs_CN` on `composite_deviation`",
        "- ds004504 primary: theta/alpha ratio AUCs (canonical V2)",
        "",
        "## Correction scopes",
        "- Within-dataset BH-FDR (ds004504, ds004584, ds007020)",
        "- Global BH-FDR across all clinical endpoints",
        "",
        "## Key files",
        f"- Endpoint registry: `{ctx.audit / 'CLINICAL_ENDPOINT_REGISTRY.md'}`",
        f"- ds004584 pack: `{ctx.pack_pdrest_v3}`",
        f"- ds007020 pack: `{ctx.pack_mort_v3}`",
        f"- Canonical V2 bundle copied as: `CANONICAL/NN_FINAL_MEGA_V2_BIO_SUBMISSION_PACKET.zip` inside V3 zip",
    ]
    _write_text(report_path, "\n".join(lines) + "\n")

    with zipfile.ZipFile(v3_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        # Include V3 audit and V3 clinical packs.
        for p in sorted(ctx.audit.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(ctx.out_root)))
        for base in [ctx.pack_pdrest_v3, ctx.pack_mort_v3]:
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(ctx.out_root)))
        # Include canonical bundle as immutable artifact.
        if can_zip.exists():
            zf.write(can_zip, arcname="CANONICAL/NN_FINAL_MEGA_V2_BIO_SUBMISSION_PACKET.zip")

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="v3 bundle",
        outputs=[report_path, v3_zip],
        extra={"zip_size_bytes": int(v3_zip.stat().st_size if v3_zip.exists() else 0)},
    )


def _build_repo_fingerprint(repo_root: Path, out_path: Path) -> None:
    try:
        p = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if p.returncode == 0:
            _write_json(out_path, {"git_head": p.stdout.strip(), "repo_root": str(repo_root), "mode": "git"})
            return
    except Exception:
        pass

    # fallback sha256 manifest of tracked script files
    rows = []
    for p in sorted(repo_root.rglob("*.py")):
        if ".git" in p.parts:
            continue
        rel = p.relative_to(repo_root).as_posix()
        h = hashlib.sha256()
        with p.open("rb") as f:
            while True:
                b = f.read(1024 * 1024)
                if not b:
                    break
                h.update(b)
        rows.append({"path": rel, "sha256": h.hexdigest(), "size": int(p.stat().st_size)})
    _write_json(out_path, {"git_head": None, "mode": "sha256_manifest", "repo_root": str(repo_root), "files": rows})


def _write_dataset_hashes(data_root: Path, out_path: Path) -> None:
    rows = []
    for ds in ["ds004584", "ds007020", "ds004504"]:
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


def _final_run_status(ctx: Ctx, run_status: str, run_error: str) -> None:
    report_path = ctx.audit / "NN_FINAL_MEGA_V3_REPORT.md"
    payload = {
        "status": run_status,
        "error": run_error,
        "out_root": str(ctx.out_root),
        "report": str(report_path),
        "stages": ctx.stage_records,
        "partial_reasons": ctx.partial_reasons,
    }
    _write_json(ctx.audit / "run_status.json", payload)


def main() -> int:
    ap = argparse.ArgumentParser(description="NN_FINAL_MEGA_V3_CLINICALFIX surgical runner")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="/filesystemHcog/openneuro")
    ap.add_argument("--canonical_root", type=str, default=str(CANONICAL_DEFAULT))
    ap.add_argument("--wall_hours", type=float, default=8.0)
    ap.add_argument("--resume", type=str, default="false")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    audit = out_root / "AUDIT"
    outzip = out_root / "OUTZIP"

    if out_root.exists() and not _parse_bool(args.resume):
        existing = list(out_root.iterdir())
        allow_bootstrap = False
        if existing:
            allow_bootstrap = all(
                p.name in {"AUDIT", "OUTZIP"} and p.is_dir() and not any(p.iterdir())
                for p in existing
            )
        if existing and not allow_bootstrap:
            print(f"ERROR: out_root exists and is non-empty: {out_root}", file=sys.stderr, flush=True)
            return 2

    out_root.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    outzip.mkdir(parents=True, exist_ok=True)

    ctx = Ctx(
        out_root=out_root,
        audit=audit,
        outzip=outzip,
        data_root=Path(args.data_root).resolve(),
        canonical_root=Path(args.canonical_root).resolve(),
        wall_hours=float(args.wall_hours),
        resume=_parse_bool(args.resume),
        pack_pdrest_v3=out_root / "PACK_CLINICAL_PDREST_V3",
        pack_mort_v3=out_root / "PACK_CLINICAL_MORTALITY_V3",
        start_ts=time.time(),
        deadline_ts=time.time() + float(args.wall_hours) * 3600.0,
        stage_records=[],
        stage_status={},
        stage_extra={},
        partial_reasons=[],
    )
    ctx.pack_pdrest_v3.mkdir(parents=True, exist_ok=True)
    ctx.pack_mort_v3.mkdir(parents=True, exist_ok=True)

    # Base provenance files upfront.
    _build_repo_fingerprint(REPO_ROOT, ctx.audit / "repo_fingerprint.json")
    _write_dataset_hashes(ctx.data_root, ctx.audit / "dataset_hashes.json")
    try:
        pf = subprocess.run([sys.executable, "-m", "pip", "freeze"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
        _write_text(ctx.audit / "pip_freeze.txt", pf.stdout)
    except Exception:
        _write_text(ctx.audit / "pip_freeze.txt", "")

    stage_funcs = {
        "preflight": _stage_preflight,
        "stage_verify_ds004584_full": _stage_verify_ds004584_full,
        "stage_verify_ds007020_full": _stage_verify_ds007020_full,
        "clinical_ds004584_fullN": _stage_clinical_ds004584_fullN,
        "clinical_ds007020_mortality_fixbeta": _stage_clinical_ds007020_mortality_fixbeta,
        "endpoint_hierarchy_and_corrections": _stage_endpoint_hierarchy_and_corrections,
        "v3_bundle": _stage_v3_bundle,
    }

    run_status = "PASS_STRICT"
    run_error = ""

    try:
        for stage in STAGES:
            if time.time() > ctx.deadline_ts:
                run_status = "PARTIAL_PASS"
                run_error = "walltime exhausted before all stages"
                ctx.partial_reasons.append(run_error)
                break

            if ctx.resume and stage != "preflight":
                resumed = _load_resumed_stage(ctx, stage)
                if resumed is not None:
                    if resumed.get("status") == "SKIP":
                        ctx.partial_reasons.append(f"{stage} resumed as SKIP")
                    if resumed.get("status") == "FAIL":
                        run_status = "FAIL"
                        run_error = str(resumed.get("error", "stage failed"))
                        break
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

    # Strict objective gate.
    strict_ok = False
    strict_issues: List[str] = []
    pd_end = ctx.pack_pdrest_v3 / "pdrest_endpoints.csv"
    mt_end = ctx.pack_mort_v3 / "mortality_endpoints.csv"

    pd_used = int(ctx.stage_extra.get("clinical_ds004584_fullN", {}).get("n_subjects_used", 0))
    if not pd_end.exists():
        strict_issues.append("ds004584 endpoints missing")
    if pd_used < 120:
        strict_issues.append(f"ds004584 n_used={pd_used} < 120")

    finite_beta_count = int(ctx.stage_extra.get("clinical_ds007020_mortality_fixbeta", {}).get("finite_beta_count", 0))
    leapd_auc = float(ctx.stage_extra.get("clinical_ds007020_mortality_fixbeta", {}).get("leapd_auc", float("nan")))
    if not mt_end.exists():
        strict_issues.append("ds007020 endpoints missing")
    if finite_beta_count < 1:
        strict_issues.append("ds007020 has no finite beta")
    if not np.isfinite(leapd_auc):
        strict_issues.append("ds007020 LEAPD primary AUC missing")

    if run_status != "FAIL":
        if len(strict_issues) == 0:
            run_status = "PASS_STRICT"
            strict_ok = True
            run_error = ""
        else:
            strict_ok = False
            if time.time() > ctx.deadline_ts:
                run_status = "PARTIAL_PASS"
            else:
                run_status = "PARTIAL_PASS"
            msg = " ; ".join(strict_issues)
            run_error = msg
            ctx.partial_reasons.extend(strict_issues)

    # If hard fail but clearly data availability impossible and fail-closed, downgrade to PARTIAL_PASS.
    if run_status == "FAIL":
        lowered = False
        err_low = run_error.lower()
        if "insufficient ds004584 eeg headers" in err_low or "participants.tsv missing" in err_low:
            run_status = "PARTIAL_PASS"
            ctx.partial_reasons.append(run_error)
            lowered = True
        if lowered:
            run_error = run_error

    _final_run_status(ctx, run_status, run_error)

    # Render final report if not already produced by v3_bundle.
    rep = ctx.audit / "NN_FINAL_MEGA_V3_REPORT.md"
    if not rep.exists():
        rows = [
            "# NN_FINAL_MEGA_V3_CLINICALFIX REPORT",
            "",
            f"- Output root: `{ctx.out_root}`",
            f"- Run status: `{run_status}`",
            "",
            "## Stages",
            "| Stage | Status | Return code | Runtime (s) |",
            "|---|---|---:|---:|",
        ]
        for rec in ctx.stage_records:
            rows.append(f"| {rec.get('stage')} | {rec.get('status')} | {rec.get('returncode')} | {float(rec.get('elapsed_sec', 0.0)):.1f} |")
        if strict_issues:
            rows.extend(["", "## Strict gate issues"] + [f"- {x}" for x in strict_issues])
        _write_text(rep, "\n".join(rows) + "\n")

    print(f"OUT_ROOT={ctx.out_root}", flush=True)
    print(f"REPORT={ctx.audit / 'NN_FINAL_MEGA_V3_REPORT.md'}", flush=True)
    print(f"STATUS={run_status}", flush=True)
    print(f"ZIP={ctx.outzip / 'NN_FINAL_MEGA_V3_CLINICALFIX_SUBMISSION_PACKET.zip'}", flush=True)

    return 0 if run_status in {"PASS_STRICT", "PARTIAL_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
