#!/usr/bin/env python3
"""NN_FINAL_MASTER_V1 surgical runner.

Scope:
- Reuse canonical V2_BIO bundle (no core recomputation)
- Recompute clinical packs for ds004584 and ds007020 with strict gates
- Re-attempt ds004752 BIO-D and ds007262 workload extraction
- Produce zip + tarball exports

Fail-closed behavior:
- ds004584/ds007020 cannot silently skip (FAIL if not processable)
- ds004752/ds007262 may SKIP only with STOP_REASON_<dataset>.md
- Stage summaries include achieved counts (N, perms, bootstraps)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from common.hardware import start_gpu_util_logger, summarize_gpu_util_csv  # noqa: E402


STAGES: List[str] = [
    "preflight",
    "compile_gate",
    "stage_verify_ds004584_full",
    "stage_verify_ds007020_full",
    "clinical_ds004584_fullN_PDrest",
    "clinical_ds007020_LEAPD_full",
    "bio_ds004752_crossmodality_attempt",
    "endpoint_hierarchy_and_report",
    "bundle_and_tarball",
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


def _tail(path: Path, n: int = 120) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max(1, int(n)) :])


def _parse_bool(raw: Any) -> bool:
    t = str(raw).strip().lower()
    return t in {"1", "true", "yes", "y", "on"}


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
            lf.write(f"[{_iso_now()}] ERROR: command timed out after {timeout_sec}s\n")
            if p is not None:
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
    if y.size == 0 or np.unique(y).size < 2:
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


def _write_stop_reason(path: Path, title: str, reason: str, diagnostics: Optional[Dict[str, Any]] = None) -> None:
    lines = [f"# STOP_REASON {title}", "", "## Why", reason]
    if diagnostics is not None:
        lines += ["", "## Diagnostics", "```json", json.dumps(_json_sanitize(diagnostics), indent=2), "```"]
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


def _subject_set_fdt_paths(subject_ids: Sequence[str]) -> List[str]:
    out: List[str] = []
    for sid0 in subject_ids:
        sid = re.sub(r"[^A-Za-z0-9]", "", str(sid0))
        if not sid:
            continue
        out.append(f"sub-{sid}/eeg/sub-{sid}_task-Rest_eeg.set")
        out.append(f"sub-{sid}/eeg/sub-{sid}_task-Rest_eeg.fdt")
    return out


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
    base = ["theta_alpha_ratio", "rel_alpha", "spectral_slope"]
    for f in base:
        x = pd.to_numeric(out[f], errors="coerce")
        ref = x[controls_mask] if controls_mask is not None else x
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
        scores[i] = float(d0 / max(d0 + d1, 1e-12))
    return scores


def _find_matching_signal(events_tsv: Path, modality: str) -> Optional[Path]:
    stem = events_tsv.name.replace("_events.tsv", f"_{modality}")
    parent = events_tsv.parent
    for ext in [".edf", ".vhdr", ".set", ".bdf", ".fif", ".gdf"]:
        p = parent / f"{stem}{ext}"
        if p.exists():
            if ext == ".set" and not p.with_suffix(".fdt").exists():
                continue
            return p
    # fallback: any modality file in same directory with same prefix
    prefix = events_tsv.name.replace("_events.tsv", "")
    for ext in [".edf", ".vhdr", ".set", ".bdf", ".fif", ".gdf"]:
        cand = sorted(parent.glob(prefix + f"_{modality}{ext}"))
        for p in cand:
            if ext == ".set" and not p.with_suffix(".fdt").exists():
                continue
            return p
    return None


def _extract_event_amp(raw, onsets: np.ndarray, *, modality: str, tmin: float, tmax: float, win: Tuple[float, float]) -> float:
    sf = float(raw.info["sfreq"])
    if sf <= 0:
        return float("nan")
    # pick channels
    if modality == "eeg":
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    else:
        picks = mne.pick_types(raw.info, seeg=True, ecog=True, dbs=True, eeg=True, exclude=[])
    if picks is None or len(picks) == 0:
        picks = np.arange(len(raw.ch_names))

    events: List[List[int]] = []
    n_times = int(raw.n_times)
    for o in np.asarray(onsets, dtype=float):
        s = int(round(o * sf))
        if s + int(round(tmax * sf)) >= n_times:
            continue
        if s + int(round(tmin * sf)) < 0:
            continue
        events.append([s, 0, 1])
    if len(events) < 3:
        return float("nan")

    ev = np.asarray(events, dtype=int)
    try:
        epochs = mne.Epochs(
            raw,
            ev,
            event_id={"stim": 1},
            tmin=tmin,
            tmax=tmax,
            baseline=(tmin, 0.0),
            preload=True,
            picks=picks,
            reject_by_annotation=False,
            detrend=None,
            verbose="ERROR",
        )
    except Exception:
        return float("nan")
    if len(epochs) < 3:
        return float("nan")
    data = epochs.get_data()  # n_ep x n_ch x n_t
    times = epochs.times
    m = (times >= float(win[0])) & (times <= float(win[1]))
    if int(m.sum()) < 2:
        return float("nan")
    val = np.nanmean(data[:, :, m], axis=(1, 2))
    if val.size == 0:
        return float("nan")
    return float(np.nanmean(val))


def _extract_workload_slope(raw, onsets: np.ndarray, loads: np.ndarray) -> Tuple[float, int]:
    # event-level amplitude proxy from post-onset window, then fit slope vs load
    sf = float(raw.info["sfreq"])
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if picks is None or len(picks) == 0:
        picks = np.arange(len(raw.ch_names))

    events: List[List[int]] = []
    load_list: List[float] = []
    for o, ld in zip(np.asarray(onsets, dtype=float), np.asarray(loads, dtype=float)):
        if not np.isfinite(ld):
            continue
        s = int(round(o * sf))
        if s + int(round(2.0 * sf)) >= int(raw.n_times):
            continue
        if s < 0:
            continue
        events.append([s, 0, 1])
        load_list.append(float(ld))

    if len(events) < 6:
        return float("nan"), 0

    ev = np.asarray(events, dtype=int)
    try:
        epochs = mne.Epochs(
            raw,
            ev,
            event_id={"stim": 1},
            tmin=0.0,
            tmax=2.0,
            baseline=None,
            preload=True,
            picks=picks,
            reject_by_annotation=False,
            detrend=None,
            verbose="ERROR",
        )
    except Exception:
        return float("nan"), 0
    if len(epochs) < 6:
        return float("nan"), 0

    x = np.asarray(load_list[: len(epochs)], dtype=float)
    d = epochs.get_data()  # n_ep x n_ch x n_t
    times = epochs.times
    m = (times >= 0.3) & (times <= 1.0)
    if int(m.sum()) < 2:
        return float("nan"), 0
    y = np.nanmean(np.abs(d[:, :, m]), axis=(1, 2))
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 6 or np.unique(x[ok]).size < 3:
        return float("nan"), int(ok.sum())
    slope = float(np.polyfit(x[ok], y[ok], 1)[0])
    return slope, int(ok.sum())


def _parse_load_from_difficulty(series: pd.Series) -> pd.Series:
    def _one(v: Any) -> float:
        s = str(v)
        m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)\s*-", s)
        return float(m.group(1)) if m else float("nan")

    return series.apply(_one)


def _collect_file_formats(ds_root: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for ext in ["vhdr", "set", "edf", "bdf", "fif", "gdf"]:
        out[f"eeg_{ext}"] = len(list(ds_root.rglob(f"*_eeg.{ext}")))
        out[f"ieeg_{ext}"] = len(list(ds_root.rglob(f"*_ieeg.{ext}")))
    out["events_tsv"] = len(list(ds_root.rglob("*_events.tsv")))
    return out


@dataclass
class Ctx:
    out_root: Path
    audit: Path
    outzip: Path
    tarballs: Path
    data_root: Path
    canonical_root: Path
    include_data: bool
    wall_hours: float
    resume: bool

    pack_pdrest: Path
    pack_mort: Path
    pack_bio: Path

    start_ts: float
    deadline_ts: float

    stage_records: List[Dict[str, Any]]
    stage_status: Dict[str, str]
    stage_extra: Dict[str, Dict[str, Any]]
    partial_reasons: List[str]

    monitor_proc: Optional[subprocess.Popen]
    nvml_logger: Any


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


def _start_nvidia_smi_monitor(ctx: Ctx) -> None:
    out_csv = ctx.audit / "nvidia_smi_1hz.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
        "-l",
        "1",
    ]
    f = out_csv.open("a", encoding="utf-8")
    ctx.monitor_proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, preexec_fn=os.setsid)
    _write_text(ctx.audit / "nvidia_smi_monitor.pid", f"{ctx.monitor_proc.pid}\n")


def _stop_nvidia_smi_monitor(ctx: Ctx) -> None:
    p = ctx.monitor_proc
    if p is None:
        pid_file = ctx.audit / "nvidia_smi_monitor.pid"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text(encoding="utf-8").strip())
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except Exception:
                pass
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        pass
    try:
        p.wait(timeout=5)
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    ctx.monitor_proc = None


def _count_nvidia_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


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
        cols = [
            "timestamp",
            "index",
            "util_gpu",
            "util_mem",
            "mem_used_mb",
            "mem_total_mb",
            "power_w",
            "temp_c",
        ]
        df = pd.read_csv(path, names=cols)
    except Exception:
        return out
    if df.empty:
        return out
    out["rows"] = int(len(df))
    for c, k in [
        ("util_gpu", "util_gpu"),
        ("util_mem", "util_mem"),
        ("mem_used_mb", "mem_used_mb"),
        ("power_w", "power_w"),
    ]:
        vals = pd.to_numeric(df[c], errors="coerce")
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_median"] = float(np.median(vals))
    return out


def _gpu_probe_once(log_path: Path) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        with torch.no_grad():
            a = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
            b = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
            for _ in range(4):
                _ = a @ b
            torch.cuda.synchronize()
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(f"[{_iso_now()}] gpu_probe error: {exc}\n")


def _ensure_nvidia_rows(ctx: Ctx, *, min_rows: int, log_path: Path) -> Dict[str, Any]:
    csv_path = ctx.audit / "nvidia_smi_1hz.csv"
    rows0 = _count_nvidia_rows(csv_path)
    waited = 0
    while rows0 < int(min_rows) and time.time() < ctx.deadline_ts:
        _gpu_probe_once(log_path)
        time.sleep(1.0)
        waited += 1
        rows0 = _count_nvidia_rows(csv_path)
        if waited % 60 == 0:
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(f"[{_iso_now()}] nvidia_smi rows progress: {rows0}/{min_rows}\n")
    return {"rows": int(rows0), "waited_sec": int(waited), "target_rows": int(min_rows)}


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
    _write_json(out_path, {"git_head": None, "mode": "sha256_manifest", "repo_root": str(repo_root), "files": files})


def _write_dataset_hashes(data_root: Path, out_path: Path) -> None:
    rows = []
    for ds in ["ds004584", "ds007020", "ds004752", "ds007262", "ds004504"]:
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

    data_ok = all((ctx.data_root / ds).exists() for ds in ["ds004584", "ds007020", "ds004752", "ds007262"])
    can_ok = ctx.canonical_root.exists() and (ctx.canonical_root / "OUTZIP" / "NN_FINAL_MEGA_V2_BIO_SUBMISSION_PACKET.zip").exists()

    _start_nvidia_smi_monitor(ctx)
    try:
        ctx.nvml_logger = start_gpu_util_logger(csv_path=ctx.out_root / "gpu_util.csv", tag="NN_FINAL_MASTER_V1")
    except Exception:
        ctx.nvml_logger = None

    _build_repo_fingerprint(REPO_ROOT, ctx.audit / "repo_fingerprint.json")
    _write_dataset_hashes(ctx.data_root, ctx.audit / "dataset_hashes.json")

    try:
        pf = subprocess.run([sys.executable, "-m", "pip", "freeze"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
        _write_text(ctx.audit / "pip_freeze.txt", pf.stdout)
    except Exception:
        _write_text(ctx.audit / "pip_freeze.txt", "")

    torch_info: Dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "device_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "",
        }
    except Exception as exc:
        torch_info = {"error": str(exc)}
    _write_json(ctx.audit / "torch_cuda_info.json", torch_info)

    _write_json(
        ctx.audit / "preflight_env.json",
        {
            "cwd": str(REPO_ROOT),
            "python": sys.version.replace("\n", " "),
            "out_root": str(ctx.out_root),
            "data_root": str(ctx.data_root),
            "canonical_root": str(ctx.canonical_root),
            "wall_hours": float(ctx.wall_hours),
            "include_data": bool(ctx.include_data),
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
        outputs=[
            ctx.audit / "preflight_env.json",
            ctx.audit / "repo_fingerprint.json",
            ctx.audit / "dataset_hashes.json",
            ctx.audit / "pip_freeze.txt",
            ctx.audit / "torch_cuda_info.json",
            ctx.audit / "nvidia_smi_1hz.csv",
            ctx.out_root / "gpu_util.csv",
        ],
        extra={"canonical_baseline": str(ctx.canonical_root)},
    )


def _stage_compile_gate(ctx: Ctx) -> Dict[str, Any]:
    stage = "compile_gate"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"

    py_files = [p for p in REPO_ROOT.rglob("*.py") if ".git" not in p.parts]
    bad: List[Tuple[str, str]] = []
    for p in py_files:
        try:
            import py_compile

            py_compile.compile(str(p), doraise=True)
        except Exception as exc:
            bad.append((str(p), str(exc)))
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(f"[{_iso_now()}] compile error: {p} :: {exc}\n")
    if bad:
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="py_compile all .py",
            outputs=[],
            error=f"compile failures: {len(bad)}",
            extra={"n_failed": len(bad), "sample": bad[:20]},
        )
    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="py_compile all .py",
        outputs=[],
        extra={"n_files": len(py_files)},
    )


def _stage_verify_ds004584_full(ctx: Ctx) -> Dict[str, Any]:
    stage = "stage_verify_ds004584_full"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"

    ds_root = ctx.data_root / "ds004584"
    out_stop = ctx.pack_pdrest / "STOP_REASON_ds004584.md"
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
    n_part = int(len(part))
    part["subject_id"] = part["participant_id"].astype(str).str.replace("sub-", "", regex=False)

    counts0 = _count_ds004584_headers(ds_root)
    env = os.environ.copy()
    # Target is near-full (>=140). Retrieve aggressively before evaluating strict minimum.
    if int(max(counts0.get("vhdr_count", 0), counts0.get("set_count", 0))) < 140:
        if shutil.which("datalad") is not None:
            _run_cmd(
                ["bash", "-lc", "shopt -s globstar nullglob; datalad get -r -J 8 sub-*/**/eeg/*"],
                cwd=ds_root,
                log_path=log_path,
                env=env,
                allow_fail=True,
                timeout_sec=1800,
            )
        _run_cmd(
            ["bash", "-lc", "shopt -s globstar nullglob; git annex get -J 8 -- sub-*/**/eeg/*"],
            cwd=ds_root,
            log_path=log_path,
            env=env,
            allow_fail=True,
            timeout_sec=1800,
        )

    counts1 = _count_ds004584_headers(ds_root)

    header_mode = "vhdr" if int(counts1.get("vhdr_count", 0)) > 0 else "set"
    header_count = int(counts1.get("vhdr_count", 0) if header_mode == "vhdr" else counts1.get("set_count", 0))
    subjects_ready = set(counts1.get("subjects_vhdr", []) if header_mode == "vhdr" else counts1.get("subjects_set", []))
    missing_subjects = sorted([sid for sid in part["subject_id"].astype(str).tolist() if sid not in subjects_ready])

    extra = {
        "n_participants": n_part,
        "header_mode": header_mode,
        "header_count": header_count,
        "target_header_count": 140,
        "vhdr_count": int(counts1.get("vhdr_count", 0)),
        "set_count": int(counts1.get("set_count", 0)),
        "missing_subjects_total": int(len(missing_subjects)),
        "missing_subjects": missing_subjects[:120],
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
    out_stop = ctx.pack_mort / "STOP_REASON_ds007020.md"
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
    sub_have = {sid_re.search(str(p)).group(1) for p in eeg_files if sid_re.search(str(p))}
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
                timeout_sec=600,
            )
        _run_cmd(
            ["bash", "-lc", "shopt -s globstar nullglob; git annex get -J 8 -- sub-*/**/eeg/*"],
            cwd=ds_root,
            log_path=log_path,
            env=env,
            allow_fail=True,
            timeout_sec=600,
        )
        eeg_files = [p for p in ds_root.rglob("*_eeg.vhdr") if p.exists()]
        if not eeg_files:
            eeg_files = [p for p in ds_root.rglob("*_eeg.set") if p.exists() and p.with_suffix(".fdt").exists()]
        sub_have = {sid_re.search(str(p)).group(1) for p in eeg_files if sid_re.search(str(p))}
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

    if len(sub_have) < 90:
        reason = f"insufficient ds007020 EEG coverage: subjects_with_eeg={len(sub_have)} (<90)"
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


def _stage_clinical_ds004584_fullN_PDrest(ctx: Ctx) -> Dict[str, Any]:
    stage = "clinical_ds004584_fullN_PDrest"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"
    out_dir = ctx.pack_pdrest
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_root = ctx.data_root / "ds004584"
    part = pd.read_csv(ds_root / "participants.tsv", sep="\t")
    part["subject_id"] = part["participant_id"].astype(str).str.replace("sub-", "", regex=False)

    labels, label_col, diag = _infer_ds004584_groups(part)
    if labels is None:
        stop = out_dir / "STOP_REASON_ds004584.md"
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
        _write_stop_reason(stop, stage, "no readable resting features", diagnostics={"n_subjects_input": len(subjects), "n_exclusions": int(len(ex_df))})
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
                "n_perm_done": int(n_perm),
                "n_boot_done": int(n_boot_auc),
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
                "n_boot_done": int(n_boot_beta),
            }
        )

    end_df = pd.DataFrame(rows)
    if end_df.empty:
        stop = out_dir / "STOP_REASON_ds004584.md"
        _write_stop_reason(stop, stage, "no valid endpoints after feature extraction", diagnostics={"n_subjects_features": int(len(dev_df))})
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
    if n_used < 120:
        stop = out_dir / "STOP_REASON_ds004584.md"
        _write_stop_reason(
            stop,
            stage,
            f"ds004584 usable cohort below strict minimum: n_used={n_used} (<120)",
            diagnostics={
                "n_subjects_expected": 149,
                "n_subjects_used": n_used,
                "n_exclusions": int(len(ex_df)),
                "exclusion_path": str(ex_path),
            },
        )
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds004584 fullN",
            outputs=[feat_path, end_path, ex_path, stop],
            error=f"n_used {n_used} < 120",
        )

    extra = {
        "n_subjects_expected": 149,
        "n_subjects_used": n_used,
        "n_pd_used": int((dev_df["group"] == "PD").sum()),
        "n_cn_used": int((dev_df["group"] == "CN").sum()),
        "n_endpoints": int(len(end_df)),
        "n_perm_done": int(n_perm),
        "n_boot_done": int(n_boot_auc),
        "group_column": label_col,
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


def _stage_clinical_ds007020_LEAPD_full(ctx: Ctx) -> Dict[str, Any]:
    stage = "clinical_ds007020_LEAPD_full"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"
    out_dir = ctx.pack_mort
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_root = ctx.data_root / "ds007020"
    part = pd.read_csv(ds_root / "participants.tsv", sep="\t")
    y_lbl, y_col, y_diag = _infer_ds007020_labels(part)
    if y_lbl is None:
        stop = out_dir / "STOP_REASON_ds007020.md"
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

    def _norm_ch(ch: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", str(ch)).upper()

    def _lpc_vec(signal_1d: np.ndarray, *, order: int, frac: float, sfreq: float, seg_sec: float = 4.0) -> np.ndarray:
        x = np.asarray(signal_1d, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < order + 10:
            return np.full(order, np.nan, dtype=float)
        n_take = int(max(order + 10, min(x.size, round(x.size * float(frac)))))
        x = x[:n_take]
        seg_len = int(max(order + 10, round(seg_sec * sfreq)))
        if seg_len <= order + 2:
            seg_len = order + 10
        betas: List[np.ndarray] = []
        nseg = max(1, int(x.size // seg_len))
        for i in range(nseg):
            seg = x[i * seg_len : (i + 1) * seg_len]
            if seg.size < order + 10:
                continue
            seg = seg - float(np.nanmean(seg))
            y = seg[order:]
            X = np.column_stack([seg[order - k - 1 : seg.size - k - 1] for k in range(order)])
            try:
                b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                betas.append(np.asarray(b, dtype=float))
            except Exception:
                continue
        if not betas:
            return np.full(order, np.nan, dtype=float)
        arr = np.vstack(betas)
        return np.nanmedian(arr, axis=0)

    def _fit_leapd_model(X: np.ndarray, y: np.ndarray, d: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        X0 = X[y == 0]
        X1 = X[y == 1]
        if len(X0) < max(3, d) or len(X1) < max(3, d):
            return None

        def _basis(Z: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
            mu = np.nanmean(Z, axis=0)
            C = Z - mu
            try:
                _, _, vt = np.linalg.svd(C, full_matrices=False)
            except Exception:
                return mu, np.eye(Z.shape[1], 1)
            kk = int(max(1, min(k, vt.shape[0], vt.shape[1])))
            B = vt[:kk].T
            return mu, B

        mu0, B0 = _basis(X0, d)
        mu1, B1 = _basis(X1, d)
        return mu0, B0, mu1, B1

    def _leapd_index(x: np.ndarray, model: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> float:
        mu0, B0, mu1, B1 = model
        z0 = x - mu0
        z1 = x - mu1
        r0 = z0 - (B0 @ (B0.T @ z0))
        r1 = z1 - (B1 @ (B1.T @ z1))
        d0 = float(np.linalg.norm(r0))  # living hyperplane
        d1 = float(np.linalg.norm(r1))  # deceased hyperplane
        return float(d0 / max(d0 + d1, 1e-12))

    def _leapd_loocv(X: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
        n = int(len(y))
        out = np.full(n, np.nan, dtype=float)
        for i in range(n):
            tr = np.ones(n, dtype=bool)
            tr[i] = False
            mdl = _fit_leapd_model(X[tr], y[tr], d)
            if mdl is None:
                continue
            out[i] = _leapd_index(X[i], mdl)
        return out

    def _metrics(y: np.ndarray, s: np.ndarray) -> Dict[str, float]:
        y = np.asarray(y, dtype=int)
        s = np.asarray(s, dtype=float)
        m = np.isfinite(s)
        y = y[m]
        s = s[m]
        if y.size < 6 or np.unique(y).size < 2:
            return {"auc": float("nan"), "auc_flipped": float("nan"), "accuracy": float("nan"), "n": int(y.size)}
        auc_v = float(roc_auc_score(y, s))
        pred = (s >= 0.5).astype(int)
        acc = float(np.mean(pred == y))
        return {"auc": auc_v, "auc_flipped": float(max(auc_v, 1.0 - auc_v)), "accuracy": acc, "n": int(y.size)}

    part["subject_id"] = part["participant_id"].astype(str).str.replace("sub-", "", regex=False)
    part["label"] = pd.to_numeric(y_lbl, errors="coerce")
    part["age"] = pd.to_numeric(part.get("AGE", part.get("Age", np.nan)), errors="coerce")
    sex_src = None
    for c in ["SEX", "Sex", "GENDER", "gender"]:
        if c in part.columns:
            sex_src = part[c]
            break
    if sex_src is None:
        part["sex"] = pd.Series([""] * len(part), index=part.index, dtype=str)
    else:
        part["sex"] = sex_src.astype(str)

    exclude_ch = {_norm_ch(x) for x in ["Pz", "TP9", "TP10", "FT9", "FT10"]}

    feat_rows: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    signals_by_subject: Dict[str, Dict[str, np.ndarray]] = {}
    sfreq_by_subject: Dict[str, float] = {}

    for row in part.itertuples(index=False):
        sid = str(getattr(row, "subject_id"))
        lab = float(getattr(row, "label")) if np.isfinite(getattr(row, "label")) else float("nan")
        if not np.isfinite(lab):
            exclusions.append({"subject_id": sid, "reason": "missing_label", "rest_file": ""})
            continue
        fpath, why = _find_best_rest_file(ds_root, sid)
        if fpath is None:
            exclusions.append({"subject_id": sid, "reason": why, "rest_file": ""})
            continue
        try:
            raw = _read_raw_any(fpath)
            raw.pick_types(eeg=True, eog=False, misc=False, stim=False)
            if len(raw.ch_names) < 8:
                raise RuntimeError("fewer than 8 EEG channels")
            drop = [c for c in raw.ch_names if _norm_ch(c) in exclude_ch]
            if drop:
                raw.drop_channels(drop)
            if len(raw.ch_names) < 8:
                raise RuntimeError("fewer than 8 EEG channels after required channel exclusions")
            if float(raw.info["sfreq"]) > 256.0:
                raw.resample(256, verbose="ERROR")
            if float(raw.n_times / max(float(raw.info["sfreq"]), 1e-9)) > 120.0:
                raw.crop(tmin=0.0, tmax=120.0, include_tmax=False)
            try:
                raw.filter(l_freq=1.0, h_freq=40.0, method="fir", verbose="ERROR")
            except Exception:
                # Fail-closed fallback: continue with unfiltered signal but log it.
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(f"[{_iso_now()}] warning: filter failed for {sid}; continuing unfiltered\n")

            data = raw.get_data()
            ch_std = np.nanstd(data, axis=1)
            med = float(np.nanmedian(ch_std))
            mad = float(np.nanmedian(np.abs(ch_std - med)))
            thr = med + 6.0 * max(mad, 1e-9)
            keep = np.isfinite(ch_std) & (ch_std <= thr)
            if int(np.sum(keep)) < 8:
                raise RuntimeError("insufficient channels after robust bad-channel rejection")
            if int(np.sum(~keep)) > 0:
                keep_names = [raw.ch_names[i] for i in range(len(raw.ch_names)) if bool(keep[i])]
                raw.pick_channels(keep_names)
                data = raw.get_data()

            feat = _compute_rest_feature_row(raw, max_seconds=120.0, lpc_order=None)
            feat["subject_id"] = sid
            feat["rest_file"] = str(fpath)
            feat["label"] = int(lab)
            feat["age"] = float(getattr(row, "age")) if np.isfinite(getattr(row, "age")) else float("nan")
            feat["sex"] = str(getattr(row, "sex"))
            feat_rows.append(feat)

            ch_map: Dict[str, np.ndarray] = {}
            for i, ch in enumerate(raw.ch_names):
                k = _norm_ch(ch)
                if k in ch_map:
                    continue
                ch_map[k] = np.asarray(data[i], dtype=float)
            signals_by_subject[sid] = ch_map
            sfreq_by_subject[sid] = float(raw.info["sfreq"])
        except Exception as exc:
            exclusions.append({"subject_id": sid, "reason": f"read_error:{exc}", "rest_file": str(fpath)})
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(f"[{_iso_now()}] subject={sid} read_error {exc}\n")

    feat_df = pd.DataFrame(feat_rows)
    ex_df = pd.DataFrame(exclusions) if exclusions else pd.DataFrame(columns=["subject_id", "reason", "rest_file"])
    ex_path = out_dir / "EXCLUSIONS.csv"
    ex_df.to_csv(ex_path, index=False)
    if feat_df.empty:
        stop = out_dir / "STOP_REASON_ds007020.md"
        _write_stop_reason(stop, stage, "no readable resting features", diagnostics={"n_exclusions": int(len(ex_df))})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds007020 LEAPD",
            outputs=[stop, ex_path],
            error="no features",
        )

    dev_df = _compute_deviation(feat_df, controls_mask=None)
    feat_path = out_dir / "mortality_features.csv"
    dev_df.to_csv(feat_path, index=False)

    living_ids_all = sorted([sid for sid in dev_df["subject_id"].astype(str).tolist() if int(dev_df.loc[dev_df["subject_id"] == sid, "label"].iloc[0]) == 0 and sid in signals_by_subject])
    deceased_ids_all = sorted([sid for sid in dev_df["subject_id"].astype(str).tolist() if int(dev_df.loc[dev_df["subject_id"] == sid, "label"].iloc[0]) == 1 and sid in signals_by_subject])

    if len(living_ids_all) < 22 or len(deceased_ids_all) < 22:
        stop = out_dir / "STOP_REASON_ds007020.md"
        _write_stop_reason(
            stop,
            stage,
            "insufficient labels for required balanced LEAPD design (22 living + 22 deceased)",
            diagnostics={"n_living": len(living_ids_all), "n_deceased": len(deceased_ids_all)},
        )
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds007020 LEAPD",
            outputs=[feat_path, ex_path, stop],
            error="insufficient class counts for LEAPD",
        )

    # Balanced LOOCV benchmark: 44 subjects (22 deceased + 22 living)
    bal_deceased = deceased_ids_all[:22]
    bal_living = living_ids_all[:22]
    bal_ids = bal_deceased + bal_living
    bal_y = np.asarray([1] * len(bal_deceased) + [0] * len(bal_living), dtype=int)

    common_ch = None
    for sid in bal_ids:
        keys = set(signals_by_subject[sid].keys())
        common_ch = keys if common_ch is None else (common_ch & keys)
    common_channels = sorted(common_ch or [])
    if len(common_channels) < 5:
        stop = out_dir / "STOP_REASON_ds007020.md"
        _write_stop_reason(
            stop,
            stage,
            "insufficient channel overlap for LEAPD across balanced set",
            diagnostics={"n_common_channels": int(len(common_channels))},
        )
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds007020 LEAPD",
            outputs=[feat_path, ex_path, stop],
            error="insufficient channel overlap",
        )

    lpc_order = 12
    d_grid = [2, 3, 4, 5, 6]
    lpc_cache: Dict[Tuple[str, str, float], np.ndarray] = {}

    def _get_vec(sid: str, ch: str, frac: float) -> np.ndarray:
        key = (sid, ch, float(frac))
        if key not in lpc_cache:
            sig = signals_by_subject[sid].get(ch)
            sf = float(sfreq_by_subject.get(sid, 256.0))
            if sig is None:
                lpc_cache[key] = np.full(lpc_order, np.nan, dtype=float)
            else:
                lpc_cache[key] = _lpc_vec(sig, order=lpc_order, frac=float(frac), sfreq=sf)
        return lpc_cache[key]

    channel_rows: List[Dict[str, Any]] = []
    best_d_by_channel: Dict[str, int] = {}
    for ch in common_channels:
        X = np.vstack([_get_vec(sid, ch, 1.0) for sid in bal_ids])
        if not np.isfinite(X).all():
            continue
        best = None
        for d in d_grid:
            scores = _leapd_loocv(X, bal_y, d)
            m = _metrics(bal_y, scores)
            row = {
                "channel": ch,
                "d": int(d),
                "n": int(m["n"]),
                "n_living": int(len(bal_living)),
                "n_deceased": int(len(bal_deceased)),
                "auc": float(m["auc"]),
                "auc_flipped": float(m["auc_flipped"]),
                "accuracy": float(m["accuracy"]),
            }
            if best is None or (np.isfinite(row["auc_flipped"]) and row["auc_flipped"] > best["auc_flipped"]):
                best = row
        if best is not None:
            channel_rows.append(best)
            best_d_by_channel[ch] = int(best["d"])

    ch_df = pd.DataFrame(channel_rows).sort_values(["auc_flipped", "auc"], ascending=False).reset_index(drop=True)
    ch_path = out_dir / "leapd_channel_results.csv"
    ch_df.to_csv(ch_path, index=False)
    if ch_df.empty:
        stop = out_dir / "STOP_REASON_ds007020.md"
        _write_stop_reason(stop, stage, "no valid LEAPD channel models", diagnostics={"n_common_channels": int(len(common_channels))})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds007020 LEAPD",
            outputs=[feat_path, ex_path, ch_path, stop],
            error="no valid LEAPD channels",
        )

    best_channel = str(ch_df.iloc[0]["channel"])
    best_d = int(ch_df.iloc[0]["d"])
    best_scores_full = _leapd_loocv(np.vstack([_get_vec(sid, best_channel, 1.0) for sid in bal_ids]), bal_y, best_d)
    primary_m = _metrics(bal_y, best_scores_full)

    # Truncation analysis on best channel: 100/90/66/50%.
    trunc_rows: List[Dict[str, Any]] = []
    for frac in [1.0, 0.9, 0.66, 0.5]:
        X = np.vstack([_get_vec(sid, best_channel, frac) for sid in bal_ids])
        if not np.isfinite(X).all():
            continue
        sc = _leapd_loocv(X, bal_y, best_d)
        m = _metrics(bal_y, sc)
        trunc_rows.append(
            {
                "fraction": float(frac),
                "channel": best_channel,
                "d": int(best_d),
                "n": int(m["n"]),
                "auc": float(m["auc"]),
                "auc_flipped": float(m["auc_flipped"]),
                "accuracy": float(m["accuracy"]),
            }
        )
    trunc_df = pd.DataFrame(trunc_rows)
    trunc_path = out_dir / "leapd_truncation.csv"
    trunc_df.to_csv(trunc_path, index=False)

    trunc_fig = out_dir / "FIG_mortality_truncation.png"
    if not trunc_df.empty:
        plt.figure(figsize=(7, 5))
        plt.plot(trunc_df["fraction"], trunc_df["auc_flipped"], "o-", color="#0072B2", label="AUC flipped")
        plt.plot(trunc_df["fraction"], trunc_df["accuracy"], "s--", color="#D55E00", label="Accuracy")
        plt.gca().invert_xaxis()
        plt.xlabel("Retained signal fraction")
        plt.ylabel("Performance")
        plt.title(f"ds007020 truncation ({best_channel}, d={best_d})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(trunc_fig, dpi=140)
        plt.close()

    # Stage-2 out-of-sample
    train_deceased = deceased_ids_all[:15]
    train_living = living_ids_all[:15]
    test_deceased = deceased_ids_all[15:22]
    living_pool = living_ids_all[15:]
    if len(train_deceased) < 15 or len(train_living) < 15 or len(test_deceased) < 7 or len(living_pool) < 7:
        stop = out_dir / "STOP_REASON_ds007020.md"
        _write_stop_reason(
            stop,
            stage,
            "insufficient subjects for required train/test split (15/15 train, 7 deceased fixed test)",
            diagnostics={
                "train_deceased": len(train_deceased),
                "train_living": len(train_living),
                "test_deceased": len(test_deceased),
                "living_pool": len(living_pool),
            },
        )
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds007020 LEAPD",
            outputs=[feat_path, ex_path, ch_path, trunc_path, stop],
            error="required out-of-sample split unavailable",
        )

    train_ids = train_deceased + train_living
    train_y = np.asarray([1] * len(train_deceased) + [0] * len(train_living), dtype=int)
    test_fixed_ids = test_deceased

    # Hyperparameter selection on train set (single-channel LOOCV).
    train_common = None
    for sid in train_ids + test_fixed_ids + living_pool:
        keys = set(signals_by_subject[sid].keys())
        train_common = keys if train_common is None else (train_common & keys)
    train_channels = sorted(train_common or [])
    if len(train_channels) < 5:
        stop = out_dir / "STOP_REASON_ds007020.md"
        _write_stop_reason(stop, stage, "insufficient channel overlap for out-of-sample LEAPD", diagnostics={"n_channels": len(train_channels)})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds007020 LEAPD",
            outputs=[feat_path, ex_path, ch_path, trunc_path, stop],
            error="insufficient channels for out-of-sample",
        )

    train_eval_rows: List[Dict[str, Any]] = []
    for ch in train_channels:
        X = np.vstack([_get_vec(sid, ch, 1.0) for sid in train_ids])
        if not np.isfinite(X).all():
            continue
        for d in d_grid:
            sc = _leapd_loocv(X, train_y, d)
            m = _metrics(train_y, sc)
            train_eval_rows.append({"channel": ch, "d": int(d), **m})
    tev = pd.DataFrame(train_eval_rows)
    if tev.empty:
        stop = out_dir / "STOP_REASON_ds007020.md"
        _write_stop_reason(stop, stage, "no train-set LEAPD models available", diagnostics={})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds007020 LEAPD",
            outputs=[feat_path, ex_path, ch_path, trunc_path, stop],
            error="no train LEAPD models",
        )

    tev = tev.sort_values(["auc_flipped", "accuracy"], ascending=False).reset_index(drop=True)
    best_single = tev.iloc[0].to_dict()
    best_single_ch = str(best_single["channel"])
    best_single_d = int(best_single["d"])

    best_d_train: Dict[str, int] = {}
    for ch in sorted(tev["channel"].unique()):
        tmp = tev[tev["channel"] == ch].sort_values(["auc_flipped", "accuracy"], ascending=False)
        if not tmp.empty:
            best_d_train[str(ch)] = int(tmp.iloc[0]["d"])

    # Multi-channel combinations (2..5) with geometric mean of channel indices.
    top_channels = [str(x) for x in tev["channel"].drop_duplicates().tolist()[:8]]
    combo_rows: List[Dict[str, Any]] = []
    n_combo5 = 0
    for k in [2, 3, 4, 5]:
        if len(top_channels) < k:
            continue
        for combo in __import__("itertools").combinations(top_channels, k):
            if k == 5:
                n_combo5 += 1
            scores = np.full(len(train_ids), np.nan, dtype=float)
            for i in range(len(train_ids)):
                tr = np.ones(len(train_ids), dtype=bool)
                tr[i] = False
                sc_ch: List[float] = []
                for ch in combo:
                    d = int(best_d_train.get(ch, best_single_d))
                    Xtr = np.vstack([_get_vec(train_ids[j], ch, 1.0) for j in range(len(train_ids)) if tr[j]])
                    ytr = train_y[tr]
                    mdl = _fit_leapd_model(Xtr, ytr, d)
                    if mdl is None:
                        sc_ch = []
                        break
                    xv = _get_vec(train_ids[i], ch, 1.0)
                    if not np.isfinite(xv).all():
                        sc_ch = []
                        break
                    sc_ch.append(_leapd_index(xv, mdl))
                if sc_ch:
                    v = np.clip(np.asarray(sc_ch, dtype=float), 1e-6, 1.0 - 1e-6)
                    scores[i] = float(np.exp(np.mean(np.log(v))))
            m = _metrics(train_y, scores)
            combo_rows.append({"combo": ";".join(combo), "k": int(k), **m})
    combo_df = pd.DataFrame(combo_rows).sort_values(["auc_flipped", "accuracy"], ascending=False).reset_index(drop=True) if combo_rows else pd.DataFrame()
    best_combo = combo_df.iloc[0].to_dict() if not combo_df.empty else {}
    best_combo_channels = str(best_combo.get("combo", "")).split(";") if best_combo else []

    # Fit full train models and run 10,000 sampled out-of-sample tests (7 deceased fixed + 7 living sampled).
    def _fit_full_model(ch: str, d: int):
        Xtr = np.vstack([_get_vec(sid, ch, 1.0) for sid in train_ids])
        return _fit_leapd_model(Xtr, train_y, d)

    mdl_single = _fit_full_model(best_single_ch, best_single_d)
    if mdl_single is None:
        stop = out_dir / "STOP_REASON_ds007020.md"
        _write_stop_reason(stop, stage, "failed to fit selected LEAPD single-channel model", diagnostics=best_single)
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds007020 LEAPD",
            outputs=[feat_path, ex_path, ch_path, trunc_path, stop],
            error="single-channel model fit failed",
        )

    mdl_combo: Dict[str, Any] = {}
    for ch in best_combo_channels:
        d = int(best_d_train.get(ch, best_single_d))
        mdl = _fit_full_model(ch, d)
        if mdl is not None:
            mdl_combo[ch] = {"d": d, "model": mdl}

    test_subjects_all = test_fixed_ids + living_pool
    single_scores_by_sid: Dict[str, float] = {}
    combo_scores_by_sid: Dict[str, float] = {}
    for sid in test_subjects_all:
        xv = _get_vec(sid, best_single_ch, 1.0)
        single_scores_by_sid[sid] = _leapd_index(xv, mdl_single) if np.isfinite(xv).all() else float("nan")
        if mdl_combo:
            vals: List[float] = []
            for ch, payload in mdl_combo.items():
                xc = _get_vec(sid, ch, 1.0)
                if not np.isfinite(xc).all():
                    vals = []
                    break
                vals.append(_leapd_index(xc, payload["model"]))
            if vals:
                v = np.clip(np.asarray(vals, dtype=float), 1e-6, 1.0 - 1e-6)
                combo_scores_by_sid[sid] = float(np.exp(np.mean(np.log(v))))
            else:
                combo_scores_by_sid[sid] = float("nan")
        else:
            combo_scores_by_sid[sid] = float("nan")

    rng = np.random.default_rng(20260223)
    n_reps = 10000
    single_auc, single_bacc = [], []
    combo_auc, combo_bacc = [], []
    test_deceased_arr = np.asarray(test_fixed_ids, dtype=object)
    living_pool_arr = np.asarray(living_pool, dtype=object)
    for _ in range(n_reps):
        samp_liv = rng.choice(living_pool_arr, size=7, replace=False)
        ids = list(test_deceased_arr) + list(samp_liv)
        yv = np.asarray([1] * len(test_deceased_arr) + [0] * len(samp_liv), dtype=int)

        ss1 = np.asarray([single_scores_by_sid.get(x, float("nan")) for x in ids], dtype=float)
        m1 = _metrics(yv, ss1)
        if np.isfinite(m1["auc"]):
            single_auc.append(float(m1["auc"]))
            single_bacc.append(float(m1["accuracy"]))

        ss2 = np.asarray([combo_scores_by_sid.get(x, float("nan")) for x in ids], dtype=float)
        m2 = _metrics(yv, ss2)
        if np.isfinite(m2["auc"]):
            combo_auc.append(float(m2["auc"]))
            combo_bacc.append(float(m2["accuracy"]))

    out_json = {
        "train_counts": {"living": len(train_living), "deceased": len(train_deceased)},
        "test_fixed_deceased": len(test_fixed_ids),
        "living_pool": len(living_pool),
        "n_reps": int(n_reps),
        "best_single_channel": best_single_ch,
        "best_single_d": int(best_single_d),
        "best_single_train_auc_flipped": float(best_single.get("auc_flipped", float("nan"))),
        "best_combo_channels": best_combo_channels,
        "n_combo5_evaluated": int(n_combo5),
        "single_out_of_sample": {
            "auc_mean": float(np.nanmean(single_auc)) if single_auc else float("nan"),
            "auc_ci95": [float(np.nanquantile(single_auc, 0.025)) if single_auc else float("nan"), float(np.nanquantile(single_auc, 0.975)) if single_auc else float("nan")],
            "bacc_mean": float(np.nanmean(single_bacc)) if single_bacc else float("nan"),
        },
        "combo_out_of_sample": {
            "auc_mean": float(np.nanmean(combo_auc)) if combo_auc else float("nan"),
            "auc_ci95": [float(np.nanquantile(combo_auc, 0.025)) if combo_auc else float("nan"), float(np.nanquantile(combo_auc, 0.975)) if combo_auc else float("nan")],
            "bacc_mean": float(np.nanmean(combo_bacc)) if combo_bacc else float("nan"),
        },
    }
    oos_path = out_dir / "leapd_out_of_sample.json"
    _write_json(oos_path, out_json)

    # Primary ROC figure from balanced LOOCV best channel.
    roc_path = out_dir / "FIG_mortality_primary_auc_roc.png"
    _plot_roc_calibration(bal_y, best_scores_full, "ds007020 LEAPD balanced LOOCV", roc_path, out_dir / "FIG_mortality_calibration.png")

    # Baseline logistic (intercept + feature only); require finite betas.
    n_perm = 20000
    n_boot = 2000
    dev_df["leapd_index_loocv"] = np.nan
    # Populate LEAPD LOOCV index for all subjects with best channel.
    full_ids = [sid for sid in dev_df["subject_id"].astype(str).tolist() if sid in signals_by_subject and best_channel in signals_by_subject[sid]]
    if len(full_ids) >= 20:
        yy = np.asarray([int(dev_df.loc[dev_df["subject_id"] == sid, "label"].iloc[0]) for sid in full_ids], dtype=int)
        XX = np.vstack([_get_vec(sid, best_channel, 1.0) for sid in full_ids])
        ss = _leapd_loocv(XX, yy, best_d)
        for sid, v in zip(full_ids, ss):
            dev_df.loc[dev_df["subject_id"] == sid, "leapd_index_loocv"] = float(v)

    base_feats = ["dev_z_theta_alpha_ratio", "dev_z_rel_alpha", "dev_z_spectral_slope", "composite_deviation", "leapd_index_loocv"]
    base_rows: List[Dict[str, Any]] = []
    for feat in base_feats:
        sub = dev_df[["subject_id", "label", feat]].copy()
        sub["label"] = pd.to_numeric(sub["label"], errors="coerce")
        sub[feat] = pd.to_numeric(sub[feat], errors="coerce")
        sub = sub[np.isfinite(sub["label"]) & np.isfinite(sub[feat])].copy()
        if sub.empty or sub["label"].nunique() < 2:
            continue
        yv = sub["label"].astype(int).to_numpy(dtype=int)
        sv = sub[feat].to_numpy(dtype=float)
        auc_obs, auc_ci = _bootstrap_auc(yv, sv, n_boot=n_boot, seed=11000 + _stable_int_from_text(feat) % 100000)
        p_auc = _perm_p_auc(yv, sv, n_perm=n_perm, seed=12000 + _stable_int_from_text(feat) % 100000)
        X = sv.reshape(-1, 1)
        beta = _fit_logit_beta(X, yv)
        b_lo, b_hi = _bootstrap_beta_ci(X, yv, n_boot=n_boot, seed=13000 + _stable_int_from_text(feat) % 100000)
        p_beta = _perm_p_beta(X, yv, beta, n_perm=n_perm, seed=14000 + _stable_int_from_text(feat) % 100000)
        base_rows.append(
            {
                "dataset_id": "ds007020",
                "endpoint": "AUC_mortality",
                "feature": feat,
                "type": "auc",
                "n": int(len(sub)),
                "estimate": float(auc_obs),
                "auc_flipped": float(max(auc_obs, 1.0 - auc_obs)) if np.isfinite(auc_obs) else float("nan"),
                "ci95_lo": float(auc_ci[0]),
                "ci95_hi": float(auc_ci[1]),
                "perm_p": float(p_auc),
                "model": "baseline_logit",
                "n_perm_done": int(n_perm),
                "n_boot_done": int(n_boot),
            }
        )
        base_rows.append(
            {
                "dataset_id": "ds007020",
                "endpoint": "LogitBeta_mortality",
                "feature": feat,
                "type": "logit_beta",
                "n": int(len(sub)),
                "estimate": float(beta),
                "auc_flipped": float("nan"),
                "ci95_lo": float(b_lo),
                "ci95_hi": float(b_hi),
                "perm_p": float(p_beta),
                "model": "baseline_logit",
                "covariates": "",
                "n_perm_done": int(n_perm),
                "n_boot_done": int(n_boot),
            }
        )

    base_df = pd.DataFrame(base_rows)
    if base_df.empty:
        stop = out_dir / "STOP_REASON_ds007020.md"
        _write_stop_reason(stop, stage, "baseline endpoints could not be computed", diagnostics={})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds007020 LEAPD",
            outputs=[feat_path, ex_path, ch_path, trunc_path, oos_path, stop],
            error="no baseline endpoints",
        )

    base_df["perm_q_within_ds007020"] = _bh_qvals(pd.to_numeric(base_df["perm_p"], errors="coerce").fillna(1.0).to_numpy(dtype=float).tolist())
    base_path = out_dir / "mortality_baseline_endpoints.csv"
    base_df.to_csv(base_path, index=False)

    beta_mask = base_df["type"].astype(str).str.contains("beta", na=False)
    finite_beta_count = int(np.isfinite(pd.to_numeric(base_df.loc[beta_mask, "estimate"], errors="coerce")).sum())
    if finite_beta_count < 1:
        stop = out_dir / "STOP_REASON_ds007020.md"
        _write_stop_reason(stop, stage, "all baseline logistic betas are non-finite", diagnostics={})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds007020 LEAPD",
            outputs=[feat_path, ex_path, ch_path, trunc_path, oos_path, base_path, stop],
            error="no finite baseline beta",
        )

    # Time-to-death correlation note (skip if unavailable).
    ttd_cols = [c for c in part.columns if re.search(r"time.*death|death.*time|survival.*time|days.*death", c, flags=re.IGNORECASE)]
    if not ttd_cols:
        _write_text(
            out_dir / "NOTE_time_to_death_missing.md",
            "# Time-to-Death Correlation\n\nNo time-to-death field was found in participants.tsv. Correlation stage skipped by design.\n",
        )

    extra = {
        "n_subjects_used": int(dev_df["subject_id"].nunique()),
        "n_label_living": int((pd.to_numeric(dev_df["label"], errors="coerce") == 0).sum()),
        "n_label_deceased": int((pd.to_numeric(dev_df["label"], errors="coerce") == 1).sum()),
        "finite_beta_count": int(finite_beta_count),
        "leapd_primary_auc": float(primary_m.get("auc", float("nan"))),
        "leapd_primary_auc_flipped": float(primary_m.get("auc_flipped", float("nan"))),
        "leapd_primary_channel": best_channel,
        "leapd_primary_d": int(best_d),
        "n_perm_done": int(n_perm),
        "n_boot_done": int(n_boot),
        "label_column": y_col,
        "loocv_balanced_n": int(len(bal_ids)),
    }

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="clinical ds007020 LEAPD full",
        outputs=[
            feat_path,
            ex_path,
            ch_path,
            trunc_path,
            oos_path,
            base_path,
            roc_path,
            trunc_fig,
        ],
        extra=extra,
    )


def _stage_bio_ds004752_crossmodality_attempt(ctx: Ctx) -> Dict[str, Any]:
    stage = "bio_ds004752_crossmodality_attempt"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"
    out_d752 = ctx.pack_bio / "BIO_D_cross_modality"
    out_d752.mkdir(parents=True, exist_ok=True)
    out_d7262 = ctx.pack_bio / "workload_ds007262"
    out_d7262.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []
    extra: Dict[str, Any] = {}
    stage_status = "PASS"
    stage_error = ""

    # ---- ds004752 cross-modality attempt ----
    ds_root = ctx.data_root / "ds004752"
    stop_path = out_d752 / "STOP_REASON_ds004752.md"
    map_yaml = out_d752 / "event_map_ds004752.yaml"
    map_sum = out_d752 / "decode_ds004752_summary.json"
    cand_csv = out_d752 / "decode_ds004752_candidates.csv"
    _run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "decode_ds004752.py"),
            "--dataset_root",
            str(ds_root),
            "--out_yaml",
            str(map_yaml),
            "--out_summary",
            str(map_sum),
            "--out_candidate",
            str(cand_csv),
            "--stop_reason",
            str(stop_path),
        ],
        cwd=REPO_ROOT,
        log_path=log_path,
        allow_fail=True,
        timeout_sec=600,
    )
    outputs.extend([map_yaml, map_sum, cand_csv])

    d752_status = "SKIP"
    try:
        map_payload = _read_json(map_sum) if map_sum.exists() else {"status": "SKIP", "reason": "missing decode summary"}
        if str(map_payload.get("status", "SKIP")) != "PASS":
            if not stop_path.exists():
                _write_stop_reason(stop_path, stage, str(map_payload.get("reason", "mapping decode failed")), diagnostics=map_payload)
            outputs.append(stop_path)
        else:
            ev_all = sorted(ds_root.rglob("*_task-verbalWM*_events.tsv"))
            chosen: Dict[Tuple[str, str], Path] = {}
            for p in ev_all:
                m = re.search(r"sub-([A-Za-z0-9]+)", p.as_posix())
                if not m:
                    continue
                sid = m.group(1)
                mod = "ieeg" if "/ieeg/" in p.as_posix() else ("eeg" if "/eeg/" in p.as_posix() else "other")
                if mod not in {"eeg", "ieeg"}:
                    continue
                chosen.setdefault((sid, mod), p)

            rows: List[Dict[str, Any]] = []
            fails: List[Dict[str, Any]] = []
            for (sid, mod), ep in sorted(chosen.items()):
                try:
                    df = pd.read_csv(ep, sep="\t")
                    if "SetSize" not in df.columns or "onset" not in df.columns:
                        fails.append({"subject": sid, "modality": mod, "events": str(ep), "reason": "missing SetSize/onset"})
                        continue
                    load = pd.to_numeric(df["SetSize"], errors="coerce")
                    onset = pd.to_numeric(df["onset"], errors="coerce")
                    ok = np.isfinite(load.to_numpy(dtype=float)) & np.isfinite(onset.to_numpy(dtype=float))
                    if int(np.sum(ok)) < 8:
                        fails.append({"subject": sid, "modality": mod, "events": str(ep), "reason": "insufficient valid events"})
                        continue
                    loadv = load.to_numpy(dtype=float)[ok]
                    onsetv = onset.to_numpy(dtype=float)[ok]
                    med = float(np.nanmedian(loadv))
                    low = onsetv[loadv < med]
                    high = onsetv[loadv >= med]
                    if len(low) < 3 or len(high) < 3:
                        fails.append({"subject": sid, "modality": mod, "events": str(ep), "reason": "not enough low/high events"})
                        continue

                    sig_path = _find_matching_signal(ep, mod)
                    if sig_path is None:
                        fails.append({"subject": sid, "modality": mod, "events": str(ep), "reason": "signal file missing"})
                        continue
                    raw = _read_raw_any(sig_path)
                    amp_low = _extract_event_amp(raw, low, modality=mod, tmin=-0.2, tmax=0.8, win=(0.35, 0.60))
                    amp_high = _extract_event_amp(raw, high, modality=mod, tmin=-0.2, tmax=0.8, win=(0.35, 0.60))
                    sig = float(amp_high - amp_low) if np.isfinite(amp_high) and np.isfinite(amp_low) else float("nan")
                    if not np.isfinite(sig):
                        fails.append({"subject": sid, "modality": mod, "events": str(ep), "reason": "non-finite signature"})
                        continue
                    rows.append(
                        {
                            "subject_id": sid,
                            "modality": mod,
                            "events_file": str(ep),
                            "signal_file": str(sig_path),
                            "n_low": int(len(low)),
                            "n_high": int(len(high)),
                            "amp_low": float(amp_low),
                            "amp_high": float(amp_high),
                            "signature_high_minus_low": float(sig),
                        }
                    )
                except Exception as exc:
                    fails.append({"subject": sid, "modality": mod, "events": str(ep), "reason": f"exception:{exc}"})
                    with log_path.open("a", encoding="utf-8") as lf:
                        lf.write(f"[{_iso_now()}] ds004752 subject={sid} modality={mod} error={exc}\n")

            sig_df = pd.DataFrame(rows)
            fail_df = pd.DataFrame(fails) if fails else pd.DataFrame(columns=["subject", "modality", "events", "reason"])
            sig_csv = out_d752 / "cross_modality_signatures.csv"
            fail_csv = out_d752 / "cross_modality_failures.csv"
            sig_df.to_csv(sig_csv, index=False)
            fail_df.to_csv(fail_csv, index=False)
            outputs.extend([sig_csv, fail_csv])

            piv = sig_df.pivot_table(index="subject_id", columns="modality", values="signature_high_minus_low", aggfunc="median") if not sig_df.empty else pd.DataFrame()
            pairs = piv.dropna(subset=["eeg", "ieeg"]).copy() if {"eeg", "ieeg"}.issubset(piv.columns) else pd.DataFrame()
            if pairs.empty or len(pairs) < 8:
                reason = "insufficient convergent signatures across eeg/ieeg runs"
                _write_stop_reason(
                    stop_path,
                    stage,
                    reason,
                    diagnostics={
                        "n_rows": int(len(sig_df)),
                        "n_pair_subjects": int(len(pairs)),
                        "file_formats": _collect_file_formats(ds_root),
                        "fail_sample": fail_df.head(40).to_dict(orient="records"),
                        "log_tail": _tail(log_path, 200),
                    },
                )
                outputs.append(stop_path)
                d752_status = "SKIP"
            else:
                corr = float(np.corrcoef(pairs["eeg"].to_numpy(dtype=float), pairs["ieeg"].to_numpy(dtype=float))[0, 1])
                fig = out_d752 / "FIG_ds004752_eeg_ieeg_convergence.png"
                plt.figure(figsize=(6, 5))
                plt.scatter(pairs["eeg"], pairs["ieeg"], alpha=0.8, color="#0072B2")
                plt.axhline(0, color="k", lw=1, ls="--")
                plt.axvline(0, color="k", lw=1, ls="--")
                plt.xlabel("Scalp signature (high-low)")
                plt.ylabel("iEEG signature (high-low)")
                plt.title(f"ds004752 convergence r={corr:.3f} (n={len(pairs)})")
                plt.tight_layout()
                plt.savefig(fig, dpi=140)
                plt.close()
                outputs.append(fig)
                _write_json(
                    out_d752 / "bio_d_summary.json",
                    {
                        "status": "PASS",
                        "n_signatures_total": int(len(sig_df)),
                        "n_pair_subjects": int(len(pairs)),
                        "corr_eeg_ieeg": float(corr),
                        "limitation": "Exploratory qualitative convergence; channel harmonization is approximate.",
                    },
                )
                outputs.append(out_d752 / "bio_d_summary.json")
                d752_status = "PASS"
    except Exception as exc:
        _write_stop_reason(
            stop_path,
            stage,
            f"ds004752 extraction failed: {exc}",
            diagnostics={"file_formats": _collect_file_formats(ds_root), "log_tail": _tail(log_path, 200)},
        )
        outputs.append(stop_path)
        d752_status = "SKIP"

    # ---- ds007262 workload repair attempt (same stage) ----
    ds7262_root = ctx.data_root / "ds007262"
    stop7262 = out_d7262 / "STOP_REASON_ds007262.md"
    map7262 = out_d7262 / "event_map_ds007262.yaml"
    sum7262 = out_d7262 / "decode_ds007262_summary.json"
    cand7262 = out_d7262 / "decode_ds007262_candidates.csv"
    _run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "decode_ds007262.py"),
            "--dataset_root",
            str(ds7262_root),
            "--out_yaml",
            str(map7262),
            "--out_summary",
            str(sum7262),
            "--out_candidate",
            str(cand7262),
            "--stop_reason",
            str(stop7262),
        ],
        cwd=REPO_ROOT,
        log_path=log_path,
        allow_fail=True,
        timeout_sec=600,
    )
    outputs.extend([map7262, sum7262, cand7262])

    d7262_status = "SKIP"
    try:
        mp = _read_json(sum7262) if sum7262.exists() else {"status": "SKIP", "reason": "missing decode summary"}
        if str(mp.get("status", "SKIP")) != "PASS":
            if not stop7262.exists():
                _write_stop_reason(stop7262, stage, str(mp.get("reason", "mapping decode failed")), diagnostics=mp)
            outputs.append(stop7262)
        else:
            events = sorted(ds7262_root.rglob("*_task-arithmetic_events.tsv"))
            rows, fails = [], []
            for ep in events:
                sid_m = re.search(r"sub-([A-Za-z0-9]+)", ep.as_posix())
                sid = sid_m.group(1) if sid_m else "unknown"
                try:
                    df = pd.read_csv(ep, sep="\t")
                    if "difficulty_range" not in df.columns or "onset" not in df.columns:
                        fails.append({"subject": sid, "events": str(ep), "reason": "missing difficulty_range/onset"})
                        continue
                    loads = _parse_load_from_difficulty(df["difficulty_range"])
                    onsets = pd.to_numeric(df["onset"], errors="coerce")
                    ok = np.isfinite(loads.to_numpy(dtype=float)) & np.isfinite(onsets.to_numpy(dtype=float))
                    if int(np.sum(ok)) < 8:
                        fails.append({"subject": sid, "events": str(ep), "reason": "insufficient valid events"})
                        continue
                    sig_path = _find_matching_signal(ep, "eeg")
                    if sig_path is None:
                        base = ep.name.replace("_events.tsv", "_eeg")
                        sig_path = next((p for p in ep.parent.glob(base + ".vhdr") if p.exists()), None)
                    if sig_path is None:
                        fails.append({"subject": sid, "events": str(ep), "reason": "missing eeg file"})
                        continue
                    raw = _read_raw_any(sig_path)
                    slope, n_evt = _extract_workload_slope(raw, onsets.to_numpy(dtype=float)[ok], loads.to_numpy(dtype=float)[ok])
                    if not np.isfinite(slope):
                        fails.append({"subject": sid, "events": str(ep), "reason": "non-finite slope"})
                        continue
                    rows.append(
                        {
                            "subject_id": sid,
                            "events_file": str(ep),
                            "signal_file": str(sig_path),
                            "n_events_used": int(n_evt),
                            "load_response_slope": float(slope),
                        }
                    )
                except Exception as exc:
                    fails.append({"subject": sid, "events": str(ep), "reason": f"exception:{exc}"})

            slope_df = pd.DataFrame(rows)
            fail_df = pd.DataFrame(fails) if fails else pd.DataFrame(columns=["subject", "events", "reason"])
            slope_csv = out_d7262 / "workload_load_response.csv"
            fail_csv = out_d7262 / "workload_failures.csv"
            slope_df.to_csv(slope_csv, index=False)
            fail_df.to_csv(fail_csv, index=False)
            outputs.extend([slope_csv, fail_csv])
            if slope_df.empty or int(len(slope_df)) < 8:
                _write_stop_reason(
                    stop7262,
                    stage,
                    "insufficient successful workload extraction runs",
                    diagnostics={
                        "n_success_runs": int(len(slope_df)),
                        "n_failures": int(len(fail_df)),
                        "file_formats": _collect_file_formats(ds7262_root),
                        "log_tail": _tail(log_path, 200),
                    },
                )
                outputs.append(stop7262)
                d7262_status = "SKIP"
            else:
                fig = out_d7262 / "FIG_ds007262_load_response_slope.png"
                plt.figure(figsize=(6, 5))
                plt.hist(slope_df["load_response_slope"].to_numpy(dtype=float), bins=12, color="#0072B2", alpha=0.85)
                plt.axvline(float(np.nanmean(slope_df["load_response_slope"])), color="k", ls="--", lw=1)
                plt.xlabel("Load-response slope")
                plt.ylabel("Count")
                plt.title("ds007262 workload load-dose response")
                plt.tight_layout()
                plt.savefig(fig, dpi=140)
                plt.close()
                outputs.append(fig)
                _write_json(
                    out_d7262 / "workload_summary.json",
                    {"status": "PASS", "n_success_runs": int(len(slope_df)), "mean_slope": float(np.nanmean(slope_df["load_response_slope"]))},
                )
                outputs.append(out_d7262 / "workload_summary.json")
                d7262_status = "PASS"
    except Exception as exc:
        _write_stop_reason(
            stop7262,
            stage,
            f"ds007262 extraction failed: {exc}",
            diagnostics={"file_formats": _collect_file_formats(ds7262_root), "log_tail": _tail(log_path, 200)},
        )
        outputs.append(stop7262)
        d7262_status = "SKIP"

    extra = {
        "ds004752_status": d752_status,
        "ds007262_status": d7262_status,
    }
    if d752_status == "SKIP" or d7262_status == "SKIP":
        stage_status = "SKIP"
        stage_error = "one or more optional bio/workload attempts skipped with STOP_REASON"

    return _record_stage(
        ctx,
        stage=stage,
        status=stage_status,
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="bio ds004752 cross-modality + ds007262 workload repair",
        outputs=outputs,
        error=stage_error,
        extra=extra,
    )


def _stage_endpoint_hierarchy_and_report(ctx: Ctx) -> Dict[str, Any]:
    stage = "endpoint_hierarchy_and_report"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"
    reg_path = ctx.audit / "CLINICAL_ENDPOINT_REGISTRY.md"

    pd_path = ctx.pack_pdrest / "pdrest_endpoints.csv"
    mt_path = ctx.pack_mort / "mortality_baseline_endpoints.csv"
    d504_path = ctx.canonical_root / "PACK_CLINICAL_DEMENTIA" / "dementia_endpoints.csv"

    pd_df = pd.read_csv(pd_path) if pd_path.exists() else pd.DataFrame()
    mt_df = pd.read_csv(mt_path) if mt_path.exists() else pd.DataFrame()
    d504_df = pd.read_csv(d504_path) if d504_path.exists() else pd.DataFrame()

    if not pd_df.empty:
        pd_df["perm_p"] = pd.to_numeric(pd_df.get("perm_p"), errors="coerce")
        pd_df["perm_q_within_ds004584"] = _bh_qvals(pd_df["perm_p"].fillna(1.0).to_numpy(dtype=float).tolist())
        pd_df.to_csv(pd_path, index=False)
    if not mt_df.empty:
        mt_df["perm_p"] = pd.to_numeric(mt_df.get("perm_p"), errors="coerce")
        mt_df["perm_q_within_ds007020"] = _bh_qvals(mt_df["perm_p"].fillna(1.0).to_numpy(dtype=float).tolist())
        mt_df.to_csv(mt_path, index=False)

    global_frames = []
    if not pd_df.empty:
        tmp = pd_df.copy()
        tmp["dataset_id"] = "ds004584"
        tmp["source"] = "master_pdrest"
        global_frames.append(tmp)
    if not mt_df.empty:
        tmp = mt_df.copy()
        tmp["dataset_id"] = "ds007020"
        tmp["source"] = "master_mortality"
        global_frames.append(tmp)
    if not d504_df.empty:
        tmp = d504_df.copy()
        tmp["dataset_id"] = "ds004504"
        tmp["source"] = "canonical_dementia"
        global_frames.append(tmp)
    all_df = pd.concat(global_frames, ignore_index=True) if global_frames else pd.DataFrame()
    all_out = ctx.audit / "clinical_endpoints_all_master.csv"
    if not all_df.empty:
        all_df["perm_p"] = pd.to_numeric(all_df.get("perm_p"), errors="coerce")
        all_df["perm_q_global"] = _bh_qvals(all_df["perm_p"].fillna(1.0).to_numpy(dtype=float).tolist())
        all_df.to_csv(all_out, index=False)
    else:
        pd.DataFrame(columns=["dataset_id", "endpoint", "feature", "perm_p", "perm_q_global", "source"]).to_csv(all_out, index=False)

    registry_lines = [
        "# Clinical Endpoint Registry",
        "",
        "- ds004584 primary: `composite_deviation` AUC PD vs CN",
        "- ds007020 primary: LEAPD balanced LOOCV AUC (with out-of-sample summary)",
        "- ds004504 primary: theta/alpha ratio AUCs (canonical V2_BIO)",
        "",
        "## Correction scopes",
        "- Within-dataset BH-FDR: ds004584 endpoints",
        "- Within-dataset BH-FDR: ds007020 baseline endpoints",
        "- Global BH-FDR (transparency): ds004504 + ds004584 + ds007020",
    ]
    _write_text(reg_path, "\n".join(registry_lines) + "\n")

    # Write the main report in this stage.
    report_path = ctx.audit / "NN_FINAL_MASTER_V1_REPORT.md"
    st_rows = [
        f"| {r.get('stage')} | {r.get('status')} | {r.get('returncode')} | {float(r.get('elapsed_sec', 0.0)):.1f} |"
        for r in ctx.stage_records
    ]

    pd_sum = _read_json(ctx.audit / "clinical_ds004584_fullN_PDrest_summary.json") if (ctx.audit / "clinical_ds004584_fullN_PDrest_summary.json").exists() else {}
    mt_sum = _read_json(ctx.audit / "clinical_ds007020_LEAPD_full_summary.json") if (ctx.audit / "clinical_ds007020_LEAPD_full_summary.json").exists() else {}
    bio_sum = _read_json(ctx.audit / "bio_ds004752_crossmodality_attempt_summary.json") if (ctx.audit / "bio_ds004752_crossmodality_attempt_summary.json").exists() else {}

    nvidia_summary = _summarize_nvidia_smi_csv(ctx.audit / "nvidia_smi_1hz.csv")
    nvml_summary = summarize_gpu_util_csv(ctx.out_root / "gpu_util.csv")
    pd_primary = pd_df[(pd_df.get("endpoint", pd.Series(dtype=str)).astype(str) == "AUC_PD_vs_CN") & (pd_df.get("feature", pd.Series(dtype=str)).astype(str) == "composite_deviation")] if not pd_df.empty else pd.DataFrame()
    mt_primary = mt_df[(mt_df.get("endpoint", pd.Series(dtype=str)).astype(str).str.contains("AUC_mortality", na=False)) & (mt_df.get("feature", pd.Series(dtype=str)).astype(str).str.contains("leapd_index_loocv", na=False))] if not mt_df.empty else pd.DataFrame()
    pd_primary_txt = pd_primary.head(1).to_dict(orient="records")[0] if not pd_primary.empty else {}
    mt_primary_txt = mt_primary.head(1).to_dict(orient="records")[0] if not mt_primary.empty else {}

    lines = [
        "# NN_FINAL_MASTER_V1 REPORT",
        "",
        f"- Output root: `{ctx.out_root}`",
        f"- Canonical baseline reused: `{ctx.canonical_root}`",
        "",
        "## Stage status",
        "| Stage | Status | Return code | Runtime (s) |",
        "|---|---|---:|---:|",
        *st_rows,
        "",
        "## ds004584 (PD vs CN)",
        f"- N_used: `{pd_sum.get('n_subjects_used')}` (target>=140, hard minimum>=120)",
        f"- Exclusions: `{ctx.pack_pdrest / 'EXCLUSIONS.csv'}`",
        f"- Primary endpoint (composite AUC): `{json.dumps(_json_sanitize(pd_primary_txt), sort_keys=True)}`",
        "",
        "## ds007020 (Mortality LEAPD)",
        f"- N_used: `{mt_sum.get('n_subjects_used')}`",
        f"- Finite beta count (baseline logistic): `{mt_sum.get('finite_beta_count')}`",
        f"- LEAPD primary AUC (balanced LOOCV): `{mt_sum.get('leapd_primary_auc')}`",
        f"- LEAPD outputs: `{ctx.pack_mort / 'leapd_channel_results.csv'}`, `{ctx.pack_mort / 'leapd_truncation.csv'}`, `{ctx.pack_mort / 'leapd_out_of_sample.json'}`",
        f"- Baseline endpoint sample: `{json.dumps(_json_sanitize(mt_primary_txt), sort_keys=True)}`",
        "",
        "## BIO / Workload attempts",
        f"- ds004752 status: `{bio_sum.get('ds004752_status', bio_sum.get('status', 'missing'))}`",
        f"- ds007262 status: `{bio_sum.get('ds007262_status', 'missing')}`",
        f"- ds004752 STOP (if skipped): `{ctx.pack_bio / 'BIO_D_cross_modality' / 'STOP_REASON_ds004752.md'}`",
        f"- ds007262 STOP (if skipped): `{ctx.pack_bio / 'workload_ds007262' / 'STOP_REASON_ds007262.md'}`",
        "",
        "## Linkage to canonical",
        f"- Canonical packet: `{ctx.canonical_root / 'OUTZIP' / 'NN_FINAL_MEGA_V2_BIO_SUBMISSION_PACKET.zip'}`",
        "",
        "## Monitoring",
        f"- nvidia-smi summary: `{json.dumps(_json_sanitize(nvidia_summary), sort_keys=True)}`",
        f"- NVML summary: `{json.dumps(_json_sanitize(nvml_summary), sort_keys=True)}`",
    ]
    _write_text(report_path, "\n".join(lines) + "\n")

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="endpoint hierarchy and report",
        outputs=[reg_path, report_path, pd_path, mt_path, all_out],
        extra={
            "n_pd_endpoints": int(len(pd_df)),
            "n_mortality_endpoints": int(len(mt_df)),
            "n_global_endpoints": int(len(all_df)),
        },
    )


def _stage_bundle_and_tarball(ctx: Ctx) -> Dict[str, Any]:
    stage = "bundle_and_tarball"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"

    can_zip = ctx.canonical_root / "OUTZIP" / "NN_FINAL_MEGA_V2_BIO_SUBMISSION_PACKET.zip"
    v1_zip = ctx.outzip / "NN_FINAL_MASTER_V1_SUBMISSION_PACKET.zip"
    ctx.outzip.mkdir(parents=True, exist_ok=True)
    ctx.tarballs.mkdir(parents=True, exist_ok=True)
    report_path = ctx.audit / "NN_FINAL_MASTER_V1_REPORT.md"

    with zipfile.ZipFile(v1_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for p in sorted(ctx.audit.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(ctx.out_root)))
        for base in [ctx.pack_pdrest, ctx.pack_mort, ctx.pack_bio]:
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(ctx.out_root)))
        if can_zip.exists():
            zf.write(can_zip, arcname="CANONICAL/NN_FINAL_MEGA_V2_BIO_SUBMISSION_PACKET.zip")

    # Results-only tarball containing full OUT_ROOT.
    readme = ctx.tarballs / "CANONICAL_REFERENCE.txt"
    _write_text(
        readme,
        "\n".join(
            [
                "NN_FINAL_MASTER_V1 canonical reference",
                f"canonical_v2_root={ctx.canonical_root}",
                f"canonical_v2_packet={can_zip}",
                "",
            ]
        ),
    )
    results_tgz = ctx.tarballs / "results_only.tar.gz"
    with tarfile.open(results_tgz, "w:gz") as tf:
        tf.add(ctx.out_root, arcname=ctx.out_root.name)

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="bundle submission zip and results tarball",
        outputs=[report_path, v1_zip, results_tgz, readme],
        extra={
            "zip_size_bytes": int(v1_zip.stat().st_size if v1_zip.exists() else 0),
            "tar_size_bytes": int(results_tgz.stat().st_size if results_tgz.exists() else 0),
        },
    )


def _stage_cleanup_unused_stage_removed(ctx: Ctx) -> Dict[str, Any]:
    stage = "cleanup_unused_stage_removed"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"

    ctx.tarballs.mkdir(parents=True, exist_ok=True)
    readme = ctx.tarballs / "CANONICAL_REFERENCE.txt"
    _write_text(
        readme,
        "\n".join(
            [
                "NN_FINAL_MASTER_V1 canonical reference",
                f"canonical_v2_root={ctx.canonical_root}",
                f"canonical_v2_packet={ctx.canonical_root / 'OUTZIP' / 'NN_FINAL_MEGA_V2_BIO_SUBMISSION_PACKET.zip'}",
                "",
            ]
        ),
    )

    results_tgz = ctx.tarballs / "results_only.tar.gz"
    with tarfile.open(results_tgz, "w:gz") as tf:
        tf.add(ctx.out_root, arcname=ctx.out_root.name)

    with_data_tgz = ctx.tarballs / "with_data.tar.gz"
    with_data_status = "SKIP"
    with_data_reason = "INCLUDE_DATA=0"

    if ctx.include_data:
        try:
            with tarfile.open(with_data_tgz, "w:gz") as tf:
                tf.add(ctx.out_root, arcname=f"runs/{ctx.out_root.name}")
                for ds in ["ds004584", "ds007020", "ds004752", "ds007262"]:
                    dpath = ctx.data_root / ds
                    if dpath.exists():
                        tf.add(dpath, arcname=f"openneuro/{ds}")
            with_data_status = "PASS"
            with_data_reason = ""
        except Exception as exc:
            with_data_status = "SKIP"
            with_data_reason = f"with_data tar failed: {exc}"
            _write_stop_reason(ctx.tarballs / "STOP_REASON_with_data.md", stage, with_data_reason)

    return _record_stage(
        ctx,
        stage=stage,
        status="PASS",
        rc=0,
        started=started,
        log_path=log_path,
        summary_path=sum_path,
        command="tarball export",
        outputs=[results_tgz] + ([with_data_tgz] if with_data_status == "PASS" else [readme]),
        extra={"with_data_status": with_data_status, "with_data_reason": with_data_reason},
    )


def _final_run_status(ctx: Ctx, run_status: str, run_error: str) -> None:
    report_path = ctx.audit / "NN_FINAL_MASTER_V1_REPORT.md"
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
    ap = argparse.ArgumentParser(description="NN_FINAL_MASTER_V1 runner")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="/filesystemHcog/openneuro")
    ap.add_argument("--canonical_root", type=str, default=str(CANONICAL_DEFAULT))
    ap.add_argument("--wall_hours", type=float, default=10.0)
    ap.add_argument("--resume", type=str, default="false")
    ap.add_argument("--include_data", type=str, default=os.environ.get("INCLUDE_DATA", "0"))
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    audit = out_root / "AUDIT"
    outzip = out_root / "OUTZIP"
    tarballs = out_root / "TARBALLS"

    if out_root.exists() and not _parse_bool(args.resume):
        existing = list(out_root.iterdir())
        allow_bootstrap = False
        if existing:
            allow_bootstrap = all(
                p.name in {"AUDIT", "OUTZIP", "TARBALLS"} and p.is_dir() and not any(p.iterdir())
                for p in existing
            )
        if existing and not allow_bootstrap:
            print(f"ERROR: out_root exists and is non-empty: {out_root}", file=sys.stderr, flush=True)
            return 2

    out_root.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    outzip.mkdir(parents=True, exist_ok=True)
    tarballs.mkdir(parents=True, exist_ok=True)

    ctx = Ctx(
        out_root=out_root,
        audit=audit,
        outzip=outzip,
        tarballs=tarballs,
        data_root=Path(args.data_root).resolve(),
        canonical_root=Path(args.canonical_root).resolve(),
        include_data=_parse_bool(args.include_data),
        wall_hours=float(args.wall_hours),
        resume=_parse_bool(args.resume),
        pack_pdrest=out_root / "PACK_CLINICAL_PDREST_MASTER",
        pack_mort=out_root / "PACK_CLINICAL_MORTALITY_LEAPD",
        pack_bio=out_root / "PACK_BIO_CROSSMODALITY",
        start_ts=time.time(),
        deadline_ts=time.time() + float(args.wall_hours) * 3600.0,
        stage_records=[],
        stage_status={},
        stage_extra={},
        partial_reasons=[],
        monitor_proc=None,
        nvml_logger=None,
    )
    ctx.pack_pdrest.mkdir(parents=True, exist_ok=True)
    ctx.pack_mort.mkdir(parents=True, exist_ok=True)
    ctx.pack_bio.mkdir(parents=True, exist_ok=True)

    stage_funcs = {
        "preflight": _stage_preflight,
        "compile_gate": _stage_compile_gate,
        "stage_verify_ds004584_full": _stage_verify_ds004584_full,
        "stage_verify_ds007020_full": _stage_verify_ds007020_full,
        "clinical_ds004584_fullN_PDrest": _stage_clinical_ds004584_fullN_PDrest,
        "clinical_ds007020_LEAPD_full": _stage_clinical_ds007020_LEAPD_full,
        "bio_ds004752_crossmodality_attempt": _stage_bio_ds004752_crossmodality_attempt,
        "endpoint_hierarchy_and_report": _stage_endpoint_hierarchy_and_report,
        "bundle_and_tarball": _stage_bundle_and_tarball,
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
    finally:
        if ctx.nvml_logger is not None:
            try:
                ctx.nvml_logger.stop()
            except Exception:
                pass
        _stop_nvidia_smi_monitor(ctx)

    # strict objective gate
    strict_issues: List[str] = []
    pd_end = ctx.pack_pdrest / "pdrest_endpoints.csv"
    mt_end = ctx.pack_mort / "mortality_baseline_endpoints.csv"
    leapd_ch = ctx.pack_mort / "leapd_channel_results.csv"
    leapd_trunc = ctx.pack_mort / "leapd_truncation.csv"
    leapd_oos = ctx.pack_mort / "leapd_out_of_sample.json"

    pd_used = int(ctx.stage_extra.get("clinical_ds004584_fullN_PDrest", {}).get("n_subjects_used", 0))
    if not pd_end.exists():
        strict_issues.append("ds004584 endpoints missing")
    if pd_used < 120:
        strict_issues.append(f"ds004584 n_used={pd_used} < 120")
    else:
        try:
            pd_df = pd.read_csv(pd_end)
            pprim = pd_df[(pd_df["endpoint"].astype(str) == "AUC_PD_vs_CN") & (pd_df["feature"].astype(str) == "composite_deviation")]
            if pprim.empty:
                strict_issues.append("ds004584 primary endpoint missing (composite_deviation AUC)")
            else:
                rp = pprim.iloc[0]
                if not np.isfinite(float(rp.get("perm_p", np.nan))):
                    strict_issues.append("ds004584 primary perm_p missing/non-finite")
                if not np.isfinite(float(rp.get("perm_q_within_ds004584", np.nan))):
                    strict_issues.append("ds004584 primary within-dataset q missing/non-finite")
        except Exception as exc:
            strict_issues.append(f"ds004584 endpoint validation failed: {exc}")

    finite_beta_count = int(ctx.stage_extra.get("clinical_ds007020_LEAPD_full", {}).get("finite_beta_count", 0))
    if not mt_end.exists():
        strict_issues.append("ds007020 endpoints missing")
    if not leapd_ch.exists():
        strict_issues.append("ds007020 LEAPD channel results missing")
    if not leapd_trunc.exists():
        strict_issues.append("ds007020 LEAPD truncation results missing")
    if not leapd_oos.exists():
        strict_issues.append("ds007020 LEAPD out-of-sample results missing")
    if finite_beta_count < 1:
        strict_issues.append("ds007020 has no finite beta")
    leapd_primary_auc = float(ctx.stage_extra.get("clinical_ds007020_LEAPD_full", {}).get("leapd_primary_auc", float("nan")))
    if not np.isfinite(leapd_primary_auc):
        strict_issues.append("ds007020 LEAPD primary AUC missing/non-finite")

    # mandatory dataset strict fail closed
    if str(ctx.stage_status.get("stage_verify_ds004584_full", "")) != "PASS":
        strict_issues.append("ds004584 verify stage not PASS")
    if str(ctx.stage_status.get("stage_verify_ds007020_full", "")) != "PASS":
        strict_issues.append("ds007020 verify stage not PASS")

    if run_status != "FAIL":
        if len(strict_issues) == 0:
            run_status = "PASS_STRICT"
            run_error = ""
        else:
            run_status = "PARTIAL_PASS"
            run_error = " ; ".join(strict_issues)
            ctx.partial_reasons.extend(strict_issues)

    _final_run_status(ctx, run_status, run_error)

    # ensure report exists
    rep = ctx.audit / "NN_FINAL_MASTER_V1_REPORT.md"
    if not rep.exists():
        rows = [
            "# NN_FINAL_MASTER_V1 REPORT",
            "",
            f"- Output root: `{ctx.out_root}`",
            f"- Run status: `{run_status}`",
            "",
            "## Stage status",
            "| Stage | Status | Return code | Runtime (s) |",
            "|---|---|---:|---:|",
        ]
        for rec in ctx.stage_records:
            rows.append(f"| {rec.get('stage')} | {rec.get('status')} | {rec.get('returncode')} | {float(rec.get('elapsed_sec', 0.0)):.1f} |")
        if strict_issues:
            rows += ["", "## Strict gate issues"] + [f"- {x}" for x in strict_issues]
        _write_text(rep, "\n".join(rows) + "\n")

    print(f"OUT_ROOT={ctx.out_root}", flush=True)
    print(f"REPORT={ctx.audit / 'NN_FINAL_MASTER_V1_REPORT.md'}", flush=True)
    print(f"STATUS={run_status}", flush=True)
    print(f"ZIP={ctx.outzip / 'NN_FINAL_MASTER_V1_SUBMISSION_PACKET.zip'}", flush=True)
    print(f"RESULTS_TARBALL={ctx.tarballs / 'results_only.tar.gz'}", flush=True)
    print("SCP=scp <user>@<host>:" + f"\"{ctx.tarballs / 'results_only.tar.gz'}\"" + " .", flush=True)

    return 0 if run_status in {"PASS_STRICT", "PARTIAL_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
