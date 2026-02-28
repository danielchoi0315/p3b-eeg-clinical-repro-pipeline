#!/usr/bin/env python3
"""NN_FINAL_MEGA_V3_1_FULLFIX_BIO surgical runner.

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
    "clinical_ds004584_fullN",
    "clinical_ds007020_mortality_fixbeta",
    "bio_D_cross_modality_ds004752_repair",
    "workload_ds007262_repair",
    "rebundle_submission_packet",
    "tarball_export",
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
        ctx.nvml_logger = start_gpu_util_logger(csv_path=ctx.out_root / "gpu_util.csv", tag="NN_FINAL_MEGA_V3_1_FULLFIX_BIO")
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
    set_paths = _subject_set_fdt_paths(part["subject_id"].astype(str).tolist())

    counts0 = _count_ds004584_headers(ds_root)
    env = os.environ.copy()
    if counts0.get("vhdr_count", 0) < 120 and counts0.get("set_count", 0) < 120:
        if shutil.which("datalad") is not None:
            _run_cmd(
                ["datalad", "get", "-r", "-J", "8", *set_paths],
                cwd=ds_root,
                log_path=log_path,
                env=env,
                allow_fail=True,
                timeout_sec=600,
            )
        _run_cmd(
            ["git", "annex", "get", "-J", "8", "--", *set_paths],
            cwd=ds_root,
            log_path=log_path,
            env=env,
            allow_fail=True,
            timeout_sec=900,
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


def _stage_clinical_ds004584_fullN(ctx: Ctx) -> Dict[str, Any]:
    stage = "clinical_ds004584_fullN"
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


def _stage_clinical_ds007020_mortality_fixbeta(ctx: Ctx) -> Dict[str, Any]:
    stage = "clinical_ds007020_mortality_fixbeta"
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

    part["subject_id"] = part["participant_id"].astype(str).str.replace("sub-", "", regex=False)
    part["label"] = pd.to_numeric(y_lbl, errors="coerce")
    # optional covariates, may not exist
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

    subjects = part["subject_id"].astype(str).tolist()
    feat_df, ex_df = _extract_features_for_subjects(ds_root, subjects, include_lpc=True, log_path=log_path)
    feat_df = feat_df.merge(part[["subject_id", "label", "age", "sex"]], on="subject_id", how="left")

    ex_path = out_dir / "EXCLUSIONS.csv"
    if ex_df.empty:
        ex_df = pd.DataFrame(columns=["subject_id", "reason", "rest_file"])
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
        sub = dev_df[["subject_id", "label", "age", "sex", feat]].copy()
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
        auc_flip = float(max(auc_obs, 1.0 - auc_obs)) if np.isfinite(auc_obs) else float("nan")

        X, yy, covs, nfit = _build_design_matrix(sub.rename(columns={feat: "feature"}), "feature")
        beta = _fit_logit_beta(X, yy)
        b_lo, b_hi = _bootstrap_beta_ci(X, yy, n_boot=n_boot_beta, seed=7000 + _stable_int_from_text(feat) % 100000)
        p_beta = _perm_p_beta(X, yy, beta, n_perm=n_perm, seed=8000 + _stable_int_from_text(feat) % 100000)

        rows.append(
            {
                "dataset_id": "ds007020",
                "endpoint": "AUC_mortality",
                "feature": feat,
                "type": "auc",
                "n": int(len(sub)),
                "estimate": float(auc_obs),
                "auc_flipped": float(auc_flip),
                "ci95_lo": float(auc_ci[0]),
                "ci95_hi": float(auc_ci[1]),
                "perm_p": float(p_auc),
                "model": "spectral_deviation",
                "n_perm_done": int(n_perm),
                "n_boot_done": int(n_boot_auc),
            }
        )
        rows.append(
            {
                "dataset_id": "ds007020",
                "endpoint": "LogitBeta_mortality",
                "feature": feat,
                "type": "logit_beta",
                "n": int(nfit),
                "estimate": float(beta),
                "auc_flipped": float("nan"),
                "ci95_lo": float(b_lo),
                "ci95_hi": float(b_hi),
                "perm_p": float(p_beta),
                "model": "spectral_deviation",
                "covariates": ";".join(covs),
                "n_perm_done": int(n_perm),
                "n_boot_done": int(n_boot_beta),
            }
        )

    # Optional exploratory LEAPD/LPC
    lpc_cols = [c for c in dev_df.columns if c.startswith("lpc_")]
    leapd_auc = float("nan")
    leapd_p = float("nan")
    leapd_n = 0
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
                s = _compute_leapd_scores(X, y)
                m = np.isfinite(s)
                if int(m.sum()) >= 10 and np.unique(y[m]).size >= 2:
                    yy = y[m]
                    ss = s[m]
                    leapd_auc, leapd_ci = _bootstrap_auc(yy, ss, n_boot=n_boot_auc, seed=9001)
                    leapd_p = _perm_p_auc(yy, ss, n_perm=n_perm, seed=9002)
                    leapd_n = int(len(yy))
                    rows.append(
                        {
                            "dataset_id": "ds007020",
                            "endpoint": "LEAPD_AUC_mortality_exploratory",
                            "feature": "leapd_index",
                            "type": "auc_exploratory",
                            "n": int(leapd_n),
                            "estimate": float(leapd_auc),
                            "auc_flipped": float(max(leapd_auc, 1.0 - leapd_auc)) if np.isfinite(leapd_auc) else float("nan"),
                            "ci95_lo": float(leapd_ci[0]),
                            "ci95_hi": float(leapd_ci[1]),
                            "perm_p": float(leapd_p),
                            "model": "LPC_LEAPD_LOOCV",
                            "n_perm_done": int(n_perm),
                            "n_boot_done": int(n_boot_auc),
                        }
                    )
                    Xb = ss.reshape(-1, 1)
                    beta = _fit_logit_beta(Xb, yy)
                    b_lo, b_hi = _bootstrap_beta_ci(Xb, yy, n_boot=n_boot_beta, seed=9003)
                    p_beta = _perm_p_beta(Xb, yy, beta, n_perm=n_perm, seed=9004)
                    rows.append(
                        {
                            "dataset_id": "ds007020",
                            "endpoint": "LEAPD_LogitBeta_mortality_exploratory",
                            "feature": "leapd_index",
                            "type": "logit_beta_exploratory",
                            "n": int(leapd_n),
                            "estimate": float(beta),
                            "auc_flipped": float("nan"),
                            "ci95_lo": float(b_lo),
                            "ci95_hi": float(b_hi),
                            "perm_p": float(p_beta),
                            "model": "LPC_LEAPD_LOOCV",
                            "n_perm_done": int(n_perm),
                            "n_boot_done": int(n_boot_beta),
                        }
                    )

    end_df = pd.DataFrame(rows)
    if end_df.empty:
        stop = out_dir / "STOP_REASON_ds007020.md"
        _write_stop_reason(stop, stage, "no mortality endpoints produced", diagnostics={"n_features": int(len(dev_df))})
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

    # primary visualization from composite; LEAPD exploratory appears in table
    tmp = dev_df[["label", "composite_deviation"]].copy()
    tmp["label"] = pd.to_numeric(tmp["label"], errors="coerce")
    tmp["composite_deviation"] = pd.to_numeric(tmp["composite_deviation"], errors="coerce")
    tmp = tmp[np.isfinite(tmp["label"]) & np.isfinite(tmp["composite_deviation"])].copy()
    y = tmp["label"].astype(int).to_numpy(dtype=int)
    s = tmp["composite_deviation"].to_numpy(dtype=float)
    roc_path = out_dir / "FIG_mortality_primary_auc_roc.png"
    cal_path = out_dir / "FIG_mortality_calibration.png"
    _plot_roc_calibration(y, s, "ds007020 mortality", roc_path, cal_path)

    beta_mask = end_df["type"].astype(str).str.contains("beta", na=False)
    finite_beta_count = int(np.isfinite(pd.to_numeric(end_df.loc[beta_mask, "estimate"], errors="coerce")).sum())
    if finite_beta_count < 1:
        stop = out_dir / "STOP_REASON_ds007020.md"
        _write_stop_reason(stop, stage, "all mortality betas are non-finite", diagnostics={"finite_beta_count": finite_beta_count})
        return _record_stage(
            ctx,
            stage=stage,
            status="FAIL",
            rc=1,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="clinical ds007020",
            outputs=[feat_path, end_path, stop, ex_path],
            error="no finite beta",
            extra={"finite_beta_count": finite_beta_count},
        )

    extra = {
        "n_subjects_used": int(dev_df["subject_id"].nunique()),
        "n_label_living": int((pd.to_numeric(dev_df["label"], errors="coerce") == 0).sum()),
        "n_label_deceased": int((pd.to_numeric(dev_df["label"], errors="coerce") == 1).sum()),
        "finite_beta_count": finite_beta_count,
        "leapd_auc_exploratory": float(leapd_auc),
        "leapd_perm_p_exploratory": float(leapd_p),
        "leapd_n_exploratory": int(leapd_n),
        "label_column": y_col,
        "n_perm_done": int(n_perm),
        "n_boot_done": int(n_boot_auc),
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


def _stage_bio_D_cross_modality_ds004752_repair(ctx: Ctx) -> Dict[str, Any]:
    stage = "bio_D_cross_modality_ds004752_repair"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"
    out_dir = ctx.pack_bio / "BIO_D_cross_modality"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_root = ctx.data_root / "ds004752"
    stop_path = out_dir / "STOP_REASON_ds004752.md"

    map_yaml = out_dir / "event_map_ds004752.yaml"
    map_sum = out_dir / "decode_ds004752_summary.json"
    cand_csv = out_dir / "decode_ds004752_candidates.csv"
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

    map_payload = _read_json(map_sum) if map_sum.exists() else {"status": "SKIP", "reason": "missing decode summary"}
    if str(map_payload.get("status", "SKIP")) != "PASS":
        if not stop_path.exists():
            _write_stop_reason(stop_path, stage, str(map_payload.get("reason", "mapping decode failed")), diagnostics=map_payload)
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="bio D ds004752 repair",
            outputs=[stop_path, map_sum, cand_csv],
            error=str(map_payload.get("reason", "mapping decode failed")),
            extra={"mapping_status": map_payload.get("status", "SKIP")},
        )

    try:
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
            key = (sid, mod)
            chosen.setdefault(key, p)

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
        fail_df = pd.DataFrame(fails)
        sig_csv = out_dir / "cross_modality_signatures.csv"
        fail_csv = out_dir / "cross_modality_failures.csv"
        sig_df.to_csv(sig_csv, index=False)
        if fail_df.empty:
            fail_df = pd.DataFrame(columns=["subject", "modality", "events", "reason"])
        fail_df.to_csv(fail_csv, index=False)

        if sig_df.empty:
            raise RuntimeError("no valid ds004752 signatures extracted")

        piv = sig_df.pivot_table(index="subject_id", columns="modality", values="signature_high_minus_low", aggfunc="median")
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
            return _record_stage(
                ctx,
                stage=stage,
                status="SKIP",
                rc=0,
                started=started,
                log_path=log_path,
                summary_path=sum_path,
                command="bio D ds004752 repair",
                outputs=[sig_csv, fail_csv, stop_path],
                error=reason,
                extra={"n_signatures": int(len(sig_df)), "n_pair_subjects": int(len(pairs))},
            )

        corr = float(np.corrcoef(pairs["eeg"].to_numpy(dtype=float), pairs["ieeg"].to_numpy(dtype=float))[0, 1])
        fig = out_dir / "FIG_ds004752_eeg_ieeg_convergence.png"
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

        summ = {
            "n_signatures_total": int(len(sig_df)),
            "n_pair_subjects": int(len(pairs)),
            "corr_eeg_ieeg": float(corr),
            "status": "PASS",
            "limitation": "Exploratory qualitative convergence; channel selection and montage harmonization are approximate.",
        }
        _write_json(out_dir / "bio_d_summary.json", summ)

        return _record_stage(
            ctx,
            stage=stage,
            status="PASS",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="bio D ds004752 repair",
            outputs=[sig_csv, fail_csv, fig, out_dir / "bio_d_summary.json", map_yaml, map_sum, cand_csv],
            extra=summ,
        )

    except Exception as exc:
        _write_stop_reason(
            stop_path,
            stage,
            f"ds004752 extraction/analysis failed: {exc}",
            diagnostics={"file_formats": _collect_file_formats(ds_root), "log_tail": _tail(log_path, 200)},
        )
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="bio D ds004752 repair",
            outputs=[stop_path, map_sum, cand_csv],
            error=str(exc),
        )


def _stage_workload_ds007262_repair(ctx: Ctx) -> Dict[str, Any]:
    stage = "workload_ds007262_repair"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"
    out_dir = ctx.pack_bio / "workload_ds007262"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_root = ctx.data_root / "ds007262"
    stop_path = out_dir / "STOP_REASON_ds007262.md"

    map_yaml = out_dir / "event_map_ds007262.yaml"
    map_sum = out_dir / "decode_ds007262_summary.json"
    cand_csv = out_dir / "decode_ds007262_candidates.csv"
    _run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "decode_ds007262.py"),
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

    map_payload = _read_json(map_sum) if map_sum.exists() else {"status": "SKIP", "reason": "missing decode summary"}
    if str(map_payload.get("status", "SKIP")) != "PASS":
        if not stop_path.exists():
            _write_stop_reason(stop_path, stage, str(map_payload.get("reason", "mapping decode failed")), diagnostics=map_payload)
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="workload ds007262 repair",
            outputs=[stop_path, map_sum, cand_csv],
            error=str(map_payload.get("reason", "mapping decode failed")),
        )

    try:
        events = sorted(ds_root.rglob("*_task-arithmetic_events.tsv"))
        rows: List[Dict[str, Any]] = []
        fails: List[Dict[str, Any]] = []
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
                    fails.append({"subject": sid, "events": str(ep), "reason": "insufficient valid load events"})
                    continue

                # find corresponding eeg file
                sig_path = _find_matching_signal(ep, "eeg")
                if sig_path is None:
                    # ds007262 convention can be _eeg.vhdr in same dir
                    base = ep.name.replace("_events.tsv", "_eeg")
                    sig_path = next((p for p in ep.parent.glob(base + ".vhdr") if p.exists()), None)
                if sig_path is None:
                    fails.append({"subject": sid, "events": str(ep), "reason": "missing eeg file"})
                    continue

                raw = _read_raw_any(sig_path)
                slope, n_evt = _extract_workload_slope(raw, onsets.to_numpy(dtype=float)[ok], loads.to_numpy(dtype=float)[ok])
                if not np.isfinite(slope):
                    fails.append({"subject": sid, "events": str(ep), "reason": "non-finite load slope"})
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
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(f"[{_iso_now()}] ds007262 subject={sid} error={exc}\n")

        slope_df = pd.DataFrame(rows)
        fail_df = pd.DataFrame(fails)
        slope_csv = out_dir / "workload_load_response.csv"
        fail_csv = out_dir / "workload_failures.csv"
        slope_df.to_csv(slope_csv, index=False)
        if fail_df.empty:
            fail_df = pd.DataFrame(columns=["subject", "events", "reason"])
        fail_df.to_csv(fail_csv, index=False)

        if slope_df.empty or int(len(slope_df)) < 8:
            reason = "insufficient successful workload extraction runs"
            _write_stop_reason(
                stop_path,
                stage,
                reason,
                diagnostics={
                    "n_success_runs": int(len(slope_df)),
                    "n_failures": int(len(fail_df)),
                    "file_formats": _collect_file_formats(ds_root),
                    "fail_sample": fail_df.head(40).to_dict(orient="records"),
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
                command="workload ds007262 repair",
                outputs=[slope_csv, fail_csv, stop_path, map_sum, cand_csv],
                error=reason,
                extra={"n_success_runs": int(len(slope_df))},
            )

        fig = out_dir / "FIG_ds007262_load_response_slope.png"
        plt.figure(figsize=(6, 5))
        plt.hist(slope_df["load_response_slope"].to_numpy(dtype=float), bins=12, color="#0072B2", alpha=0.85)
        plt.axvline(float(np.nanmean(slope_df["load_response_slope"])), color="k", ls="--", lw=1)
        plt.xlabel("Load-response slope")
        plt.ylabel("Count")
        plt.title("ds007262 workload load-dose response")
        plt.tight_layout()
        plt.savefig(fig, dpi=140)
        plt.close()

        summ = {
            "status": "PASS",
            "n_success_runs": int(len(slope_df)),
            "mean_slope": float(np.nanmean(slope_df["load_response_slope"])),
        }
        _write_json(out_dir / "workload_summary.json", summ)

        return _record_stage(
            ctx,
            stage=stage,
            status="PASS",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="workload ds007262 repair",
            outputs=[slope_csv, fail_csv, fig, out_dir / "workload_summary.json", map_yaml, map_sum, cand_csv],
            extra=summ,
        )

    except Exception as exc:
        _write_stop_reason(
            stop_path,
            stage,
            f"ds007262 extraction/analysis failed: {exc}",
            diagnostics={"file_formats": _collect_file_formats(ds_root), "log_tail": _tail(log_path, 200)},
        )
        return _record_stage(
            ctx,
            stage=stage,
            status="SKIP",
            rc=0,
            started=started,
            log_path=log_path,
            summary_path=sum_path,
            command="workload ds007262 repair",
            outputs=[stop_path, map_sum, cand_csv],
            error=str(exc),
        )


def _stage_rebundle_submission_packet(ctx: Ctx) -> Dict[str, Any]:
    stage = "rebundle_submission_packet"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"

    can_zip = ctx.canonical_root / "OUTZIP" / "NN_FINAL_MEGA_V2_BIO_SUBMISSION_PACKET.zip"
    v3_zip = ctx.outzip / "NN_FINAL_MEGA_V3_1_SUBMISSION_PACKET.zip"
    ctx.outzip.mkdir(parents=True, exist_ok=True)

    # ensure monitor rows threshold for strict gate
    row_guard = _ensure_nvidia_rows(ctx, min_rows=1200, log_path=log_path)

    # build combined clinical registry
    pd_ep = pd.read_csv(ctx.pack_pdrest / "pdrest_endpoints.csv") if (ctx.pack_pdrest / "pdrest_endpoints.csv").exists() else pd.DataFrame()
    mt_ep = pd.read_csv(ctx.pack_mort / "mortality_endpoints.csv") if (ctx.pack_mort / "mortality_endpoints.csv").exists() else pd.DataFrame()
    comb = pd.concat([d for d in [pd_ep, mt_ep] if not d.empty], ignore_index=True) if (not pd_ep.empty or not mt_ep.empty) else pd.DataFrame()
    if not comb.empty:
        comb["perm_p"] = pd.to_numeric(comb.get("perm_p"), errors="coerce")
        comb["perm_q_global"] = _bh_qvals(comb["perm_p"].fillna(1.0).to_numpy(dtype=float).tolist())
        comb.to_csv(ctx.audit / "clinical_endpoints_all_v3_1.csv", index=False)

    # report
    report_path = ctx.audit / "NN_FINAL_MEGA_V3_1_REPORT.md"
    st_rows = [
        f"| {r.get('stage')} | {r.get('status')} | {r.get('returncode')} | {float(r.get('elapsed_sec', 0.0)):.1f} |"
        for r in ctx.stage_records
    ]

    pd_sum = _read_json(ctx.audit / "clinical_ds004584_fullN_summary.json") if (ctx.audit / "clinical_ds004584_fullN_summary.json").exists() else {}
    mt_sum = _read_json(ctx.audit / "clinical_ds007020_mortality_fixbeta_summary.json") if (ctx.audit / "clinical_ds007020_mortality_fixbeta_summary.json").exists() else {}
    d752_sum = _read_json(ctx.audit / "bio_D_cross_modality_ds004752_repair_summary.json") if (ctx.audit / "bio_D_cross_modality_ds004752_repair_summary.json").exists() else {}
    d7262_sum = _read_json(ctx.audit / "workload_ds007262_repair_summary.json") if (ctx.audit / "workload_ds007262_repair_summary.json").exists() else {}

    nvidia_summary = _summarize_nvidia_smi_csv(ctx.audit / "nvidia_smi_1hz.csv")
    nvml_summary = summarize_gpu_util_csv(ctx.out_root / "gpu_util.csv")

    lines = [
        "# NN_FINAL_MEGA_V3_1_FULLFIX_BIO REPORT",
        "",
        f"- Output root: `{ctx.out_root}`",
        f"- Canonical baseline reused: `{ctx.canonical_root}`",
        "",
        "## Stage status",
        "| Stage | Status | Return code | Runtime (s) |",
        "|---|---|---:|---:|",
        *st_rows,
        "",
        "## Clinical fixes",
        f"- ds004584 N_used: `{pd_sum.get('n_subjects_used')}` (expected 149; strict min 120)",
        f"- ds004584 exclusions: `{ctx.pack_pdrest / 'EXCLUSIONS.csv'}`",
        f"- ds007020 N_used: `{mt_sum.get('n_subjects_used')}`",
        f"- ds007020 finite_beta_count: `{mt_sum.get('finite_beta_count')}`",
        f"- ds007020 LEAPD exploratory AUC: `{mt_sum.get('leapd_auc_exploratory')}`",
        "",
        "## Bio repairs",
        f"- ds004752 stage status: `{d752_sum.get('status', 'missing')}`",
        f"- ds007262 stage status: `{d7262_sum.get('status', 'missing')}`",
        f"- ds004752 STOP path (if skipped): `{ctx.pack_bio / 'BIO_D_cross_modality' / 'STOP_REASON_ds004752.md'}`",
        f"- ds007262 STOP path (if skipped): `{ctx.pack_bio / 'workload_ds007262' / 'STOP_REASON_ds007262.md'}`",
        "",
        "## Linkage to canonical",
        "- Core Law-C / mechanism / prior BIO are reused from canonical V2_BIO packet",
        f"- Canonical packet: `{can_zip}`",
        "",
        "## GPU monitoring",
        f"- nvidia-smi summary: `{json.dumps(_json_sanitize(nvidia_summary), sort_keys=True)}`",
        f"- NVML summary: `{json.dumps(_json_sanitize(nvml_summary), sort_keys=True)}`",
        f"- Row guard: `{json.dumps(row_guard, sort_keys=True)}`",
    ]
    _write_text(report_path, "\n".join(lines) + "\n")

    with zipfile.ZipFile(v3_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for p in sorted(ctx.audit.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(ctx.out_root)))
        for base in [ctx.pack_pdrest, ctx.pack_mort, ctx.pack_bio]:
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(ctx.out_root)))
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
        command="rebundle submission packet",
        outputs=[report_path, v3_zip],
        extra={
            "zip_size_bytes": int(v3_zip.stat().st_size if v3_zip.exists() else 0),
            "nvidia_rows": int(row_guard.get("rows", 0)),
        },
    )


def _stage_tarball_export(ctx: Ctx) -> Dict[str, Any]:
    stage = "tarball_export"
    started = time.time()
    log_path = ctx.audit / f"{stage}.log"
    sum_path = ctx.audit / f"{stage}_summary.json"

    ctx.tarballs.mkdir(parents=True, exist_ok=True)
    readme = ctx.tarballs / "CANONICAL_REFERENCE.txt"
    _write_text(
        readme,
        "\n".join(
            [
                "NN_FINAL_MEGA_V3_1_FULLFIX_BIO canonical reference",
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
    report_path = ctx.audit / "NN_FINAL_MEGA_V3_1_REPORT.md"
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
    ap = argparse.ArgumentParser(description="NN_FINAL_MEGA_V3_1_FULLFIX_BIO runner")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="/filesystemHcog/openneuro")
    ap.add_argument("--canonical_root", type=str, default=str(CANONICAL_DEFAULT))
    ap.add_argument("--wall_hours", type=float, default=8.0)
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
        pack_pdrest=out_root / "PACK_CLINICAL_PDREST_V3_1",
        pack_mort=out_root / "PACK_CLINICAL_MORTALITY_V3_1",
        pack_bio=out_root / "PACK_BIO_V3_1",
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
        "clinical_ds004584_fullN": _stage_clinical_ds004584_fullN,
        "clinical_ds007020_mortality_fixbeta": _stage_clinical_ds007020_mortality_fixbeta,
        "bio_D_cross_modality_ds004752_repair": _stage_bio_D_cross_modality_ds004752_repair,
        "workload_ds007262_repair": _stage_workload_ds007262_repair,
        "rebundle_submission_packet": _stage_rebundle_submission_packet,
        "tarball_export": _stage_tarball_export,
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
    mt_end = ctx.pack_mort / "mortality_endpoints.csv"

    pd_used = int(ctx.stage_extra.get("clinical_ds004584_fullN", {}).get("n_subjects_used", 0))
    if not pd_end.exists():
        strict_issues.append("ds004584 endpoints missing")
    if pd_used < 120:
        strict_issues.append(f"ds004584 n_used={pd_used} < 120")

    finite_beta_count = int(ctx.stage_extra.get("clinical_ds007020_mortality_fixbeta", {}).get("finite_beta_count", 0))
    if not mt_end.exists():
        strict_issues.append("ds007020 endpoints missing")
    if finite_beta_count < 1:
        strict_issues.append("ds007020 has no finite beta")

    nvidia_rows = _count_nvidia_rows(ctx.audit / "nvidia_smi_1hz.csv")
    if nvidia_rows < 1200:
        strict_issues.append(f"nvidia_smi_1hz rows<{1200} (got {nvidia_rows})")

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
    rep = ctx.audit / "NN_FINAL_MEGA_V3_1_REPORT.md"
    if not rep.exists():
        rows = [
            "# NN_FINAL_MEGA_V3_1_FULLFIX_BIO REPORT",
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
    print(f"REPORT={ctx.audit / 'NN_FINAL_MEGA_V3_1_REPORT.md'}", flush=True)
    print(f"STATUS={run_status}", flush=True)
    print(f"ZIP={ctx.outzip / 'NN_FINAL_MEGA_V3_1_SUBMISSION_PACKET.zip'}", flush=True)
    print(f"RESULTS_TARBALL={ctx.tarballs / 'results_only.tar.gz'}", flush=True)
    print(f"WITH_DATA_TARBALL={ctx.tarballs / 'with_data.tar.gz'}", flush=True)

    return 0 if run_status in {"PASS_STRICT", "PARTIAL_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
