#!/usr/bin/env python3
"""Deep mechanism evaluation for ds003838 (load -> pupil -> P3)."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

from common.lawc_audit import align_probe_events_to_epochs, bh_fdr  # noqa: E402
from p3b_pipeline.config import load_yaml  # noqa: E402
from p3b_pipeline.h5io import iter_subject_feature_files, read_subject_h5  # noqa: E402
from p3b_pipeline.pupil import find_eyetrack_file, load_pupil_timeseries  # noqa: E402


@dataclass
class SubjectPayload:
    subject_key: str
    n_trials: int
    load_z: np.ndarray
    pupil_z: np.ndarray
    p3_z: np.ndarray
    covs_z: np.ndarray


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--dataset_id", type=str, default="ds003838")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--event_map", type=Path, default=Path("configs/mechanism_event_map.yaml"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--seeds", type=str, default="0-49")
    ap.add_argument("--parallel_procs", type=int, default=6)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--min_trials", type=int, default=20)
    ap.add_argument("--onset_tolerance", type=float, default=0.012)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


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


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _zscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mu = float(np.nanmean(arr)) if np.isfinite(arr).any() else 0.0
    sd = float(np.nanstd(arr)) if np.isfinite(arr).any() else 0.0
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.full_like(arr, np.nan, dtype=float)
    return (arr - mu) / sd


def _lstsq_beta(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    yv = np.asarray(y, dtype=float)
    xv = np.asarray(x, dtype=float)
    if xv.ndim == 1:
        xv = xv[:, None]
    m = np.isfinite(yv)
    for j in range(xv.shape[1]):
        m = m & np.isfinite(xv[:, j])
    yv = yv[m]
    xv = xv[m]
    if yv.size <= xv.shape[1] + 2:
        return np.full(xv.shape[1] + 1, np.nan, dtype=float), yv, xv
    X = np.column_stack([np.ones(len(yv), dtype=float), xv])
    try:
        beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.full(X.shape[1], np.nan, dtype=float)
    return beta, yv, xv


def _effects_from_arrays(load_z: np.ndarray, pupil_z: np.ndarray, p3_z: np.ndarray, covs_z: np.ndarray) -> Dict[str, float]:
    if covs_z.ndim == 1:
        covs_z = covs_z[:, None]
    cov_use = np.asarray(covs_z, dtype=float)
    keep_cov_cols = []
    for c in range(cov_use.shape[1]):
        col = cov_use[:, c]
        if np.isfinite(col).mean() >= 0.5 and np.nanstd(col) > 1e-12:
            keep_cov_cols.append(c)
    cov_use = cov_use[:, keep_cov_cols] if keep_cov_cols else np.zeros((len(load_z), 0), dtype=float)

    beta_a, ya, xa = _lstsq_beta(pupil_z, load_z)
    a = float(beta_a[1]) if beta_a.size > 1 else float("nan")

    xb = np.column_stack([load_z, pupil_z, cov_use]) if cov_use.size else np.column_stack([load_z, pupil_z])
    beta_b, yb, xb_used = _lstsq_beta(p3_z, xb)
    c_prime = float(beta_b[1]) if beta_b.size > 1 else float("nan")
    b = float(beta_b[2]) if beta_b.size > 2 else float("nan")

    interaction = load_z * pupil_z
    xm = np.column_stack([load_z, pupil_z, interaction, cov_use]) if cov_use.size else np.column_stack([load_z, pupil_z, interaction])
    beta_m, ym, xm_used = _lstsq_beta(p3_z, xm)
    inter = float(beta_m[3]) if beta_m.size > 3 else float("nan")

    n_trials = int(min(len(ya), len(yb), len(ym)))
    return {
        "a": a,
        "b": b,
        "c_prime": c_prime,
        "ab": float(a * b) if np.isfinite(a) and np.isfinite(b) else float("nan"),
        "interaction": inter,
        "n_trials": float(n_trials),
    }


def _events_path_for_attrs(dataset_root: Path, attrs: Dict[str, Any]) -> Optional[Path]:
    subject = _safe_str(attrs.get("bids_subject") or attrs.get("subject")).strip()
    if not subject:
        return None
    session = _safe_str(attrs.get("bids_session") or attrs.get("session")).strip()
    task = _safe_str(attrs.get("task")).strip()
    run = _safe_str(attrs.get("bids_run") or attrs.get("run")).strip()

    sub_dir = dataset_root / f"sub-{subject}"
    if session and session.lower() not in {"na", "none"}:
        eeg_dir = sub_dir / f"ses-{session}" / "eeg"
    else:
        eeg_dir = sub_dir / "eeg"
    if not eeg_dir.exists():
        return None

    patterns: List[str] = []
    if task and run:
        patterns.extend(
            [
                f"*task-{task}*run-{run}*_events.tsv",
                f"*task-{task}*run-{run}*_events.tsv.gz",
            ]
        )
    if task:
        patterns.extend([f"*task-{task}*_events.tsv", f"*task-{task}*_events.tsv.gz"])
    patterns.extend([f"sub-{subject}*_events.tsv", f"sub-{subject}*_events.tsv.gz"])

    for pat in patterns:
        hits = sorted(eeg_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _match_load_from_text(
    text: str,
    regex: re.Pattern[str],
    *,
    use_denominator_as_load: bool,
    require_num_equals_den: bool,
) -> float:
    m = regex.search(str(text))
    if m is None:
        return float("nan")
    groups = [g for g in m.groups() if g is not None and str(g).strip() != ""]
    if not groups:
        return float("nan")
    try:
        if len(groups) >= 2:
            num = int(float(groups[0]))
            den = int(float(groups[1]))
            if require_num_equals_den and num != den:
                return float("nan")
            return float(den if use_denominator_as_load else num)
        return float(int(float(groups[0])))
    except Exception:
        return float("nan")


def _window_stats(
    *,
    onset_s: np.ndarray,
    time_s: np.ndarray,
    pupil: np.ndarray,
    baseline_w: Tuple[float, float],
    response_w: Tuple[float, float],
    min_samples: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    baseline = np.full(len(onset_s), np.nan, dtype=float)
    auc = np.full(len(onset_s), np.nan, dtype=float)

    for i, t0 in enumerate(np.asarray(onset_s, dtype=float)):
        if not np.isfinite(t0):
            continue
        bmask = (time_s >= t0 + baseline_w[0]) & (time_s <= t0 + baseline_w[1])
        rmask = (time_s >= t0 + response_w[0]) & (time_s <= t0 + response_w[1])
        if int(np.sum(bmask)) < min_samples or int(np.sum(rmask)) < min_samples:
            continue
        bvals = np.asarray(pupil[bmask], dtype=float)
        rvals = np.asarray(pupil[rmask], dtype=float)
        rt = np.asarray(time_s[rmask], dtype=float)
        bvals = bvals[np.isfinite(bvals)]
        keep = np.isfinite(rvals) & np.isfinite(rt)
        rvals = rvals[keep]
        rt = rt[keep]
        if bvals.size < min_samples or rvals.size < min_samples:
            continue
        b = float(np.mean(bvals))
        baseline[i] = b
        auc[i] = float(np.trapz(rvals - b, rt))

    return baseline, auc


def _collect_trials(
    *,
    features_root: Path,
    data_root: Path,
    dataset_id: str,
    mechanism_map: Dict[str, Any],
    config: Dict[str, Any],
    onset_tolerance: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    spec = (mechanism_map.get("datasets", {}) or {}).get(dataset_id, {})
    required_keys = ["event_filter", "load_column", "load_regex"]
    missing = [k for k in required_keys if not str(spec.get(k, "")).strip()]
    if missing:
        raise RuntimeError(
            f"{dataset_id}: mechanism event map missing required keys: {missing}. "
            f"Fix {spec} in configs/mechanism_event_map.yaml"
        )

    event_filter = str(spec["event_filter"])
    load_column = str(spec["load_column"])
    load_regex = re.compile(str(spec["load_regex"]))
    use_den = bool(spec.get("use_denominator_as_load", True))
    require_equal = bool(spec.get("require_num_equals_den", True))
    keep_task_contains = str(spec.get("task_contains", "memory")).strip().lower()

    dataset_root = data_root / dataset_id
    if not dataset_root.exists():
        raise RuntimeError(f"Dataset root missing: {dataset_root}")

    baseline_cfg = config.get("pupil", {}).get("baseline_s", [-0.2, 0.0])
    response_cfg = config.get("pupil", {}).get("response_s", [0.5, 2.5])
    baseline_w = (float(baseline_cfg[0]), float(baseline_cfg[1]))
    response_w = (float(response_cfg[0]), float(response_cfg[1]))

    rows: List[pd.DataFrame] = []
    errors: List[str] = []
    skip_counts: Dict[str, int] = {
        "missing_fields": 0,
        "events_missing": 0,
        "event_filter_zero": 0,
        "no_alignment": 0,
        "no_pupil": 0,
        "task_filtered": 0,
    }
    channel_counts: Dict[str, int] = {}

    for fp in iter_subject_feature_files(features_root):
        arrays, attrs = read_subject_h5(fp)
        ds = _safe_str(attrs.get("dataset_id")).strip()
        if ds != dataset_id:
            continue

        if not {"p3b_amp", "memory_load", "onset_s", "pdr"}.issubset(arrays.keys()):
            skip_counts["missing_fields"] += 1
            continue

        task = _safe_str(attrs.get("task")).strip().lower()
        if keep_task_contains and keep_task_contains not in task:
            skip_counts["task_filtered"] += 1
            continue

        events_path = _events_path_for_attrs(dataset_root, attrs)
        if events_path is None or not events_path.exists():
            skip_counts["events_missing"] += 1
            errors.append(f"{fp}: events file missing for attrs task={attrs.get('task')} run={attrs.get('run')}")
            continue

        try:
            ev = pd.read_csv(events_path, sep="\t")
        except Exception as exc:
            skip_counts["events_missing"] += 1
            errors.append(f"{fp}: failed to read events {events_path}: {exc}")
            continue

        if load_column not in ev.columns:
            skip_counts["missing_fields"] += 1
            errors.append(f"{fp}: load_column '{load_column}' missing from events columns={list(ev.columns)}")
            continue

        try:
            probe = ev.query(event_filter, engine="python").copy()
        except Exception as exc:
            skip_counts["event_filter_zero"] += 1
            errors.append(f"{fp}: event_filter failed: {exc}")
            continue

        if probe.empty:
            skip_counts["event_filter_zero"] += 1
            errors.append(f"{fp}: event_filter selected zero rows")
            continue

        probe["onset_s"] = pd.to_numeric(probe.get("onset"), errors="coerce")
        probe["memory_load"] = [
            _match_load_from_text(
                x,
                load_regex,
                use_denominator_as_load=use_den,
                require_num_equals_den=require_equal,
            )
            for x in probe[load_column].astype(str)
        ]
        probe = probe[np.isfinite(probe["onset_s"]) & np.isfinite(probe["memory_load"])].copy()
        if probe.empty:
            skip_counts["event_filter_zero"] += 1
            errors.append(f"{fp}: probe rows had no finite onset/load after regex parse")
            continue

        aligned = align_probe_events_to_epochs(
            epoch_onsets_s=np.asarray(arrays["onset_s"], dtype=float),
            probe_df=probe,
            tolerance_s=float(onset_tolerance),
        )
        matched = aligned[aligned["matched"]].copy()
        if matched.empty:
            skip_counts["no_alignment"] += 1
            errors.append(f"{fp}: no matched probe events to epochs")
            continue

        idx = matched["epoch_idx"].to_numpy(dtype=int)
        p3 = np.asarray(arrays["p3b_amp"], dtype=float)[idx]
        pdr_peak = np.asarray(arrays["pdr"], dtype=float)[idx]
        load = matched["memory_load"].to_numpy(dtype=float)
        onset = matched["probe_onset_s"].to_numpy(dtype=float)

        subj = _safe_str(attrs.get("bids_subject") or attrs.get("subject"))
        ses = _safe_str(attrs.get("bids_session") or attrs.get("session"))
        task_raw = _safe_str(attrs.get("task"))
        run = _safe_str(attrs.get("bids_run") or attrs.get("run"))
        skey = _safe_str(attrs.get("subject_key"))
        if not skey:
            skey = f"{dataset_id}:sub-{subj}" if not ses or ses.lower() in {"na", "none"} else f"{dataset_id}:sub-{subj}:ses-{ses}"

        eyetrack = find_eyetrack_file(
            bids_root=dataset_root,
            subject=subj,
            task=task_raw or None,
            run=run or None,
            session=ses or None,
        )
        if eyetrack is None:
            skip_counts["no_pupil"] += 1
            errors.append(f"{fp}: eyetrack file missing for subject={subj} task={task_raw} run={run} ses={ses}")
            continue

        try:
            t_s, pup = load_pupil_timeseries(eyetrack, config)
        except Exception as exc:
            skip_counts["no_pupil"] += 1
            errors.append(f"{fp}: failed loading eyetrack {eyetrack}: {exc}")
            continue

        baseline_pupil, pupil_auc = _window_stats(
            onset_s=onset,
            time_s=np.asarray(t_s, dtype=float),
            pupil=np.asarray(pup, dtype=float),
            baseline_w=baseline_w,
            response_w=response_w,
            min_samples=5,
        )

        p3_channel = np.asarray(arrays.get("p3b_channel", np.asarray(["unknown"] * len(idx))), dtype=str)[idx]
        for ch in p3_channel:
            channel_counts[str(ch)] = channel_counts.get(str(ch), 0) + 1

        frame = pd.DataFrame(
            {
                "dataset_id": dataset_id,
                "subject_key": skey,
                "p3_amp": p3,
                "memory_load": load,
                "pupil_peak": pdr_peak,
                "pupil_auc": baseline_pupil * 0.0 + pupil_auc,
                "pupil_baseline": baseline_pupil,
                "p3_channel": p3_channel,
                "events_path": str(events_path),
                "eyetrack_path": str(eyetrack),
            }
        )
        rows.append(frame)

    if not rows:
        raise RuntimeError(
            f"No usable ds003838 trial rows after fail-closed filtering. "
            f"skip_counts={skip_counts}; first_errors={errors[:5]}"
        )

    out = pd.concat(rows, axis=0, ignore_index=True)
    out = out[np.isfinite(out["p3_amp"]) & np.isfinite(out["memory_load"]) & np.isfinite(out["pupil_peak"])].copy()

    diag = {
        "n_trials": int(len(out)),
        "n_subjects": int(out["subject_key"].nunique()),
        "load_levels": sorted([float(x) for x in out["memory_load"].dropna().unique().tolist()]),
        "skip_counts": skip_counts,
        "errors_preview": errors[:50],
        "p3_channel_counts": channel_counts,
        "non_pz_trials": int(
            np.sum([str(x).strip().upper() != "PZ" for x in out.get("p3_channel", pd.Series([], dtype=str)).tolist()])
        ),
        "finite_rates": {
            "pupil_auc": float(np.isfinite(pd.to_numeric(out["pupil_auc"], errors="coerce")).mean()),
            "pupil_baseline": float(np.isfinite(pd.to_numeric(out["pupil_baseline"], errors="coerce")).mean()),
        },
    }
    return out.reset_index(drop=True), diag


def _subject_payloads(df: pd.DataFrame, min_trials: int) -> Tuple[List[SubjectPayload], pd.DataFrame]:
    payloads: List[SubjectPayload] = []
    subj_rows: List[Dict[str, Any]] = []

    for sid, g in df.groupby("subject_key"):
        g = g.copy().reset_index(drop=True)

        load = pd.to_numeric(g["memory_load"], errors="coerce").to_numpy(dtype=float)
        pupil = pd.to_numeric(g["pupil_peak"], errors="coerce").to_numpy(dtype=float)
        p3 = pd.to_numeric(g["p3_amp"], errors="coerce").to_numpy(dtype=float)
        base = pd.to_numeric(g.get("pupil_baseline", np.nan), errors="coerce").to_numpy(dtype=float)
        auc = pd.to_numeric(g.get("pupil_auc", np.nan), errors="coerce").to_numpy(dtype=float)

        keep = np.isfinite(load) & np.isfinite(pupil) & np.isfinite(p3)
        if int(np.sum(keep)) < int(min_trials):
            continue

        load = load[keep]
        pupil = pupil[keep]
        p3 = p3[keep]
        base = base[keep]
        auc = auc[keep]

        if np.unique(load[np.isfinite(load)]).size < 2:
            continue

        # Fill missing covariates subject-wise for stable regression design.
        if not np.isfinite(base).all():
            finite_base = base[np.isfinite(base)]
            fill = float(np.nanmean(finite_base)) if finite_base.size else 0.0
            base = np.where(np.isfinite(base), base, fill)
        if not np.isfinite(auc).all():
            finite_auc = auc[np.isfinite(auc)]
            fill = float(np.nanmean(finite_auc)) if finite_auc.size else 0.0
            auc = np.where(np.isfinite(auc), auc, fill)

        load_z = _zscore(load)
        pupil_z = _zscore(pupil)
        p3_z = _zscore(p3)
        base_z = _zscore(base)
        auc_z = _zscore(auc)
        covs = np.column_stack([base_z, auc_z])

        m = np.isfinite(load_z) & np.isfinite(pupil_z) & np.isfinite(p3_z)
        m = m & np.all(np.isfinite(covs), axis=1)
        if int(np.sum(m)) < int(min_trials):
            continue

        pl = SubjectPayload(
            subject_key=str(sid),
            n_trials=int(np.sum(m)),
            load_z=load_z[m],
            pupil_z=pupil_z[m],
            p3_z=p3_z[m],
            covs_z=covs[m, :],
        )
        eff = _effects_from_arrays(pl.load_z, pl.pupil_z, pl.p3_z, pl.covs_z)
        if not np.isfinite(eff.get("b", np.nan)):
            continue

        payloads.append(pl)
        subj_rows.append({"subject_key": pl.subject_key, **{k: float(v) for k, v in eff.items()}})

    return payloads, pd.DataFrame(subj_rows)


def _perm_null(
    payloads: Sequence[SubjectPayload],
    *,
    n_perm: int,
    seed: int,
    mode: str,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    metrics = ["a", "b", "c_prime", "ab", "interaction"]
    out = {k: np.full(int(n_perm), np.nan, dtype=float) for k in metrics}

    for i in range(int(n_perm)):
        vals: Dict[str, List[float]] = {k: [] for k in metrics}
        for s in payloads:
            if mode == "pupil_shuffle":
                idx = rng.permutation(len(s.pupil_z))
                load = s.load_z
                pupil = s.pupil_z[idx]
            elif mode == "load_shuffle":
                idx = rng.permutation(len(s.load_z))
                load = s.load_z[idx]
                pupil = s.pupil_z
            else:
                raise ValueError(f"Unknown permutation mode: {mode}")

            eff = _effects_from_arrays(load, pupil, s.p3_z, s.covs_z)
            for k in metrics:
                v = float(eff.get(k, np.nan))
                if np.isfinite(v):
                    vals[k].append(v)

        for k in metrics:
            if vals[k]:
                out[k][i] = float(np.median(np.asarray(vals[k], dtype=float)))
    return out


def _empirical_p(obs: float, null: np.ndarray) -> float:
    finite = np.asarray(null, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or not np.isfinite(obs):
        return float("nan")
    return float((1.0 + np.sum(np.abs(finite) >= abs(float(obs)))) / (1.0 + finite.size))


def _bootstrap_ci(values: Sequence[float], *, n_boot: int = 2000, seed: int = 0) -> List[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(int(seed))
    boots = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        smp = arr[rng.integers(0, arr.size, size=arr.size)]
        boots[i] = float(np.median(smp))
    return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def _run_seed_summary(seed: int, subject_df: pd.DataFrame) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    metrics = ["a", "b", "c_prime", "ab", "interaction"]
    if subject_df.empty:
        return {
            "seed": int(seed),
            "status": "SKIP",
            "reason": "empty subject_df",
        }

    idx = rng.integers(0, len(subject_df), size=len(subject_df))
    sampled = subject_df.iloc[idx].copy()

    effects = {}
    for m in metrics:
        vals = pd.to_numeric(sampled[m], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        effects[m] = float(np.median(vals)) if vals.size else float("nan")

    return {
        "seed": int(seed),
        "status": "PASS",
        "n_subjects_sampled": int(len(sampled)),
        "effects": effects,
    }


def _plot_load_vs_pupil(df: pd.DataFrame, out_path: Path) -> None:
    g = (
        df.groupby(["subject_key", "memory_load"], as_index=False)
        .agg(pupil_peak_mean=("pupil_peak", "mean"))
        .sort_values("memory_load")
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    loads = sorted([float(x) for x in g["memory_load"].dropna().unique().tolist()])
    for load in loads:
        y = g.loc[g["memory_load"] == load, "pupil_peak_mean"].to_numpy(dtype=float)
        x = np.full_like(y, fill_value=load, dtype=float)
        jitter = np.linspace(-0.12, 0.12, num=len(y), dtype=float) if len(y) > 1 else np.asarray([0.0])
        ax.scatter(x + jitter, y, alpha=0.45, s=14, color="#0b6e4f")
    mean_by_load = g.groupby("memory_load")["pupil_peak_mean"].mean().reset_index()
    ax.plot(
        mean_by_load["memory_load"].to_numpy(dtype=float),
        mean_by_load["pupil_peak_mean"].to_numpy(dtype=float),
        color="#ff7f11",
        linewidth=2.0,
        marker="o",
        label="Subject-mean pupil peak",
    )
    ax.set_xlabel("Memory load")
    ax.set_ylabel("Pupil peak (a.u.)")
    ax.set_title("Load vs Pupil Peak (ds003838)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _partial_resid(y: np.ndarray, x_cov: np.ndarray) -> np.ndarray:
    beta, yv, xv = _lstsq_beta(y, x_cov)
    if x_cov.ndim == 1:
        x_cov = x_cov[:, None]
    m = np.isfinite(y)
    for j in range(x_cov.shape[1]):
        m = m & np.isfinite(x_cov[:, j])
    out = np.full(len(y), np.nan, dtype=float)
    if not np.isfinite(beta).all():
        return out
    X = np.column_stack([np.ones(int(np.sum(m)), dtype=float), x_cov[m]])
    out[m] = y[m] - X @ beta
    return out


def _plot_partial(df: pd.DataFrame, out_path: Path) -> None:
    load = pd.to_numeric(df["memory_load"], errors="coerce").to_numpy(dtype=float)
    pupil = pd.to_numeric(df["pupil_peak"], errors="coerce").to_numpy(dtype=float)
    p3 = pd.to_numeric(df["p3_amp"], errors="coerce").to_numpy(dtype=float)
    cov = np.column_stack([
        load,
        pd.to_numeric(df.get("pupil_baseline", np.nan), errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df.get("pupil_auc", np.nan), errors="coerce").to_numpy(dtype=float),
    ])

    p_res = _partial_resid(pupil, cov)
    y_res = _partial_resid(p3, cov)
    m = np.isfinite(p_res) & np.isfinite(y_res)
    if int(np.sum(m)) == 0:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(p_res[m], y_res[m], s=10, alpha=0.3, color="#10454f")
    beta, _, _ = _lstsq_beta(y_res[m], p_res[m])
    if np.isfinite(beta).all() and beta.size >= 2:
        xx = np.linspace(float(np.nanmin(p_res[m])), float(np.nanmax(p_res[m])), 120)
        yy = beta[0] + beta[1] * xx
        ax.plot(xx, yy, color="#b33f62", linewidth=2.0)
    ax.set_xlabel("Pupil residual (load/covariates removed)")
    ax.set_ylabel("P3 residual (load/covariates removed)")
    ax.set_title("Partial Pupil-P3 Relationship")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_mediation_ab(observed_ab: float, null_ab: np.ndarray, out_path: Path) -> None:
    finite = np.asarray(null_ab, dtype=float)
    finite = finite[np.isfinite(finite)]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    if finite.size > 0:
        ax.hist(finite, bins=60, alpha=0.7, color="#6c8ead", density=True, label="Permutation null (pupil shuffle)")
    ax.axvline(observed_ab, color="#d62828", linewidth=2.0, label=f"Observed ab={observed_ab:.4g}")
    ax.set_xlabel("Median mediation effect (ab)")
    ax.set_ylabel("Density")
    ax.set_title("Mediation Effect Null Overlay")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_seed_summary(seed_rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not seed_rows:
        return
    metrics = ["a", "b", "c_prime", "ab", "interaction"]
    seeds = [int(r["seed"]) for r in seed_rows]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    axes = axes.ravel()
    for i, m in enumerate(metrics):
        vals = np.asarray([float((r.get("effects") or {}).get(m, np.nan)) for r in seed_rows], dtype=float)
        ax = axes[i]
        ax.plot(seeds, vals, color="#155e63", linewidth=1.2)
        ax.scatter(seeds, vals, color="#0b0f0f", s=12)
        ax.set_title(m)
        ax.set_xlabel("Seed")
        ax.grid(alpha=0.2)
    axes[-1].axis("off")
    fig.suptitle("Mechanism Seed Stability (0-49)", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mechanism_map = yaml.safe_load(args.event_map.read_text(encoding="utf-8")) if args.event_map.exists() else {}
    if not isinstance(mechanism_map, dict):
        raise RuntimeError(f"Invalid mechanism event map: {args.event_map}")

    config = load_yaml(args.config)

    trial_df, collect_diag = _collect_trials(
        features_root=args.features_root,
        data_root=args.data_root,
        dataset_id=args.dataset_id,
        mechanism_map=mechanism_map,
        config=config,
        onset_tolerance=float(args.onset_tolerance),
    )

    trial_csv = args.out_dir / "mechanism_trials.csv"
    trial_df.to_csv(trial_csv, index=False)

    payloads, subject_df = _subject_payloads(trial_df, min_trials=int(args.min_trials))
    if subject_df.empty or not payloads:
        raise RuntimeError(
            f"No subjects passed mechanism model gate (min_trials={args.min_trials}). "
            f"collect_diag={collect_diag}"
        )

    subject_csv = args.out_dir / "subject_effects.csv"
    subject_df.to_csv(subject_csv, index=False)

    metrics = ["a", "b", "c_prime", "ab", "interaction"]
    observed = {
        m: float(np.nanmedian(pd.to_numeric(subject_df[m], errors="coerce").to_numpy(dtype=float)))
        for m in metrics
    }
    ci95 = {
        m: _bootstrap_ci(
            pd.to_numeric(subject_df[m], errors="coerce").to_numpy(dtype=float),
            n_boot=2000,
            seed=int(args.seed) + 17 + i,
        )
        for i, m in enumerate(metrics)
    }

    null_pupil = _perm_null(payloads, n_perm=int(args.n_perm), seed=int(args.seed), mode="pupil_shuffle")
    null_load = _perm_null(payloads, n_perm=int(args.n_perm), seed=int(args.seed) + 1_000_003, mode="load_shuffle")

    pvals = {m: _empirical_p(observed[m], null_pupil[m]) for m in metrics}

    primary = ["b", "ab", "interaction"]
    qvals_primary = bh_fdr([pvals[m] for m in primary])
    qvals = {m: float("nan") for m in metrics}
    for m, q in zip(primary, qvals_primary):
        qvals[m] = float(q)

    # Multi-seed stability summaries.
    seeds = _parse_seeds(args.seeds)
    seed_rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.parallel_procs))) as ex:
        futs = {ex.submit(_run_seed_summary, int(seed), subject_df): int(seed) for seed in seeds}
        for fut in as_completed(futs):
            seed = futs[fut]
            res = fut.result()
            seed_rows.append(res)
            seed_dir = args.out_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    seed_rows = sorted(seed_rows, key=lambda x: int(x.get("seed", 0)))

    aggregate_metrics: Dict[str, Any] = {}
    for m in metrics:
        vals = np.asarray([float((r.get("effects") or {}).get(m, np.nan)) for r in seed_rows], dtype=float)
        finite = vals[np.isfinite(vals)]
        aggregate_metrics[m] = {
            "n_seeds": int(finite.size),
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "ci95": [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))] if finite.size else [float("nan"), float("nan")],
            "worst_min": float(np.min(finite)) if finite.size else float("nan"),
            "worst_max": float(np.max(finite)) if finite.size else float("nan"),
        }

    table_rows: List[Dict[str, Any]] = []
    for m in metrics:
        n_pupil = np.asarray(null_pupil[m], dtype=float)
        n_pupil = n_pupil[np.isfinite(n_pupil)]
        n_load = np.asarray(null_load[m], dtype=float)
        n_load = n_load[np.isfinite(n_load)]
        table_rows.append(
            {
                "metric": m,
                "observed_median": observed[m],
                "ci95_low": ci95[m][0],
                "ci95_high": ci95[m][1],
                "p_value": pvals[m],
                "q_value": qvals[m],
                "null_pupil_median": float(np.median(n_pupil)) if n_pupil.size else float("nan"),
                "null_load_median": float(np.median(n_load)) if n_load.size else float("nan"),
                "control_pupil_degrade": bool(abs(float(np.median(n_pupil))) < abs(float(observed[m]))) if n_pupil.size and np.isfinite(observed[m]) else False,
                "control_load_degrade": bool(abs(float(np.median(n_load))) < abs(float(observed[m]))) if n_load.size and np.isfinite(observed[m]) else False,
            }
        )

    table_df = pd.DataFrame(table_rows)
    table_csv = args.out_dir / "Table_mechanism_effects.csv"
    table_df.to_csv(table_csv, index=False)

    _plot_load_vs_pupil(trial_df, args.out_dir / "FIG_load_vs_pupil.png")
    _plot_partial(trial_df, args.out_dir / "FIG_pupil_vs_p3_partial.png")
    _plot_mediation_ab(observed_ab=float(observed["ab"]), null_ab=null_pupil["ab"], out_path=args.out_dir / "FIG_mediation_ab.png")
    _plot_seed_summary(seed_rows, args.out_dir / "FIG_mechanism_summary.png")

    aggregate_payload = {
        "dataset_id": args.dataset_id,
        "features_root": str(args.features_root),
        "data_root": str(args.data_root),
        "event_map": str(args.event_map),
        "n_perm": int(args.n_perm),
        "n_subjects_modeled": int(subject_df["subject_key"].nunique()),
        "n_trials_modeled": int(len(trial_df)),
        "collect_diagnostics": collect_diag,
        "observed_medians": observed,
        "observed_ci95": ci95,
        "p_values": pvals,
        "q_values": qvals,
        "null_summaries": {
            "pupil_shuffle": {
                m: {
                    "median": float(np.nanmedian(null_pupil[m])) if np.isfinite(null_pupil[m]).any() else float("nan"),
                    "ci95": [
                        float(np.nanquantile(null_pupil[m], 0.025)) if np.isfinite(null_pupil[m]).any() else float("nan"),
                        float(np.nanquantile(null_pupil[m], 0.975)) if np.isfinite(null_pupil[m]).any() else float("nan"),
                    ],
                }
                for m in metrics
            },
            "load_shuffle": {
                m: {
                    "median": float(np.nanmedian(null_load[m])) if np.isfinite(null_load[m]).any() else float("nan"),
                    "ci95": [
                        float(np.nanquantile(null_load[m], 0.025)) if np.isfinite(null_load[m]).any() else float("nan"),
                        float(np.nanquantile(null_load[m], 0.975)) if np.isfinite(null_load[m]).any() else float("nan"),
                    ],
                }
                for m in metrics
            },
        },
        "seed_stability": {
            "seeds": seeds,
            "aggregate_metrics": aggregate_metrics,
            "rows": seed_rows,
        },
        "artifacts": {
            "trial_csv": str(trial_csv),
            "subject_csv": str(subject_csv),
            "table_csv": str(table_csv),
            "figures": [
                str(args.out_dir / "FIG_load_vs_pupil.png"),
                str(args.out_dir / "FIG_pupil_vs_p3_partial.png"),
                str(args.out_dir / "FIG_mediation_ab.png"),
                str(args.out_dir / "FIG_mechanism_summary.png"),
            ],
        },
    }

    out_json = args.out_dir / "aggregate_mechanism.json"
    out_json.write_text(json.dumps(aggregate_payload, indent=2), encoding="utf-8")

    print(f"MECHANISM_AGGREGATE={out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
