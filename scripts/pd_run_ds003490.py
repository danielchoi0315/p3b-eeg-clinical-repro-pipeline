#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.multitest import multipletests


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_subject(v: object) -> str:
    s = str(v).strip()
    return s[4:] if s.startswith("sub-") else s


def _norm(v: object) -> str:
    return str(v).strip().lower()


def _encode_sex(series: pd.Series) -> np.ndarray:
    vals = series.fillna("").astype(str).str.strip().str.lower()
    out = np.full(len(vals), np.nan, dtype=float)
    for i, v in enumerate(vals):
        if v in {"m", "male", "1"}:
            out[i] = 1.0
        elif v in {"f", "female", "0"}:
            out[i] = 0.0
    return out


def _bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray, *, n_boot: int = 2000, seed: int = 0) -> Tuple[float, List[float]]:
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


def _robust_group_beta(df: pd.DataFrame, score_col: str, group_col: str) -> Tuple[float, int]:
    work = df.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work[group_col] = pd.to_numeric(work[group_col], errors="coerce")
    work["age"] = pd.to_numeric(work.get("age"), errors="coerce")
    work["sex_num"] = _encode_sex(work.get("sex", pd.Series([""] * len(work))))

    fit = work[[score_col, group_col, "age", "sex_num"]].dropna().copy()
    if len(fit) < 8 or fit[group_col].nunique() < 2:
        return float("nan"), int(len(fit))

    y = fit[score_col].astype(float)
    X = sm.add_constant(fit[[group_col, "age", "sex_num"]].astype(float), has_constant="add")
    res = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    return float(res.params.get(group_col, np.nan)), int(len(fit))


def _perm_p_group_beta(df: pd.DataFrame, score_col: str, group_col: str, *, n_perm: int, seed: int) -> float:
    obs, n_fit = _robust_group_beta(df, score_col, group_col)
    if not np.isfinite(obs) or n_fit <= 0:
        return float("nan")

    rng = np.random.default_rng(int(seed))
    arr = pd.to_numeric(df[group_col], errors="coerce").to_numpy(dtype=float)
    null = np.full(int(n_perm), np.nan, dtype=float)
    for i in range(int(n_perm)):
        perm = arr.copy()
        rng.shuffle(perm)
        tmp = df.copy()
        tmp[group_col] = perm
        b, _ = _robust_group_beta(tmp, score_col, group_col)
        null[i] = b

    finite = null[np.isfinite(null)]
    if finite.size == 0:
        return float("nan")
    return float((1.0 + np.sum(np.abs(finite) >= abs(obs))) / (1.0 + finite.size))


def _choose_channel(raw: mne.io.BaseRaw) -> Tuple[str, bool]:
    names = list(raw.ch_names)
    lower_to_name = {n.lower(): n for n in names}
    if "pz" in lower_to_name:
        return lower_to_name["pz"], False

    for c in ["cpz", "poz", "cz", "p3", "p4", "oz", "cp1", "cp2"]:
        if c in lower_to_name:
            return lower_to_name[c], True
    return names[0], True


def _session_from_name(path: Path) -> str:
    m = re.search(r"ses-([A-Za-z0-9]+)", str(path))
    if not m:
        return "01"
    return m.group(1)


def _subject_from_name(path: Path) -> str:
    m = re.search(r"sub-([A-Za-z0-9]+)", str(path))
    return m.group(1) if m else ""


def _load_raw(path: Path):
    suf = path.suffix.lower()
    if suf == ".set":
        return mne.io.read_raw_eeglab(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".edf":
        return mne.io.read_raw_edf(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".bdf":
        return mne.io.read_raw_bdf(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".vhdr":
        return mne.io.read_raw_brainvision(path.as_posix(), preload=True, verbose="ERROR")
    if suf == ".fif":
        return mne.io.read_raw_fif(path.as_posix(), preload=True, verbose="ERROR")
    raise RuntimeError(f"Unsupported EEG file suffix: {path.suffix}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--mapping_json", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_perm", type=int, default=20000)
    ap.add_argument("--stop_reason", type=Path, required=True)
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_reason: Path = args.stop_reason

    for stale in [
        out_dir / "pd_deviation_scores.csv",
        out_dir / "pd_endpoints.csv",
        out_dir / "FIG_pd_primary_auc_roc.png",
        out_dir / "FIG_pd_deviation_by_group_medstate.png",
        out_dir / "FIG_pd_on_off_paired.png",
    ]:
        if stale.exists():
            stale.unlink()

    summary_json = out_dir / "pd_run_summary.json"
    mapping = json.loads(args.mapping_json.read_text(encoding="utf-8")) if args.mapping_json.exists() else {}
    if str(mapping.get("status", "")).upper() != "PASS":
        reason = "Mapping is not PASS; refusing to run PD extraction"
        _write_text(stop_reason, f"# STOP_REASON\n\n{reason}\n")
        _write_json(summary_json, {"status": "SKIP", "reason": reason})
        return 0

    col = str(mapping.get("column", ""))
    std_vals = {_norm(x) for x in mapping.get("stim_standard_values", [])}
    tgt_vals = {_norm(x) for x in mapping.get("stim_target_values", [])}
    nov_vals = {_norm(x) for x in mapping.get("stim_novel_values", [])}
    if not col or not std_vals or not tgt_vals or not nov_vals:
        reason = "Mapping payload missing required column/value sets"
        _write_text(stop_reason, f"# STOP_REASON\n\n{reason}\n")
        _write_json(summary_json, {"status": "SKIP", "reason": reason})
        return 0

    participants_tsv = args.dataset_root / "participants.tsv"
    if not participants_tsv.exists():
        reason = f"Missing participants.tsv at {participants_tsv}"
        _write_text(stop_reason, f"# STOP_REASON\n\n{reason}\n")
        _write_json(summary_json, {"status": "SKIP", "reason": reason})
        return 0

    part = pd.read_csv(participants_tsv, sep="\t")
    part["subject_id"] = part["participant_id"].map(_safe_subject)
    part["group"] = part["Group"].astype(str).str.upper()
    part["age"] = pd.to_numeric(part["age"], errors="coerce")
    part["sex"] = part["sex"].astype(str)

    meta: Dict[str, dict] = {}
    for _, r in part.iterrows():
        sid = str(r["subject_id"])
        meta[sid] = {
            "group": str(r.get("group", "UNK")),
            "age": float(r.get("age", np.nan)),
            "sex": str(r.get("sex", "")),
            "sess1_med": str(r.get("sess1_Med", "")).strip().upper(),
            "sess2_med": str(r.get("sess2_Med", "")).strip().upper(),
        }

    eeg_files = sorted(args.dataset_root.rglob("*_eeg.set"))
    if not eeg_files:
        reason = "No EEG .set files found"
        _write_text(stop_reason, f"# STOP_REASON\n\n{reason}\n")
        _write_json(summary_json, {"status": "SKIP", "reason": reason})
        return 0

    rows: List[dict] = []
    skipped_runs: List[str] = []

    for eeg in eeg_files:
        ev = eeg.with_name(eeg.name.replace("_eeg.set", "_events.tsv"))
        if not ev.exists():
            skipped_runs.append(f"{eeg}: missing events")
            continue
        sid = _subject_from_name(eeg)
        ses = _session_from_name(eeg)
        if not sid:
            skipped_runs.append(f"{eeg}: missing subject id")
            continue

        try:
            df = pd.read_csv(ev, sep="\t")
        except Exception as exc:
            skipped_runs.append(f"{eeg}: cannot read events ({exc})")
            continue

        if col not in df.columns or "onset" not in df.columns:
            skipped_runs.append(f"{eeg}: events missing required columns")
            continue

        vals = df[col].astype(str).map(_norm)
        cls = np.full(len(df), "", dtype=object)
        cls[np.isin(vals, list(std_vals))] = "standard"
        cls[np.isin(vals, list(tgt_vals))] = "target"
        cls[np.isin(vals, list(nov_vals))] = "novel"

        dfx = df.copy()
        dfx["_class"] = cls
        dfx["onset"] = pd.to_numeric(dfx["onset"], errors="coerce")
        dfx = dfx[np.isfinite(dfx["onset"])].copy()

        n_std = int((dfx["_class"] == "standard").sum())
        n_tgt = int((dfx["_class"] == "target").sum())
        n_nov = int((dfx["_class"] == "novel").sum())
        if min(n_std, n_tgt, n_nov) <= 0:
            skipped_runs.append(f"{eeg}: missing one oddball class")
            continue

        try:
            raw = _load_raw(eeg)
            raw.pick_types(eeg=True, eog=False, misc=False, stim=False)
            if len(raw.ch_names) == 0:
                skipped_runs.append(f"{eeg}: no EEG channels after pick")
                continue
            ch, fallback = _choose_channel(raw)
            sf = float(raw.info["sfreq"])

            events_np = []
            ev_id = {"standard": 1, "target": 2, "novel": 3}
            for lab, code in ev_id.items():
                on = pd.to_numeric(dfx.loc[dfx["_class"] == lab, "onset"], errors="coerce").to_numpy(dtype=float)
                on = on[np.isfinite(on)]
                smp = np.asarray(np.round(on * sf), dtype=int)
                smp = smp[(smp > 0) & (smp < raw.n_times)]
                for s in smp:
                    events_np.append([int(s), 0, int(code)])
            if not events_np:
                skipped_runs.append(f"{eeg}: no valid events after conversion")
                continue

            events_np = np.asarray(sorted(events_np, key=lambda x: x[0]), dtype=int)
            picks = mne.pick_channels(raw.ch_names, include=[ch])
            epochs = mne.Epochs(
                raw,
                events_np,
                event_id=ev_id,
                tmin=-0.2,
                tmax=0.8,
                baseline=(-0.2, 0.0),
                preload=True,
                picks=picks,
                reject_by_annotation=False,
                verbose="ERROR",
            )

            if len(epochs["target"]) == 0 or len(epochs["standard"]) == 0:
                skipped_runs.append(f"{eeg}: empty target/standard epochs")
                continue

            times = epochs.times
            idx_primary = (times >= 0.35) & (times <= 0.60)
            idx_explore = (times >= 0.30) & (times <= 0.70)
            if int(idx_primary.sum()) < 2:
                skipped_runs.append(f"{eeg}: primary window has <2 samples")
                continue

            tar = epochs["target"].get_data(copy=True)[:, 0, :]
            std = epochs["standard"].get_data(copy=True)[:, 0, :]
            nov = epochs["novel"].get_data(copy=True)[:, 0, :] if len(epochs["novel"]) else np.zeros_like(tar[:1])

            tar_mean = np.nanmean(tar, axis=0) * 1e6
            std_mean = np.nanmean(std, axis=0) * 1e6
            nov_mean = np.nanmean(nov, axis=0) * 1e6 if nov.size else np.full_like(tar_mean, np.nan)

            p3_target = float(np.nanmean(tar_mean[idx_primary]))
            p3_standard = float(np.nanmean(std_mean[idx_primary]))
            p3_delta = float(p3_target - p3_standard)

            if int(idx_explore.sum()) > 2:
                seg = tar_mean[idx_explore]
                seg_t = times[idx_explore]
                pk_idx = int(np.nanargmax(seg))
                p3_peak = float(seg[pk_idx])
                p3_lat = float(seg_t[pk_idx])
            else:
                p3_peak = float("nan")
                p3_lat = float("nan")

            rt_mean = float(pd.to_numeric(dfx.loc[dfx["_class"] == "target", "response_time"], errors="coerce").mean()) if "response_time" in dfx.columns else float("nan")
            acc = float(pd.to_numeric(dfx.loc[dfx["_class"] == "target", "response_time"], errors="coerce").notna().mean()) if "response_time" in dfx.columns else float("nan")

            meta_row = meta.get(sid, {})
            grp = str(meta_row.get("group", "UNK"))
            if grp == "CTL":
                med_state = "CTL"
            elif grp == "PD":
                if ses == "01":
                    med_state = str(meta_row.get("sess1_med", ""))
                elif ses == "02":
                    med_state = str(meta_row.get("sess2_med", ""))
                else:
                    med_state = ""
            else:
                med_state = ""

            rows.append(
                {
                    "subject_id": sid,
                    "session": ses,
                    "group": grp,
                    "med_state": med_state,
                    "age": float(meta_row.get("age", np.nan)),
                    "sex": str(meta_row.get("sex", "")),
                    "eeg_file": str(eeg),
                    "events_file": str(ev),
                    "p3_channel": ch,
                    "fallback_channel_used": bool(fallback),
                    "n_standard": int(len(epochs["standard"])),
                    "n_target": int(len(epochs["target"])),
                    "n_novel": int(len(epochs["novel"])),
                    "p3_amp_uV": p3_target,
                    "p3_amp_standard_uV": p3_standard,
                    "target_minus_standard_uV": p3_delta,
                    "p3_peak_030_070_uV": p3_peak,
                    "p3_lat_030_070_s": p3_lat,
                    "rt_target_s": rt_mean,
                    "accuracy_target": acc,
                }
            )
        except Exception as exc:
            skipped_runs.append(f"{eeg}: {exc}")
        finally:
            try:
                raw.close()  # type: ignore[name-defined]
            except Exception:
                pass

    run_df = pd.DataFrame(rows)
    if run_df.empty:
        reason = "No usable oddball runs after fail-closed filtering"
        _write_text(
            stop_reason,
            "\n".join([
                "# STOP_REASON",
                "",
                reason,
                "",
                "## Sample skipped runs",
                *[f"- {x}" for x in skipped_runs[:80]],
            ])
            + "\n",
        )
        _write_json(summary_json, {"status": "SKIP", "reason": reason, "n_skipped_runs": len(skipped_runs)})
        return 0

    # Subject/session rows (single-run in this dataset; keep robust if multiple).
    agg = (
        run_df.groupby(["subject_id", "session", "group", "med_state", "age", "sex"], as_index=False)
        .agg(
            p3_amp_uV=("p3_amp_uV", "mean"),
            target_minus_standard_uV=("target_minus_standard_uV", "mean"),
            p3_peak_030_070_uV=("p3_peak_030_070_uV", "mean"),
            p3_lat_030_070_s=("p3_lat_030_070_s", "mean"),
            rt_target_s=("rt_target_s", "mean"),
            accuracy_target=("accuracy_target", "mean"),
            n_target=("n_target", "sum"),
            fallback_channel_used=("fallback_channel_used", "max"),
        )
        .copy()
    )

    # Normative model on controls only.
    ctrl = agg[(agg["group"] == "CTL") & np.isfinite(pd.to_numeric(agg["p3_amp_uV"], errors="coerce"))].copy()
    if ctrl.empty:
        reason = "No control sessions for normative model"
        _write_text(stop_reason, f"# STOP_REASON\n\n{reason}\n")
        _write_json(summary_json, {"status": "SKIP", "reason": reason})
        return 0

    agg["age"] = pd.to_numeric(agg["age"], errors="coerce")
    agg["sex_num"] = _encode_sex(agg["sex"])
    ctrl["age"] = pd.to_numeric(ctrl["age"], errors="coerce")
    ctrl["sex_num"] = _encode_sex(ctrl["sex"])

    fit = ctrl[["p3_amp_uV", "age", "sex_num"]].dropna().copy()
    if len(fit) >= 10:
        y = fit["p3_amp_uV"].astype(float)
        X = sm.add_constant(fit[["age", "sex_num"]].astype(float), has_constant="add")
        res = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()

        X_eval = sm.add_constant(agg[["age", "sex_num"]].astype(float), has_constant="add")
        pred = np.asarray(res.predict(X_eval), dtype=float)

        X_ctrl = sm.add_constant(ctrl[["age", "sex_num"]].astype(float), has_constant="add")
        pred_ctrl = np.asarray(res.predict(X_ctrl), dtype=float)
        resid_ctrl = pd.to_numeric(ctrl["p3_amp_uV"], errors="coerce").to_numpy(dtype=float) - pred_ctrl
    else:
        mu = float(pd.to_numeric(ctrl["p3_amp_uV"], errors="coerce").mean())
        pred = np.full(len(agg), mu, dtype=float)
        resid_ctrl = pd.to_numeric(ctrl["p3_amp_uV"], errors="coerce").to_numpy(dtype=float) - mu

    resid_ctrl = resid_ctrl[np.isfinite(resid_ctrl)]
    if resid_ctrl.size == 0:
        sigma = 1.0
    else:
        mad = float(np.median(np.abs(resid_ctrl - np.median(resid_ctrl))))
        sigma = float(max(1e-6, 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.std(resid_ctrl)))
        if not np.isfinite(sigma) or sigma <= 1e-6:
            sigma = 1.0

    agg["pred_p3_amp_uV"] = pred
    agg["deviation_z"] = (pd.to_numeric(agg["p3_amp_uV"], errors="coerce").to_numpy(dtype=float) - pred) / sigma

    # Orientation so larger score corresponds to PD burden.
    pd_off = agg[(agg["group"] == "PD") & (agg["med_state"].astype(str).str.upper() == "OFF")]
    ctl = agg[agg["group"] == "CTL"]
    if len(pd_off) > 0 and len(ctl) > 0:
        orient = 1.0 if np.nanmedian(pd_off["deviation_z"].to_numpy(dtype=float)) >= np.nanmedian(ctl["deviation_z"].to_numpy(dtype=float)) else -1.0
    else:
        pd_any = agg[agg["group"] == "PD"]
        orient = 1.0 if (len(pd_any) > 0 and np.nanmedian(pd_any["deviation_z"].to_numpy(dtype=float)) >= np.nanmedian(ctl["deviation_z"].to_numpy(dtype=float))) else -1.0
    agg["deviation_oriented"] = orient * pd.to_numeric(agg["deviation_z"], errors="coerce")

    # Secondary deviation based on target-minus-standard.
    ctrl_delta = pd.to_numeric(ctrl["target_minus_standard_uV"], errors="coerce")
    mu_delta = float(ctrl_delta.mean()) if np.isfinite(ctrl_delta).any() else 0.0
    sd_delta = float(ctrl_delta.std()) if np.isfinite(ctrl_delta).any() else 1.0
    if not np.isfinite(sd_delta) or sd_delta <= 1e-6:
        sd_delta = 1.0
    agg["delta_deviation_z"] = (pd.to_numeric(agg["target_minus_standard_uV"], errors="coerce").to_numpy(dtype=float) - mu_delta) / sd_delta
    agg["delta_oriented"] = orient * pd.to_numeric(agg["delta_deviation_z"], errors="coerce")

    # Endpoints.
    endpoints: List[dict] = []
    n_perm = int(max(20000, int(args.n_perm)))

    def add_auc_endpoint(name: str, score_col: str, pd_mask: pd.Series, seed_base: int) -> Optional[pd.DataFrame]:
        sub = agg[((agg["group"] == "CTL") | pd_mask)].copy()
        sub["y"] = (sub["group"] == "PD").astype(int)
        sub["s"] = pd.to_numeric(sub[score_col], errors="coerce")
        sub = sub[np.isfinite(sub["s"])].copy()
        if sub.empty or sub["y"].nunique() < 2:
            return None
        y = sub["y"].to_numpy(dtype=int)
        s = sub["s"].to_numpy(dtype=float)
        auc, ci = _bootstrap_auc(y, s, n_boot=3000, seed=seed_base + 1)
        p = _perm_p_auc(y, s, n_perm=n_perm, seed=seed_base + 2)
        endpoints.append(
            {
                "endpoint": name,
                "type": "auc",
                "n": int(len(sub)),
                "estimate": float(auc),
                "ci95_lo": float(ci[0]),
                "ci95_hi": float(ci[1]),
                "perm_p": float(p),
                "score": score_col,
            }
        )
        return sub

    off_mask = (agg["group"] == "PD") & (agg["med_state"].astype(str).str.upper() == "OFF")
    on_mask = (agg["group"] == "PD") & (agg["med_state"].astype(str).str.upper() == "ON")

    auc_sub = add_auc_endpoint("AUC_PD_OFF_vs_CTL_primary", "deviation_oriented", off_mask, 100)
    add_auc_endpoint("AUC_PD_ON_vs_CTL_primary", "deviation_oriented", on_mask, 200)
    add_auc_endpoint("AUC_PD_OFF_vs_CTL_delta", "delta_oriented", off_mask, 300)

    if auc_sub is not None:
        model_df = auc_sub.copy()
        model_df["group_bin"] = model_df["y"].astype(float)
        b, n_fit = _robust_group_beta(model_df, "s", "group_bin")
        p_b = _perm_p_group_beta(model_df, "s", "group_bin", n_perm=n_perm, seed=401)
        endpoints.append(
            {
                "endpoint": "RobustBeta_PD_OFF_vs_CTL_primary",
                "type": "robust_beta",
                "n": int(n_fit),
                "estimate": float(b),
                "ci95_lo": float("nan"),
                "ci95_hi": float("nan"),
                "perm_p": float(p_b),
                "score": "deviation_oriented",
            }
        )

    # Paired ON/OFF.
    pairs = []
    for sid, g in agg[agg["group"] == "PD"].groupby("subject_id"):
        off = g[g["med_state"].astype(str).str.upper() == "OFF"]
        on = g[g["med_state"].astype(str).str.upper() == "ON"]
        if len(off) and len(on):
            pairs.append({"subject_id": sid, "off": float(off.iloc[0]["deviation_oriented"]), "on": float(on.iloc[0]["deviation_oriented"])})

    pair_df = pd.DataFrame(pairs)
    if not pair_df.empty:
        diff = pair_df["on"].to_numpy(dtype=float) - pair_df["off"].to_numpy(dtype=float)
        obs = float(np.nanmean(diff))
        rng = np.random.default_rng(777)
        null = np.full(n_perm, np.nan, dtype=float)
        for i in range(n_perm):
            signs = rng.choice([-1.0, 1.0], size=len(diff))
            null[i] = float(np.nanmean(diff * signs))
        finite = null[np.isfinite(null)]
        p_pair = float((1.0 + np.sum(np.abs(finite) >= abs(obs))) / (1.0 + len(finite))) if finite.size else float("nan")
        endpoints.append(
            {
                "endpoint": "Paired_PD_ON_minus_OFF_primary",
                "type": "paired_signflip",
                "n": int(len(pair_df)),
                "estimate": float(obs),
                "ci95_lo": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
                "ci95_hi": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
                "perm_p": float(p_pair),
                "score": "deviation_oriented",
            }
        )
        pair_df.to_csv(out_dir / "pd_on_off_pairs.csv", index=False)

    end_df = pd.DataFrame(endpoints)
    if end_df.empty:
        reason = "No PD endpoints could be computed"
        _write_text(stop_reason, f"# STOP_REASON\n\n{reason}\n")
        _write_json(summary_json, {"status": "SKIP", "reason": reason})
        return 0

    pvals = pd.to_numeric(end_df["perm_p"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    qvals = multipletests(pvals, alpha=0.05, method="fdr_bh")[1]
    end_df["perm_q"] = qvals
    if not np.all(np.isfinite(pd.to_numeric(end_df["perm_q"], errors="coerce").to_numpy(dtype=float))):
        _write_json(summary_json, {"status": "FAIL", "reason": "NaN q-values in PD endpoints"})
        return 1

    agg.to_csv(out_dir / "pd_deviation_scores.csv", index=False)
    end_df.to_csv(out_dir / "pd_endpoints.csv", index=False)

    # Figures.
    fig_auc = out_dir / "FIG_pd_primary_auc_roc.png"
    fig_grp = out_dir / "FIG_pd_deviation_by_group_medstate.png"
    fig_pair = out_dir / "FIG_pd_on_off_paired.png"

    try:
        off_vs_ctl = agg[((agg["group"] == "CTL") | off_mask)].copy()
        off_vs_ctl["y"] = (off_vs_ctl["group"] == "PD").astype(int)
        off_vs_ctl["s"] = pd.to_numeric(off_vs_ctl["deviation_oriented"], errors="coerce")
        off_vs_ctl = off_vs_ctl[np.isfinite(off_vs_ctl["s"])].copy()
        if len(off_vs_ctl) > 0 and off_vs_ctl["y"].nunique() == 2:
            fpr, tpr, _ = roc_curve(off_vs_ctl["y"].to_numpy(dtype=int), off_vs_ctl["s"].to_numpy(dtype=float))
            auc = float(roc_auc_score(off_vs_ctl["y"].to_numpy(dtype=int), off_vs_ctl["s"].to_numpy(dtype=float)))
            fig, ax = plt.subplots(figsize=(5.2, 5.2))
            ax.plot(fpr, tpr, color="#224b8f", linewidth=2.0, label=f"AUC={auc:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("PD OFF vs CTL ROC (primary)")
            ax.legend(frameon=False)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(fig_auc, dpi=160)
            plt.close(fig)
    except Exception:
        pass

    try:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        plot_rows = []
        for _, r in agg.iterrows():
            g = str(r.get("group", "UNK"))
            m = str(r.get("med_state", "")).upper()
            if g == "CTL":
                lab = "CTL"
            elif g == "PD" and m == "OFF":
                lab = "PD OFF"
            elif g == "PD" and m == "ON":
                lab = "PD ON"
            else:
                lab = "PD/UNK"
            plot_rows.append({"label": lab, "z": float(r.get("deviation_oriented", np.nan))})
        pdf = pd.DataFrame(plot_rows)
        order = ["CTL", "PD OFF", "PD ON", "PD/UNK"]
        for i, lab in enumerate(order, start=1):
            vals = pd.to_numeric(pdf.loc[pdf["label"] == lab, "z"], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            x = np.full(vals.shape, i, dtype=float)
            jit = np.linspace(-0.15, 0.15, num=len(vals), dtype=float) if len(vals) > 1 else np.asarray([0.0])
            ax.scatter(x + jit, vals, alpha=0.55, s=14)
            ax.hlines(float(np.median(vals)), i - 0.2, i + 0.2, color="#b22222", linewidth=2.0)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax.set_xticks(range(1, len(order) + 1), order)
        ax.set_ylabel("Deviation (oriented)")
        ax.set_title("PD deviation by group / medication state")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(fig_grp, dpi=160)
        plt.close(fig)
    except Exception:
        pass

    try:
        if not pair_df.empty:
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            for _, r in pair_df.iterrows():
                ax.plot([0, 1], [float(r["off"]), float(r["on"])], color="#2a607f", alpha=0.45, linewidth=1.1)
            ax.set_xticks([0, 1], ["OFF", "ON"])
            ax.set_ylabel("Deviation (oriented)")
            ax.set_title("Within-subject PD ON/OFF deviation")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(fig_pair, dpi=160)
            plt.close(fig)
    except Exception:
        pass

    incl = {
        "n_runs_used": int(len(run_df)),
        "n_subject_sessions": int(len(agg)),
        "n_controls": int((agg["group"] == "CTL").sum()),
        "n_pd": int((agg["group"] == "PD").sum()),
        "n_pd_off": int(((agg["group"] == "PD") & (agg["med_state"].astype(str).str.upper() == "OFF")).sum()),
        "n_pd_on": int(((agg["group"] == "PD") & (agg["med_state"].astype(str).str.upper() == "ON")).sum()),
        "n_paired_on_off": int(len(pair_df)),
        "fallback_channel_count": int(pd.to_numeric(agg["fallback_channel_used"], errors="coerce").fillna(0).astype(bool).sum()),
        "n_skipped_runs": int(len(skipped_runs)),
    }
    _write_json(out_dir / "inclusion_exclusion_summary.json", incl)

    _write_json(
        summary_json,
        {
            "status": "PASS",
            "reason": "",
            "n_endpoints": int(len(end_df)),
            "n_perm": int(n_perm),
            "inclusion_exclusion": incl,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
