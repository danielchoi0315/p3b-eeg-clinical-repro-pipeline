#!/usr/bin/env python3
"""Normative cross-validated deviation and risk association for ds004796."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import roc_auc_score

from common.lawc_audit import bh_fdr
from p3b_pipeline.clinical import subject_load_deviation_score


def _parse_seeds(spec: str) -> List[int]:
    out: List[int] = []
    for raw in str(spec).split(","):
        t = raw.strip()
        if not t:
            continue
        if "-" in t:
            a, b = t.split("-", 1)
            ai = int(a)
            bi = int(b)
            step = 1 if ai <= bi else -1
            out.extend(range(ai, bi + step, step))
        else:
            out.append(int(t))
    return sorted(set(out))


def _encode_sex(series: pd.Series) -> np.ndarray:
    vals = series.fillna("").astype(str).str.strip().str.lower()
    out = np.full(len(vals), np.nan, dtype=float)
    for i, v in enumerate(vals):
        if v in {"m", "male", "1"}:
            out[i] = 1.0
        elif v in {"f", "female", "0"}:
            out[i] = 0.0
    return out


def _prepare_participants(parts: pd.DataFrame) -> pd.DataFrame:
    p = parts.copy()
    if "participant_id" in p.columns:
        p["subject_id"] = p["participant_id"].astype(str).str.replace(r"^sub-", "", regex=True)
    elif "subject_id" in p.columns:
        p["subject_id"] = p["subject_id"].astype(str).str.replace(r"^sub-", "", regex=True)
    else:
        raise RuntimeError("participants table missing participant_id/subject_id column")

    age_col = next((c for c in ["age", "Age", "AGE"] if c in p.columns), None)
    sex_col = next((c for c in ["sex", "Sex", "SEX", "gender", "Gender"] if c in p.columns), None)
    age_src = p[age_col] if age_col is not None else pd.Series([np.nan] * len(p), index=p.index)
    sex_src = p[sex_col] if sex_col is not None else pd.Series([""] * len(p), index=p.index)
    p["age"] = pd.to_numeric(age_src, errors="coerce")
    p["sex"] = sex_src.astype(str)
    p["sex_num"] = _encode_sex(p["sex"])

    if "APOE_haplotype" in p.columns:
        apo = p["APOE_haplotype"].astype(str).str.lower()
        p["APOE_e4_carrier"] = apo.str.contains("e4", na=False).astype(float)
    else:
        p["APOE_e4_carrier"] = np.nan

    if "PICALM_rs3851179" in p.columns:
        gen = p["PICALM_rs3851179"].astype(str).str.upper()
        # Orientation is reported transparently; no risk-allele claim here.
        p["PICALM_A_carrier"] = gen.str.contains("A", regex=False, na=False).astype(float)
        p["PICALM_genotype"] = gen
    else:
        p["PICALM_A_carrier"] = np.nan
        p["PICALM_genotype"] = ""

    # Construct memory composites from CVLT columns when available.
    cvlt_cols = [c for c in p.columns if c.upper().startswith("CVLT_")]
    if cvlt_cols:
        mat = p[cvlt_cols].apply(pd.to_numeric, errors="coerce")
        p["CVLT_total"] = mat.sum(axis=1, min_count=1)
        p["CVLT_mean"] = mat.mean(axis=1)
    else:
        p["CVLT_total"] = np.nan
        p["CVLT_mean"] = np.nan

    return p


def _fold_assignments(subjects: Sequence[str], n_folds: int, seed: int) -> Dict[str, int]:
    subs = sorted(set(str(x) for x in subjects if str(x)))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(subs)
    folds = np.array_split(np.asarray(subs, dtype=object), int(max(2, n_folds)))
    out: Dict[str, int] = {}
    for i, arr in enumerate(folds):
        for s in arr.tolist():
            out[str(s)] = int(i)
    return out


def _fit_predict_fold(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[np.ndarray, float]:
    feats = ["memory_load", "age_cov", "trial_order_cov", "sex_cov"]
    tr = train.copy()
    te = test.copy()
    for c in feats:
        med = pd.to_numeric(tr[c], errors="coerce").median()
        if not np.isfinite(med):
            med = 0.0
        tr[c] = pd.to_numeric(tr[c], errors="coerce").fillna(med)
        te[c] = pd.to_numeric(te[c], errors="coerce").fillna(med)

    x_tr = tr[feats].to_numpy(dtype=float)
    x_te = te[feats].to_numpy(dtype=float)
    y_tr = pd.to_numeric(tr["p3_amp_uV"], errors="coerce").to_numpy(dtype=float)

    m = np.nanmean(x_tr, axis=0)
    s = np.nanstd(x_tr, axis=0)
    s[s <= 1e-8] = 1.0
    x_tr_n = (x_tr - m) / s
    x_te_n = (x_te - m) / s

    model = None
    try:
        model = HuberRegressor(alpha=0.0, epsilon=1.35, max_iter=300)
        model.fit(x_tr_n, y_tr)
        y_hat_tr = model.predict(x_tr_n)
        y_hat_te = model.predict(x_te_n)
    except Exception:
        model = LinearRegression()
        model.fit(x_tr_n, y_tr)
        y_hat_tr = model.predict(x_tr_n)
        y_hat_te = model.predict(x_te_n)

    resid = y_tr - y_hat_tr
    sigma = float(np.nanstd(resid))
    if not np.isfinite(sigma) or sigma < 1e-6:
        sigma = 1.0
    return y_hat_te, sigma


def _seed_worker(
    *,
    seed: int,
    trial_csv: str,
    participants_csv: str,
    out_dir: str,
    cv_folds: int,
) -> Dict[str, Any]:
    trial = pd.read_csv(trial_csv)
    parts = _prepare_participants(pd.read_csv(participants_csv, sep="\t"))

    trial["subject_id"] = trial["subject_id"].astype(str).str.replace(r"^sub-", "", regex=True)
    # Avoid x/y suffix collisions when trial tables already carry demographic columns.
    overlap_cols = [
        "age",
        "sex",
        "sex_num",
        "APOE_e4_carrier",
        "PICALM_A_carrier",
        "PICALM_genotype",
        "CVLT_total",
        "CVLT_mean",
    ]
    drop_cols = [c for c in overlap_cols if c in trial.columns and c != "subject_id"]
    if drop_cols:
        trial = trial.drop(columns=drop_cols, errors="ignore")

    merged = trial.merge(
        parts[
            [
                "subject_id",
                "age",
                "sex_num",
                "APOE_e4_carrier",
                "PICALM_A_carrier",
                "PICALM_genotype",
                "CVLT_total",
                "CVLT_mean",
            ]
        ],
        on="subject_id",
        how="left",
    )
    merged["age_cov"] = pd.to_numeric(merged["age"], errors="coerce")
    merged["sex_cov"] = pd.to_numeric(merged["sex_num"], errors="coerce")
    merged["trial_order_cov"] = pd.to_numeric(merged["trial_order"], errors="coerce")
    merged["memory_load"] = pd.to_numeric(merged["memory_load"], errors="coerce")
    merged["p3_amp_uV"] = pd.to_numeric(merged["p3_amp_uV"], errors="coerce")
    merged = merged[np.isfinite(merged["memory_load"]) & np.isfinite(merged["p3_amp_uV"])].copy()
    if merged.empty:
        return {"seed": seed, "status": "FAIL", "reason": "no finite load/p3 trials after merge"}

    subjects = sorted(merged["subject_id"].astype(str).unique().tolist())
    if len(subjects) < max(10, cv_folds):
        return {"seed": seed, "status": "FAIL", "reason": f"too few subjects for CV ({len(subjects)})"}

    fold_map = _fold_assignments(subjects, n_folds=cv_folds, seed=seed)
    merged["fold"] = merged["subject_id"].map(fold_map).astype(int)

    z_all = np.full(len(merged), np.nan, dtype=float)
    for f in sorted(set(merged["fold"].tolist())):
        tr = merged[merged["fold"] != int(f)].copy()
        te = merged[merged["fold"] == int(f)].copy()
        if tr.empty or te.empty:
            continue
        y_hat_te, sigma = _fit_predict_fold(tr, te)
        idx = te.index.to_numpy(dtype=int)
        y = te["p3_amp_uV"].to_numpy(dtype=float)
        z_all[idx] = (y - y_hat_te) / sigma

    merged["z"] = z_all
    merged = merged[np.isfinite(merged["z"])].copy()
    if merged.empty:
        return {"seed": seed, "status": "FAIL", "reason": "all held-out z values are non-finite"}

    subj_df = subject_load_deviation_score(
        merged[["subject_key", "subject_id", "memory_load", "z"]]
        .rename(columns={"subject_key": "subject"})
        .copy()
    )
    if subj_df.empty:
        return {"seed": seed, "status": "FAIL", "reason": "subject deviation table is empty"}
    subj_df = subj_df.rename(columns={"subject": "subject_key"})
    subj_df["subject_id"] = subj_df["subject_key"].astype(str).str.extract(r"sub-([^:]+)", expand=False)
    subj_df["seed"] = int(seed)
    subj_df = subj_df.merge(
        parts[
            [
                "subject_id",
                "age",
                "sex",
                "APOE_e4_carrier",
                "PICALM_A_carrier",
                "PICALM_genotype",
                "CVLT_total",
                "CVLT_mean",
            ]
        ],
        on="subject_id",
        how="left",
    )

    out_seed = Path(out_dir) / "seeds" / f"seed_{seed:03d}"
    out_seed.mkdir(parents=True, exist_ok=True)
    subj_csv = out_seed / "subject_deviation.csv"
    subj_df.to_csv(subj_csv, index=False)

    fold_hash = hashlib.sha256(
        "\n".join(f"{k}:{fold_map[k]}" for k in sorted(fold_map.keys())).encode("utf-8")
    ).hexdigest()
    checksum = hashlib.sha256(
        pd.util.hash_pandas_object(subj_df[["subject_id", "z_mean", "z_hi_minus_lo"]], index=False)
        .to_numpy()
        .tobytes()
    ).hexdigest()

    return {
        "seed": int(seed),
        "status": "PASS",
        "n_subjects": int(subj_df["subject_id"].nunique()),
        "n_trials_heldout": int(len(merged)),
        "fold_hash": fold_hash,
        "seed_checksum": checksum,
        "subject_deviation_csv": str(subj_csv),
        "z_hi_minus_lo_mean": float(pd.to_numeric(subj_df["z_hi_minus_lo"], errors="coerce").mean()),
        "z_mean_mean": float(pd.to_numeric(subj_df["z_mean"], errors="coerce").mean()),
    }


def _bootstrap_auc(y: np.ndarray, s: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    m = np.isfinite(y) & np.isfinite(s)
    y = y[m].astype(int)
    s = s[m].astype(float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    auc = float(roc_auc_score(y, s))
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        ss = s[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(float(roc_auc_score(yy, ss)))
    if not vals:
        return auc, float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    return auc, float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def _robust_beta(y: np.ndarray, x: np.ndarray, covars: np.ndarray) -> float:
    import statsmodels.api as sm

    X = np.column_stack([x, covars])
    X = sm.add_constant(X, has_constant="add")
    model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    res = model.fit()
    return float(res.params[1])


def _perm_beta_ols(y: np.ndarray, x: np.ndarray, covars: np.ndarray, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(int(seed))
    X0 = np.column_stack([np.ones(len(y)), covars])
    obs_x = np.column_stack([X0, x])
    obs_beta = float(np.linalg.lstsq(obs_x, y, rcond=None)[0][-1])
    null = np.full(int(n_perm), np.nan, dtype=float)
    for i in range(int(n_perm)):
        xp = x.copy()
        rng.shuffle(xp)
        Xp = np.column_stack([X0, xp])
        null[i] = float(np.linalg.lstsq(Xp, y, rcond=None)[0][-1])
    finite = null[np.isfinite(null)]
    if finite.size == 0:
        return float("nan")
    return float((1.0 + np.sum(np.abs(finite) >= abs(obs_beta))) / (1.0 + finite.size))


def _analyze_binary(subj: pd.DataFrame, endpoint: str, n_perm: int, seed: int) -> Dict[str, Any]:
    df = subj.copy()
    df["y"] = pd.to_numeric(df["z_hi_minus_lo"], errors="coerce")
    df["grp"] = pd.to_numeric(df[endpoint], errors="coerce")
    df["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    df["sex"] = _encode_sex(df.get("sex", pd.Series([""] * len(df))))
    df = df.dropna(subset=["y", "grp"]).copy()
    if df.empty or df["grp"].nunique() < 2:
        return {"endpoint": endpoint, "type": "binary", "status": "SKIP", "reason": "insufficient groups"}
    if min((df["grp"] == 0).sum(), (df["grp"] == 1).sum()) < 10:
        return {"endpoint": endpoint, "type": "binary", "status": "SKIP", "reason": "group count <10"}

    cov = np.column_stack(
        [
            np.nan_to_num(df["age"].to_numpy(dtype=float), nan=float(np.nanmedian(df["age"])) if np.isfinite(df["age"]).any() else 0.0),
            np.nan_to_num(df["sex"].to_numpy(dtype=float), nan=0.0),
        ]
    )
    y = df["y"].to_numpy(dtype=float)
    g = df["grp"].to_numpy(dtype=float)
    beta = _robust_beta(y, g, cov)
    p_perm = _perm_beta_ols(y, g, cov, n_perm=n_perm, seed=seed + 17)
    auc, auc_lo, auc_hi = _bootstrap_auc(g, y, n_boot=2000, seed=seed + 29)

    return {
        "endpoint": endpoint,
        "type": "binary",
        "status": "PASS",
        "n": int(len(df)),
        "n_group0": int((g == 0).sum()),
        "n_group1": int((g == 1).sum()),
        "beta": float(beta),
        "perm_p": float(p_perm),
        "auc": float(auc),
        "auc_ci95_lo": float(auc_lo),
        "auc_ci95_hi": float(auc_hi),
        "notes": "robust beta (RLM); permutation p from OLS beta under label shuffling",
    }


def _analyze_continuous(subj: pd.DataFrame, endpoint: str, n_perm: int, seed: int) -> Dict[str, Any]:
    df = subj.copy()
    df["y"] = pd.to_numeric(df[endpoint], errors="coerce")
    df["x"] = pd.to_numeric(df["z_hi_minus_lo"], errors="coerce")
    df["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    df["sex"] = _encode_sex(df.get("sex", pd.Series([""] * len(df))))
    df = df.dropna(subset=["y", "x"]).copy()
    if len(df) < 30:
        return {"endpoint": endpoint, "type": "continuous", "status": "SKIP", "reason": "n<30"}
    cov = np.column_stack(
        [
            np.nan_to_num(df["age"].to_numpy(dtype=float), nan=float(np.nanmedian(df["age"])) if np.isfinite(df["age"]).any() else 0.0),
            np.nan_to_num(df["sex"].to_numpy(dtype=float), nan=0.0),
        ]
    )
    y = df["y"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    beta = _robust_beta(y, x, cov)
    p_perm = _perm_beta_ols(y, x, cov, n_perm=n_perm, seed=seed + 71)
    return {
        "endpoint": endpoint,
        "type": "continuous",
        "status": "PASS",
        "n": int(len(df)),
        "beta": float(beta),
        "perm_p": float(p_perm),
        "auc": float("nan"),
        "auc_ci95_lo": float("nan"),
        "auc_ci95_hi": float("nan"),
        "notes": "robust beta (RLM); permutation p from OLS beta under predictor shuffling",
    }


def _plot_binary(subj: pd.DataFrame, rows: pd.DataFrame, out_path: Path) -> None:
    use = rows[(rows["type"] == "binary") & (rows["status"] == "PASS")].copy()
    if use.empty:
        return
    eps = use["endpoint"].tolist()[:3]
    fig, axes = plt.subplots(1, len(eps), figsize=(6.0 * len(eps), 4.8), squeeze=False)
    for i, ep in enumerate(eps):
        ax = axes[0, i]
        dd = subj.copy()
        dd["grp"] = pd.to_numeric(dd[ep], errors="coerce")
        dd["score"] = pd.to_numeric(dd["z_hi_minus_lo"], errors="coerce")
        dd = dd.dropna(subset=["grp", "score"])
        for g, col in [(0, "#1f5b7a"), (1, "#c13a24")]:
            vals = dd.loc[dd["grp"] == g, "score"].to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            x = np.full(len(vals), g + 1, dtype=float)
            jit = np.linspace(-0.15, 0.15, len(vals)) if len(vals) > 1 else np.asarray([0.0])
            ax.scatter(x + jit, vals, s=14, alpha=0.45, color=col)
            ax.hlines(float(np.median(vals)), g + 0.8, g + 1.2, color="black", linewidth=2)
        ax.set_xticks([1, 2], ["0", "1"])
        ax.set_title(ep)
        ax.set_ylabel("z_hi_minus_lo")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_continuous(subj: pd.DataFrame, rows: pd.DataFrame, out_path: Path) -> None:
    use = rows[(rows["type"] == "continuous") & (rows["status"] == "PASS")].copy()
    if use.empty:
        return
    eps = use["endpoint"].tolist()[:4]
    fig, axes = plt.subplots(1, len(eps), figsize=(5.6 * len(eps), 4.8), squeeze=False)
    for i, ep in enumerate(eps):
        ax = axes[0, i]
        dd = subj.copy()
        dd["x"] = pd.to_numeric(dd["z_hi_minus_lo"], errors="coerce")
        dd["y"] = pd.to_numeric(dd[ep], errors="coerce")
        dd = dd.dropna(subset=["x", "y"])
        if dd.empty:
            continue
        ax.scatter(dd["x"], dd["y"], s=14, alpha=0.5, color="#1f5b7a")
        if len(dd) >= 5:
            p = np.polyfit(dd["x"].to_numpy(dtype=float), dd["y"].to_numpy(dtype=float), deg=1)
            xx = np.linspace(dd["x"].min(), dd["x"].max(), 100)
            yy = p[0] * xx + p[1]
            ax.plot(xx, yy, color="#c13a24", linewidth=2)
        ax.set_title(ep)
        ax.set_xlabel("z_hi_minus_lo")
        ax.set_ylabel(ep)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_auc(rows: pd.DataFrame, out_path: Path) -> None:
    use = rows[(rows["type"] == "binary") & (rows["status"] == "PASS")].copy()
    if use.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x = np.arange(len(use), dtype=float)
    auc = pd.to_numeric(use["auc"], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(use["auc_ci95_lo"], errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(use["auc_ci95_hi"], errors="coerce").to_numpy(dtype=float)
    ax.bar(x, auc, color="#245b7f", alpha=0.85)
    yerr = np.vstack([auc - lo, hi - auc])
    yerr = np.where(np.isfinite(yerr), yerr, 0.0)
    ax.errorbar(x, auc, yerr=yerr, fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(x, use["endpoint"].astype(str).tolist(), rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial_table_csv", type=Path, required=True)
    ap.add_argument("--participants_tsv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--seeds", type=str, default="0-199")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--n_perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seeds").mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        raise RuntimeError("empty seeds specification")

    # Multi-seed CV
    recs: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [
            ex.submit(
                _seed_worker,
                seed=int(s),
                trial_csv=str(args.trial_table_csv),
                participants_csv=str(args.participants_tsv),
                out_dir=str(out_dir),
                cv_folds=int(args.cv_folds),
            )
            for s in seeds
        ]
        for fut in as_completed(futs):
            recs.append(fut.result())

    seed_df = pd.DataFrame(recs).sort_values("seed")
    pass_df = seed_df[seed_df["status"] == "PASS"].copy()
    if pass_df.empty:
        raise RuntimeError("all seeds failed; no normative outputs")

    subj_rows: List[pd.DataFrame] = []
    for _, r in pass_df.iterrows():
        p = Path(str(r["subject_deviation_csv"]))
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d["seed"] = int(r["seed"])
        subj_rows.append(d)
    if not subj_rows:
        raise RuntimeError("no per-seed subject deviation CSV files found")
    all_subj = pd.concat(subj_rows, axis=0, ignore_index=True)

    subj = (
        all_subj.groupby("subject_id", as_index=False)
        .agg(
            z_mean=("z_mean", "mean"),
            z_hi_minus_lo=("z_hi_minus_lo", "mean"),
            n_trials=("n_trials", "mean"),
            n_seeds=("seed", "nunique"),
            age=("age", "first"),
            sex=("sex", "first"),
            APOE_e4_carrier=("APOE_e4_carrier", "first"),
            PICALM_A_carrier=("PICALM_A_carrier", "first"),
            PICALM_genotype=("PICALM_genotype", "first"),
            CVLT_total=("CVLT_total", "first"),
            CVLT_mean=("CVLT_mean", "first"),
        )
    )
    deviation_csv = out_dir / "deviation_scores.csv"
    subj.to_csv(deviation_csv, index=False)

    # Endpoint testing
    endpoint_rows: List[Dict[str, Any]] = []
    if "APOE_e4_carrier" in subj.columns:
        endpoint_rows.append(_analyze_binary(subj, "APOE_e4_carrier", n_perm=int(args.n_perm), seed=int(args.seed) + 101))
    if "PICALM_A_carrier" in subj.columns:
        endpoint_rows.append(_analyze_binary(subj, "PICALM_A_carrier", n_perm=int(args.n_perm), seed=int(args.seed) + 211))

    for col in ["CVLT_total", "CVLT_mean"]:
        if col in subj.columns:
            endpoint_rows.append(_analyze_continuous(subj, col, n_perm=int(args.n_perm), seed=int(args.seed) + 307))

    res = pd.DataFrame(endpoint_rows)
    if not res.empty:
        pvals = pd.to_numeric(res["perm_p"], errors="coerce").to_numpy(dtype=float)
        q = bh_fdr([float(x) if np.isfinite(x) else 1.0 for x in pvals.tolist()])
        res["perm_q"] = q
    else:
        res = pd.DataFrame(
            columns=[
                "endpoint",
                "type",
                "status",
                "n",
                "beta",
                "perm_p",
                "perm_q",
                "auc",
                "auc_ci95_lo",
                "auc_ci95_hi",
                "notes",
            ]
        )
    risk_csv = out_dir / "clinical_risk_results.csv"
    res.to_csv(risk_csv, index=False)

    fig_group = out_dir / "FIG_deviation_by_risk_group.png"
    fig_mem = out_dir / "FIG_deviation_vs_memory_scores.png"
    fig_auc = out_dir / "FIG_auc_risk_group.png"
    _plot_binary(subj, res, fig_group)
    _plot_continuous(subj, res, fig_mem)
    _plot_auc(res, fig_auc)

    # Seed stability report
    zvals = pd.to_numeric(pass_df["z_hi_minus_lo_mean"], errors="coerce").to_numpy(dtype=float)
    zvals = zvals[np.isfinite(zvals)]
    if zvals.size:
        ci = [float(np.quantile(zvals, 0.025)), float(np.quantile(zvals, 0.975))]
        worst = [float(np.min(zvals)), float(np.max(zvals))]
    else:
        ci = [float("nan"), float("nan")]
        worst = [float("nan"), float("nan")]
    stability = {
        "n_seeds_requested": int(len(seeds)),
        "n_seeds_completed": int(len(pass_df)),
        "seed_records": seed_df.to_dict(orient="records"),
        "seed_split_hashes": pass_df["fold_hash"].astype(str).tolist() if "fold_hash" in pass_df.columns else [],
        "seed_effect_checksums": pass_df["seed_checksum"].astype(str).tolist() if "seed_checksum" in pass_df.columns else [],
        "z_hi_minus_lo_seed_mean": {
            "mean": float(np.nanmean(zvals)) if zvals.size else float("nan"),
            "ci95": ci,
            "worst": worst,
            "values": zvals.tolist(),
        },
    }
    stability_json = out_dir / "normative_seed_stability.json"
    stability_json.write_text(json.dumps(stability, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "PASS",
                "deviation_scores_csv": str(deviation_csv),
                "clinical_risk_results_csv": str(risk_csv),
                "normative_seed_stability_json": str(stability_json),
                "n_seeds_completed": int(len(pass_df)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
