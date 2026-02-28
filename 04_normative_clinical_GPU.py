#!/usr/bin/env python3
"""04_normative_clinical_GPU.py

GPU module: normative modeling + clinical deviation scoring + RT linkage.

Additions in this revision:
- GH200/H100-safe runtime policy (no hardcoded SM12 arch)
- NVML utilization logger at 1 Hz
- seed override for multi-seed sweep
- automatic batch-size tuning (~85-92% VRAM)
- compile fallback logging
- optional clinical cohort application (clean skip when absent)
- explicit normative metrics output (NLL, calibration, z-score stability)
- subject_key-safe grouping (dataset-qualified IDs)
- seed-meaningful subject bootstrap + prediction checksums
"""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import time

import h5py
import numpy as np
import pandas as pd
import torch

from common.hardware import (
    apply_cpu_thread_env,
    auto_tune_batch_size,
    configure_cuda_arch_list,
    configure_torch_backends,
    detect_hardware_info,
    start_gpu_util_logger,
)
from common.lawc_audit import checksum_array_sha256
from p3b_pipeline.clinical import robust_regress_severity, subject_load_deviation_score, trial_zscore
from p3b_pipeline.config import cfg_get, load_yaml
from p3b_pipeline.env import ThreadConfig, apply_thread_config, cuda_device_summary, require_cuda_or_die
from p3b_pipeline.h5io import iter_subject_feature_files, read_subject_h5
from p3b_pipeline.logging_utils import configure_logging
from p3b_pipeline.manifest import write_manifest
from p3b_pipeline.normative import HeteroscedasticMLP, gaussian_nll, predict_ensemble, train_one_member
from p3b_pipeline.rt_linkage import margin_to_next_rt
from p3b_pipeline.torch_utils import seed_all, select_amp_dtype


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--run_id", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--healthy_cohort", type=str, default="healthy")
    ap.add_argument("--clinical_cohort", type=str, default="clinical")
    ap.add_argument(
        "--healthy_dataset_ids",
        type=str,
        default="ds005095,ds003655,ds004117",
        help="Comma-separated healthy dataset ids used to train normative model",
    )
    ap.add_argument("--severity_csv", type=Path, default=None, help="CSV with columns: subject,severity")
    ap.add_argument("--severity_col", type=str, default="severity")
    ap.add_argument("--gpu_log_csv", type=Path, default=None)
    ap.add_argument("--gpu_log_tag", type=str, default="")
    ap.add_argument("--enable_fp8", action="store_true", help="Enable FP8 probe + autocast if stable")
    ap.add_argument("--allow_deterministic", action="store_true", help="Allow identical seed checksums")
    return ap.parse_args()


def _parse_csv_set(raw: str) -> set[str]:
    return {x.strip() for x in str(raw).split(",") if x.strip()}


def _dataset_from_attrs_or_path(attrs: dict, fp: Path, features_root: Path) -> str:
    if "dataset_id" in attrs and attrs["dataset_id"]:
        return str(attrs["dataset_id"])
    try:
        rel = fp.relative_to(features_root)
        return rel.parts[0]
    except Exception:
        return "unknown"


def _subject_key_from_attrs(attrs: dict, dataset_id: str) -> str:
    if attrs.get("subject_key"):
        return str(attrs["subject_key"])
    bids_sub = str(attrs.get("bids_subject", attrs.get("subject", "unknown")))
    bids_ses = str(attrs.get("bids_session", attrs.get("session", "")))
    if bids_ses and bids_ses.lower() not in {"", "na", "none"}:
        return f"{dataset_id}:sub-{bids_sub}:ses-{bids_ses}"
    return f"{dataset_id}:sub-{bids_sub}"


def _collect_trials(
    files: list[Path],
    *,
    features_root: Path,
    cohort_value: str,
    allowed_datasets: set[str] | None,
) -> pd.DataFrame:
    rows = []
    for fp in files:
        arrays, attrs = read_subject_h5(fp)
        cohort = str(attrs.get("cohort", ""))
        if cohort != cohort_value:
            continue

        dataset_id = _dataset_from_attrs_or_path(attrs, fp, features_root)
        if allowed_datasets is not None and dataset_id not in allowed_datasets:
            continue
        if not dataset_id:
            raise RuntimeError(f"Missing dataset_id in feature file: {fp}")

        subject_key = _subject_key_from_attrs(attrs, dataset_id)
        if not subject_key:
            raise RuntimeError(f"Missing subject_key in feature file: {fp}")

        bids_sub = str(attrs.get("bids_subject", attrs.get("subject", "")))
        bids_ses = str(attrs.get("bids_session", attrs.get("session", "")))
        bids_run = str(attrs.get("bids_run", attrs.get("run", "")))

        load = arrays.get("memory_load")
        age = arrays.get("age")
        order = arrays.get("trial_order")
        y = arrays.get("p3b_amp")
        rt = arrays.get("rt")
        acc = arrays.get("accuracy")

        if load is None or order is None or y is None:
            continue

        n = len(y)
        age_arr = age if age is not None else np.full(n, np.nan, dtype=np.float32)

        rows.append(
            pd.DataFrame(
                {
                    "subject_key": [subject_key] * n,
                    # Keep compatibility with code expecting `subject`; this now equals canonical subject_key.
                    "subject": [subject_key] * n,
                    "dataset_id": [dataset_id] * n,
                    "bids_subject": [bids_sub] * n,
                    "bids_session": [bids_ses] * n,
                    "bids_run": [bids_run] * n,
                    "memory_load": load.astype(float),
                    "age": age_arr.astype(float),
                    "trial_order": order.astype(float),
                    "p3b_amp": y.astype(float),
                    "rt": rt.astype(float) if rt is not None else np.full(n, np.nan),
                    "accuracy": acc.astype(float) if acc is not None else np.full(n, np.nan),
                }
            )
        )

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, axis=0, ignore_index=True)
    if (df["dataset_id"].astype(str).str.len() == 0).any():
        raise RuntimeError("Encountered empty dataset_id in collected trials")
    if (df["subject_key"].astype(str).str.len() == 0).any():
        raise RuntimeError("Encountered empty subject_key in collected trials")
    return df


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0) + 1e-8
    return mu, sd


def _standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


@torch.no_grad()
def _predict_in_batches(models, x: torch.Tensor, *, batch_size: int, amp_dtype: torch.dtype):
    mus = []
    vars_ = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size]
        mu, var = predict_ensemble(models, xb, amp_dtype=amp_dtype)
        mus.append(mu)
        vars_.append(var)
    return torch.cat(mus, dim=0), torch.cat(vars_, dim=0)


def _df_to_h5(f: h5py.File, group_name: str, df: pd.DataFrame) -> None:
    g = f.require_group(group_name)
    for k in list(g.keys()):
        del g[k]
    for col in df.columns:
        v = df[col].to_numpy()
        if v.dtype.kind in {"U", "S", "O"}:
            dt = h5py.string_dtype(encoding="utf-8")
            str_vals = pd.Series(v).where(pd.notna(v), "").astype(str).to_numpy(dtype=object)
            g.create_dataset(col, data=str_vals, dtype=dt)
        else:
            g.create_dataset(col, data=v, compression="gzip", compression_opts=4, shuffle=True)
    g.attrs["n_rows"] = len(df)


def _mean_gaussian_nll(y: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    var = np.maximum(np.asarray(var, dtype=float), 1e-6)
    return float(np.mean(0.5 * np.log(2.0 * math.pi * var) + 0.5 * ((y - mu) ** 2) / var))


def _calibration_summary(z: np.ndarray) -> dict:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return {
            "z_mean": float("nan"),
            "z_std": float("nan"),
            "frac_abs_lt_1": float("nan"),
            "frac_abs_lt_2": float("nan"),
        }
    return {
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "frac_abs_lt_1": float(np.mean(np.abs(z) <= 1.0)),
        "frac_abs_lt_2": float(np.mean(np.abs(z) <= 2.0)),
    }


def _z_stability(df: pd.DataFrame) -> dict:
    if df.empty or "z" not in df.columns:
        return {"subject_mean_std": float("nan"), "subject_mean_iqr": float("nan")}
    subj = df.groupby("subject_key")["z"].mean()
    q1 = float(subj.quantile(0.25)) if len(subj) else float("nan")
    q3 = float(subj.quantile(0.75)) if len(subj) else float("nan")
    return {
        "subject_mean_std": float(subj.std()) if len(subj) else float("nan"),
        "subject_mean_iqr": float(q3 - q1) if len(subj) else float("nan"),
    }


def _probe_vram_fraction_for_normative_batch(
    *,
    batch_size: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    cfg: dict,
    amp_dtype: torch.dtype,
) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = HeteroscedasticMLP(
        in_features=x_train.shape[1],
        hidden_sizes=list(cfg.get("hidden_sizes", [128, 128, 64])),
        dropout=float(cfg.get("dropout", 0.05)),
    ).cuda()
    optim = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("lr", 2e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4))
    )

    bs = min(batch_size, x_train.shape[0])
    idx = torch.arange(bs, dtype=torch.long)
    xb = x_train[idx].cuda(non_blocking=True)
    yb = y_train[idx].cuda(non_blocking=True)

    optim.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        mu, log_var = model(xb)
        loss = gaussian_nll(yb, mu, log_var)
    loss.backward()
    optim.step()

    peak = float(torch.cuda.max_memory_allocated())
    total = float(torch.cuda.get_device_properties(0).total_memory)

    del model
    del optim
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return peak / max(total, 1.0)


def _subject_bootstrap(df: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, Dict[str, Any]]:
    subjects = np.asarray(sorted(df["subject_key"].astype(str).unique()))
    if subjects.size == 0:
        raise RuntimeError("Cannot bootstrap empty subject set")

    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, subjects.size, size=subjects.size)
    sampled_subjects = subjects[idx]

    frames = [df[df["subject_key"] == sid] for sid in sampled_subjects]
    boot = pd.concat(frames, axis=0, ignore_index=True)

    h = hashlib.sha256()
    h.update(np.asarray(idx, dtype=np.int64).tobytes())
    h.update("|".join(sampled_subjects.tolist()).encode("utf-8"))

    meta = {
        "mode": "subject_bootstrap",
        "n_unique_subjects_original": int(subjects.size),
        "n_unique_subjects_sampled": int(len(set(sampled_subjects.tolist()))),
        "subject_bootstrap_indices_hash": h.hexdigest(),
        "subject_bootstrap_indices": idx.tolist(),
    }
    return boot, meta


def main() -> None:
    args = parse_args()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

    apply_cpu_thread_env(allow_override=True)
    apply_thread_config(ThreadConfig(), allow_override=True)

    require_cuda_or_die()
    arch_action = configure_cuda_arch_list(for_extension_build=False)
    configure_torch_backends(enable_tf32=True)
    device = torch.device("cuda")

    cfg = load_yaml(args.config)
    norm_cfg = dict(cfg_get(cfg, "normative", {}))

    log_dir = args.out_root / "logs"
    logger = configure_logging(log_dir=log_dir, run_id=run_id, name="04_normative_clinical_GPU")

    manifest_path = write_manifest(
        out_dir=log_dir,
        run_id=run_id,
        entrypoint="04_normative_clinical_GPU",
        args={k: str(v) for k, v in vars(args).items()},
        extra={
            "cuda": cuda_device_summary(),
            "cuda_arch_action": arch_action,
            "hardware": detect_hardware_info().__dict__,
        },
    )
    logger.info("Wrote manifest: %s", manifest_path)
    logger.info("Using device: %s", cuda_device_summary())

    seed = int(args.seed if args.seed is not None else norm_cfg.get("seed", 0))
    seed_all(seed)
    norm_cfg["seed"] = seed

    files = iter_subject_feature_files(args.features_root)
    if not files:
        raise RuntimeError(f"No feature files found under {args.features_root}")

    healthy_ids = _parse_csv_set(args.healthy_dataset_ids)

    df_h = _collect_trials(
        files,
        features_root=args.features_root,
        cohort_value=args.healthy_cohort,
        allowed_datasets=healthy_ids if healthy_ids else None,
    )
    df_c = _collect_trials(
        files,
        features_root=args.features_root,
        cohort_value=args.clinical_cohort,
        allowed_datasets=None,
    )

    if df_h.empty:
        raise RuntimeError(
            f"No trials found for healthy_cohort='{args.healthy_cohort}' in datasets={sorted(healthy_ids)}"
        )

    logger.info("Healthy trials: %d (subjects=%d)", len(df_h), df_h["subject_key"].nunique())
    logger.info("Clinical trials: %d (subjects=%d)", len(df_c), df_c["subject_key"].nunique() if not df_c.empty else 0)

    # Seed-meaningful stochasticity: subject-level bootstrap resampling.
    df_train, bootstrap_meta = _subject_bootstrap(df_h, seed=seed)
    logger.info(
        "Bootstrap mode=%s orig_subjects=%d sampled_unique=%d hash=%s",
        bootstrap_meta["mode"],
        bootstrap_meta["n_unique_subjects_original"],
        bootstrap_meta["n_unique_subjects_sampled"],
        bootstrap_meta["subject_bootstrap_indices_hash"],
    )

    y_train_np = (df_train["p3b_amp"].to_numpy(dtype=float) * 1e6).astype(np.float32)
    x_train_np = df_train[["memory_load", "age", "trial_order"]].to_numpy(dtype=float).astype(np.float32)
    if np.isfinite(x_train_np[:, 1]).any():
        x_train_np[np.isnan(x_train_np[:, 1]), 1] = float(np.nanmean(x_train_np[:, 1]))
    else:
        x_train_np[:, 1] = 0.0

    x_mu, x_sd = _standardize_fit(x_train_np)
    y_mu = float(np.mean(y_train_np))
    y_sd = float(np.std(y_train_np) + 1e-8)

    x_train_z = _standardize_apply(x_train_np, x_mu, x_sd).astype(np.float32)
    y_train_z = ((y_train_np - y_mu) / y_sd).astype(np.float32)

    x_train = torch.from_numpy(x_train_z)
    y_train = torch.from_numpy(y_train_z)

    amp_dtype, fp8_probe = select_amp_dtype(
        str(norm_cfg.get("amp", "bf16")), enable_fp8=bool(args.enable_fp8 or norm_cfg.get("enable_fp8", False))
    )
    logger.info("AMP dtype=%s fp8_probe=%s", amp_dtype, fp8_probe.reason)

    tune_cfg = cfg_get(cfg, "normative.batch_tuning", {})
    tuned_batch_size = int(norm_cfg.get("batch_size", 16384))
    if bool(tune_cfg.get("enabled", True)):
        tuner = auto_tune_batch_size(
            initial_batch_size=tuned_batch_size,
            min_batch_size=int(tune_cfg.get("min_batch_size", 256)),
            max_batch_size=int(tune_cfg["max_batch_size"]) if "max_batch_size" in tune_cfg else None,
            target_low=float(tune_cfg.get("target_low", 0.85)),
            target_high=float(tune_cfg.get("target_high", 0.92)),
            growth=float(tune_cfg.get("growth", 1.35)),
            backoff=float(tune_cfg.get("backoff", 0.90)),
            max_trials=int(tune_cfg.get("max_trials", 8)),
            probe_fn=lambda b: _probe_vram_fraction_for_normative_batch(
                batch_size=b,
                x_train=x_train,
                y_train=y_train,
                cfg=norm_cfg,
                amp_dtype=amp_dtype,
            ),
            logger=logger,
        )
        tuned_batch_size = int(tuner.batch_size)

    norm_cfg["batch_size"] = tuned_batch_size

    model_dir = args.out_root / "models" / "normative" / run_id
    report_dir = args.out_root / "reports" / "normative" / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    h5_out = report_dir / "normative_outputs.h5"
    metrics_out = report_dir / "normative_metrics.json"
    rt_summary_out = report_dir / "rt_linkage_summary.json"
    if args.resume and h5_out.exists() and metrics_out.exists() and rt_summary_out.exists():
        logger.info("Found completed outputs for run_id=%s; skipping due --resume", run_id)
        return

    gpu_logger = None
    if args.gpu_log_csv is not None:
        tag = args.gpu_log_tag or f"module04_seed{seed}"
        gpu_logger = start_gpu_util_logger(csv_path=args.gpu_log_csv, tag=tag, logger=logger, interval_s=1.0)

    try:
        models = []
        ensemble_size = int(norm_cfg.get("ensemble_size", 5))
        for k in range(ensemble_size):
            logger.info("Training ensemble member %d/%d", k + 1, ensemble_size)
            m = train_one_member(
                member_idx=k,
                x_train=x_train,
                y_train=y_train,
                cfg=norm_cfg,
                out_dir=model_dir,
                device=device,
                amp_dtype=amp_dtype,
                compile_model=bool(norm_cfg.get("compile", True)),
                resume=args.resume,
                data_seed=seed + 10000,
                logger=logger,
            )
            models.append(m)

        norm_params = {
            "x_mu": x_mu.tolist(),
            "x_sd": x_sd.tolist(),
            "y_mu": y_mu,
            "y_sd": y_sd,
            "y_scale": "microvolts",
            "features": ["memory_load", "age", "trial_order"],
            "seed": seed,
            "batch_size": tuned_batch_size,
            "fp8_probe": fp8_probe.reason,
            "bootstrap": bootstrap_meta,
        }
        (model_dir / "normalization.json").write_text(json.dumps(norm_params, indent=2), encoding="utf-8")

        # Fixed evaluation sets (healthy full + optional clinical), trained with seed-bootstrap.
        age_fill = float(np.nanmean(x_train_np[:, 1])) if np.isfinite(x_train_np[:, 1]).any() else 0.0

        def apply(df: pd.DataFrame, label: str) -> pd.DataFrame:
            if df.empty:
                return df
            y = (df["p3b_amp"].to_numpy(dtype=float) * 1e6).astype(np.float32)
            x = df[["memory_load", "age", "trial_order"]].to_numpy(dtype=float).astype(np.float32)
            x[np.isnan(x[:, 1]), 1] = age_fill

            xz = _standardize_apply(x, x_mu, x_sd).astype(np.float32)
            xt = torch.from_numpy(xz).to(device, non_blocking=True)

            mu_z, var_z = _predict_in_batches(
                models,
                xt,
                batch_size=int(norm_cfg.get("batch_size", 16384)),
                amp_dtype=amp_dtype,
            )

            mu = (mu_z.cpu().numpy() * y_sd + y_mu).astype(np.float32)
            var = (var_z.cpu().numpy() * (y_sd**2)).astype(np.float32)
            z = trial_zscore(y=y, mu=mu, var=var).astype(np.float32)

            out = df.copy()
            out["p3b_uV"] = y
            out["mu_uV"] = mu
            out["var_uV"] = var
            out["z"] = z
            out["cohort"] = label
            return out

        df_h_pred = apply(df_h, "healthy")
        df_c_pred = apply(df_c, "clinical") if not df_c.empty else pd.DataFrame()

        seed_effect_checksum = checksum_array_sha256(df_h_pred["mu_uV"].to_numpy(dtype=np.float32))

        df_h_pred.to_csv(report_dir / "healthy_trial_predictions.csv", index=False)
        if not df_c_pred.empty:
            df_c_pred.to_csv(report_dir / "clinical_trial_predictions.csv", index=False)

        with h5py.File(h5_out, "w") as f:
            _df_to_h5(f, "healthy_trial_predictions", df_h_pred)
            if not df_c_pred.empty:
                _df_to_h5(f, "clinical_trial_predictions", df_c_pred)
            for k, v in norm_params.items():
                f.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
            f.attrs["seed_effect_checksum"] = seed_effect_checksum
            f.flush()
        logger.info("Wrote HDF5 outputs: %s", h5_out)

        if not df_c_pred.empty:
            df_subj = subject_load_deviation_score(df_c_pred)
            if args.severity_csv and args.severity_csv.exists():
                sev = pd.read_csv(args.severity_csv)
                if "subject" in sev.columns and args.severity_col in sev.columns:
                    df_subj = df_subj.merge(sev[["subject", args.severity_col]], on="subject", how="left")
                    df_subj = df_subj.rename(columns={args.severity_col: "severity"})

            df_subj.to_csv(report_dir / "clinical_subject_deviation_scores.csv", index=False)
            with h5py.File(h5_out, "a") as f:
                _df_to_h5(f, "clinical_subject_deviation_scores", df_subj)
                f.flush()

            reg = robust_regress_severity(df_subj, score_col="z_hi_minus_lo", severity_col="severity", covariates=("age",))
            if reg is not None:
                (report_dir / "clinical_robust_regression.json").write_text(json.dumps(reg, indent=2), encoding="utf-8")
                with h5py.File(h5_out, "a") as f:
                    f.attrs["clinical_robust_regression"] = json.dumps(reg)
                    f.flush()
            else:
                logger.warning("Skipped clinical regression (missing/insufficient severity data).")

        linkage_out = {}
        for label, dfp in [("healthy", df_h_pred), ("clinical", df_c_pred)]:
            if dfp.empty:
                continue
            if dfp["rt"].notna().sum() < 50:
                continue
            summary = margin_to_next_rt(dfp.assign(p3b_amp=dfp["p3b_uV"], subject=dfp["subject_key"]))
            df_slopes = summary["df_subject"]
            df_slopes.to_csv(report_dir / f"rt_linkage_{label}_per_subject.csv", index=False)
            linkage_out[label] = {
                "n_subjects": summary["n_subjects"],
                "median_beta_margin": summary["median_beta_margin"],
                "mean_beta_margin": summary["mean_beta_margin"],
            }

        rt_summary = {
            "run_id": run_id,
            "seed": seed,
            "linkage": linkage_out,
        }
        rt_summary_out.write_text(json.dumps(rt_summary, indent=2), encoding="utf-8")
        with h5py.File(h5_out, "a") as f:
            f.attrs["rt_linkage_summary"] = json.dumps(rt_summary)
            f.flush()

        metrics = {
            "run_id": run_id,
            "seed": seed,
            "batch_size": tuned_batch_size,
            "seed_effect_checksum": seed_effect_checksum,
            "bootstrap": bootstrap_meta,
            "healthy": {
                "n_trials": int(len(df_h_pred)),
                "n_subjects": int(df_h_pred["subject_key"].nunique()),
                "nll": _mean_gaussian_nll(df_h_pred["p3b_uV"], df_h_pred["mu_uV"], df_h_pred["var_uV"]),
                "calibration": _calibration_summary(df_h_pred["z"].to_numpy()),
                "z_stability": _z_stability(df_h_pred),
                "subject_key_counts_by_dataset": {
                    str(k): int(v)
                    for k, v in df_h_pred.groupby("dataset_id")["subject_key"].nunique().to_dict().items()
                },
            },
            "clinical": {
                "n_trials": int(len(df_c_pred)),
                "n_subjects": int(df_c_pred["subject_key"].nunique()) if not df_c_pred.empty else 0,
                "nll": _mean_gaussian_nll(df_c_pred["p3b_uV"], df_c_pred["mu_uV"], df_c_pred["var_uV"])
                if not df_c_pred.empty
                else float("nan"),
                "calibration": _calibration_summary(df_c_pred["z"].to_numpy()) if not df_c_pred.empty else _calibration_summary(np.array([])),
                "z_stability": _z_stability(df_c_pred) if not df_c_pred.empty else _z_stability(pd.DataFrame()),
            },
            "allow_deterministic": bool(args.allow_deterministic),
        }
        metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        logger.info("Wrote normative metrics: %s", metrics_out)
        logger.info("Wrote RT linkage summary: %s", rt_summary_out)
        logger.info("Normative module complete. Outputs in %s", report_dir)

    finally:
        if gpu_logger is not None:
            gpu_logger.stop()


if __name__ == "__main__":
    main()
