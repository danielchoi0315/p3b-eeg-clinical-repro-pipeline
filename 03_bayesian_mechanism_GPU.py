#!/usr/bin/env python3
"""03_bayesian_mechanism_GPU.py

GPU module: hierarchical Bayesian mediation model (Load -> PDR -> P3b) via Torch VI.

Additions in this revision:
- GH200/H100-safe hardware runtime policy (no hardcoded SM12 arch)
- NVML utilization logging (1 Hz) -> gpu_util.csv
- seed override and dataset filtering for multi-seed sweeps
- automatic batch-size tuning to ~85-92% VRAM with OOM-safe backoff
- optional CUDA prefetch stream
- torch.compile fallback logging
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import time

import h5py
import numpy as np
import torch

from common.hardware import (
    PrefetchToDeviceLoader,
    apply_cpu_thread_env,
    auto_tune_batch_size,
    configure_cuda_arch_list,
    configure_torch_backends,
    detect_hardware_info,
    start_gpu_util_logger,
)
from p3b_pipeline.bayes_mediation import HierarchicalMediationVI, MediationBatch
from p3b_pipeline.config import cfg_get, load_yaml
from p3b_pipeline.env import ThreadConfig, apply_thread_config, cuda_device_summary, require_cuda_or_die
from p3b_pipeline.h5io import iter_subject_feature_files, read_subject_h5
from p3b_pipeline.logging_utils import configure_logging
from p3b_pipeline.manifest import write_manifest
from p3b_pipeline.torch_utils import safe_compile, seed_all, select_amp_dtype


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--run_id", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--only_cohort", type=str, default=None, help="Filter by attrs['cohort']")
    ap.add_argument("--only_dataset", type=str, default=None, help="Filter by attrs['dataset_id']")
    ap.add_argument("--gpu_log_csv", type=Path, default=None)
    ap.add_argument("--gpu_log_tag", type=str, default="")
    ap.add_argument("--enable_fp8", action="store_true", help="Enable FP8 probe + autocast if stable")
    return ap.parse_args()


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _dataset_from_attrs_or_path(attrs: dict, fp: Path, features_root: Path) -> str:
    if "dataset_id" in attrs and attrs["dataset_id"]:
        return str(attrs["dataset_id"])
    try:
        rel = fp.relative_to(features_root)
        return rel.parts[0]
    except Exception:
        return "unknown"


def _probe_vram_fraction_for_batch(
    *,
    batch_size: int,
    n_subjects: int,
    tensors: tuple[torch.Tensor, ...],
    amp_dtype: torch.dtype,
) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = HierarchicalMediationVI(n_subjects=n_subjects).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=2e-3)

    bs = min(batch_size, tensors[0].shape[0])
    idx = torch.arange(bs, dtype=torch.long)
    t_subj, t_load, t_age, t_order, t_pdr, t_p3b = tensors

    batch = MediationBatch(
        subj_idx=t_subj[idx].cuda(non_blocking=True),
        load_z=t_load[idx].cuda(non_blocking=True),
        age_z=t_age[idx].cuda(non_blocking=True),
        order_z=t_order[idx].cuda(non_blocking=True),
        pdr_z=t_pdr[idx].cuda(non_blocking=True),
        p3b_z=t_p3b[idx].cuda(non_blocking=True),
    )

    optim.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        loss = model(batch, scale_factor=1.0)
    loss.backward()
    optim.step()

    peak = float(torch.cuda.max_memory_allocated())
    total = float(torch.cuda.get_device_properties(0).total_memory)

    del model
    del optim
    del batch
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return peak / max(total, 1.0)


def _to_cuda_if_needed(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t if t.device.type == "cuda" else t.to(device, non_blocking=True)


def main() -> None:
    args = parse_args()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

    # Keep this early for NumPy/MKL consumers.
    apply_cpu_thread_env(allow_override=True)
    apply_thread_config(ThreadConfig(), allow_override=True)

    require_cuda_or_die()
    arch_action = configure_cuda_arch_list(for_extension_build=False)
    configure_torch_backends(enable_tf32=True)

    cfg = load_yaml(args.config)
    bayes_cfg = cfg_get(cfg, "bayes_mediation", {})

    log_dir = args.out_root / "logs"
    logger = configure_logging(log_dir=log_dir, run_id=run_id, name="03_bayesian_mechanism_GPU")

    manifest_path = write_manifest(
        out_dir=log_dir,
        run_id=run_id,
        entrypoint="03_bayesian_mechanism_GPU",
        args={k: str(v) for k, v in vars(args).items()},
        extra={
            "cuda": cuda_device_summary(),
            "cuda_arch_action": arch_action,
            "hardware": detect_hardware_info().__dict__,
        },
    )
    logger.info("Wrote manifest: %s", manifest_path)
    logger.info("Using device: %s", cuda_device_summary())

    seed = int(args.seed if args.seed is not None else bayes_cfg.get("seed", 0))
    seed_all(seed)

    files = iter_subject_feature_files(args.features_root)
    if not files:
        raise RuntimeError(f"No feature files found under {args.features_root}")

    subj_to_idx: dict[str, int] = {}
    rows = []
    n_files_kept = 0

    for fp in files:
        arrays, attrs = read_subject_h5(fp)

        cohort = str(attrs.get("cohort", ""))
        if args.only_cohort is not None and cohort != str(args.only_cohort):
            continue

        dataset_id = _dataset_from_attrs_or_path(attrs, fp, args.features_root)
        if args.only_dataset is not None and dataset_id != str(args.only_dataset):
            continue

        p3b = arrays.get("p3b_amp")
        pdr = arrays.get("pdr")
        load = arrays.get("memory_load")
        age = arrays.get("age")
        order = arrays.get("trial_order")

        if p3b is None or pdr is None or load is None or order is None:
            continue
        if np.all(~np.isfinite(pdr)):
            continue

        bids_sub = str(attrs.get("bids_subject", attrs.get("subject", "unknown")))
        bids_ses = str(attrs.get("bids_session", attrs.get("session", "")))
        sid = str(
            attrs.get(
                "subject_key",
                f"{dataset_id}:sub-{bids_sub}" + (f":ses-{bids_ses}" if bids_ses and bids_ses.lower() not in {"na", "none"} else ""),
            )
        )
        if sid not in subj_to_idx:
            subj_to_idx[sid] = len(subj_to_idx)

        n = len(p3b)
        rows.append(
            {
                "subj_idx": np.full(n, subj_to_idx[sid], dtype=np.int64),
                "load": load.astype(np.float32),
                "age": age.astype(np.float32) if age is not None else np.full(n, np.nan, dtype=np.float32),
                "order": order.astype(np.float32),
                "pdr": pdr.astype(np.float32),
                "p3b": p3b.astype(np.float32),
                "dataset_id": np.asarray([dataset_id] * n),
            }
        )
        n_files_kept += 1

    if not rows:
        raise RuntimeError("No usable feature rows for mediation (need p3b_amp + pdr + load).")
    logger.info("Loaded %d file(s), %d subject(s) for mediation.", n_files_kept, len(subj_to_idx))

    subj_idx = np.concatenate([r["subj_idx"] for r in rows], axis=0)
    load = np.concatenate([r["load"] for r in rows], axis=0)
    age = np.concatenate([r["age"] for r in rows], axis=0)
    order = np.concatenate([r["order"] for r in rows], axis=0)
    pdr = np.concatenate([r["pdr"] for r in rows], axis=0)
    p3b = np.concatenate([r["p3b"] for r in rows], axis=0)

    m = np.isfinite(load) & np.isfinite(order) & np.isfinite(pdr) & np.isfinite(p3b) & np.isfinite(subj_idx)
    age_m = age.copy()
    if np.isfinite(age_m).any():
        age_m[~np.isfinite(age_m)] = float(np.nanmean(age_m))
    else:
        age_m[:] = 0.0

    subj_idx = subj_idx[m]
    load = load[m]
    age_m = age_m[m]
    order = order[m]
    pdr = pdr[m]
    p3b = p3b[m]

    def z(x: np.ndarray) -> np.ndarray:
        mu = float(np.mean(x))
        sd = float(np.std(x) + 1e-8)
        return (x - mu) / sd

    t_subj = torch.from_numpy(subj_idx.astype(np.int64))
    t_load = torch.from_numpy(z(load).astype(np.float32))
    t_age = torch.from_numpy(z(age_m).astype(np.float32))
    t_order = torch.from_numpy(z(order).astype(np.float32))
    t_pdr = torch.from_numpy(z(pdr).astype(np.float32))
    t_p3b = torch.from_numpy(z(p3b).astype(np.float32))

    ds = torch.utils.data.TensorDataset(t_subj, t_load, t_age, t_order, t_pdr, t_p3b)

    base_batch_size = int(bayes_cfg.get("batch_size", 8192))
    num_workers = int(bayes_cfg.get("num_workers", 8))
    max_steps = int(bayes_cfg.get("max_steps", 20000))

    amp_dtype, fp8_probe = select_amp_dtype(
        str(bayes_cfg.get("amp", "bf16")), enable_fp8=bool(args.enable_fp8 or bayes_cfg.get("enable_fp8", False))
    )
    logger.info("AMP dtype=%s fp8_probe=%s", amp_dtype, fp8_probe.reason)

    tune_cfg = cfg_get(cfg, "bayes_mediation.batch_tuning", {})
    n_total = len(ds)
    if n_total <= 0:
        raise RuntimeError("No mediation training rows after filtering.")
    max_batch_by_data = max(1, int(n_total))
    tune_max_cfg = tune_cfg.get("max_batch_size", None)
    tune_max = max_batch_by_data if tune_max_cfg is None else min(int(tune_max_cfg), max_batch_by_data)
    tune_min = min(int(tune_cfg.get("min_batch_size", 256)), max_batch_by_data)
    tuned_batch_size = min(base_batch_size, max_batch_by_data)
    if bool(tune_cfg.get("enabled", True)):
        tuner = auto_tune_batch_size(
            initial_batch_size=min(base_batch_size, max_batch_by_data),
            min_batch_size=max(1, tune_min),
            max_batch_size=max(1, tune_max),
            target_low=float(tune_cfg.get("target_low", 0.85)),
            target_high=float(tune_cfg.get("target_high", 0.92)),
            growth=float(tune_cfg.get("growth", 1.35)),
            backoff=float(tune_cfg.get("backoff", 0.90)),
            max_trials=int(tune_cfg.get("max_trials", 8)),
            probe_fn=lambda b: _probe_vram_fraction_for_batch(
                batch_size=b,
                n_subjects=len(subj_to_idx),
                tensors=(t_subj, t_load, t_age, t_order, t_pdr, t_p3b),
                amp_dtype=amp_dtype,
            ),
            logger=logger,
        )
        tuned_batch_size = int(tuner.batch_size)
    tuned_batch_size = max(1, min(int(tuned_batch_size), max_batch_by_data))

    # Use a replacement sampler to avoid iterator-reset overhead and keep an
    # effectively endless training stream for fixed-step VI optimization.
    sampler = torch.utils.data.RandomSampler(
        ds,
        replacement=True,
        num_samples=max(int(max_steps * tuned_batch_size), int(n_total)),
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=tuned_batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    device = torch.device("cuda")
    prefetch_enabled = bool(bayes_cfg.get("prefetch_to_gpu", True))
    stream_loader = PrefetchToDeviceLoader(dl, device=device, enabled=prefetch_enabled)

    model = HierarchicalMediationVI(n_subjects=len(subj_to_idx)).to(device)
    model = safe_compile(
        model,
        enabled=bool(bayes_cfg.get("compile", True)),
        backend="inductor",
        logger=logger,
        context="module03.mediation",
    )

    optim = torch.optim.Adam(model.parameters(), lr=float(bayes_cfg.get("lr", 2e-3)))

    ckpt_dir = args.out_root / "checkpoints" / "mediation" / run_id
    ckpt_path = ckpt_dir / "checkpoint.pt"

    report_dir = args.out_root / "reports" / "mediation" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "mediation_summary.json"
    h5_path = report_dir / "mediation_effect_samples.h5"

    if args.resume and summary_path.exists() and h5_path.exists() and ckpt_path.exists():
        logger.info("Found completed outputs and checkpoint for run_id=%s; skipping due --resume", run_id)
        return

    step0 = 0
    if args.resume and ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(payload["model_state"])
        optim.load_state_dict(payload["optim_state"])
        step0 = int(payload.get("step", 0))
        logger.info("Resumed from checkpoint: %s (step=%d)", ckpt_path, step0)

    grad_clip = float(bayes_cfg.get("grad_clip", 5.0))
    scale_factor = n_total / float(max(tuned_batch_size, 1))
    if len(dl) <= 0:
        raise RuntimeError(
            f"Empty DataLoader for mediation training (n_total={n_total}, batch_size={tuned_batch_size}, drop_last=True)."
        )

    gpu_logger = None
    if args.gpu_log_csv is not None:
        tag = args.gpu_log_tag or f"module03_seed{seed}"
        gpu_logger = start_gpu_util_logger(csv_path=args.gpu_log_csv, tag=tag, logger=logger, interval_s=1.0)

    logger.info(
        "Training VI: steps=%d batch=%d scale_factor=%.3f prefetch_to_gpu=%s",
        max_steps,
        tuned_batch_size,
        scale_factor,
        prefetch_enabled,
    )

    dl_iter = iter(stream_loader)

    try:
        for step in range(step0, max_steps):
            try:
                b = next(dl_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    "Mediation training iterator exhausted unexpectedly before max_steps; "
                    f"step={step} max_steps={max_steps} batch_size={tuned_batch_size}"
                ) from exc

            subj_b, load_b, age_b, order_b, pdr_b, p3b_b = b
            batch = MediationBatch(
                subj_idx=_to_cuda_if_needed(subj_b, device),
                load_z=_to_cuda_if_needed(load_b, device),
                age_z=_to_cuda_if_needed(age_b, device),
                order_z=_to_cuda_if_needed(order_b, device),
                pdr_z=_to_cuda_if_needed(pdr_b, device),
                p3b_z=_to_cuda_if_needed(p3b_b, device),
            )

            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss = model(batch, scale_factor=scale_factor)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optim.step()

            if step % 50 == 0:
                logger.info("step=%d loss=%.6f", step, float(loss.detach().cpu()))

            if step % 500 == 0 and step > 0:
                atomic_torch_save(
                    ckpt_path,
                    {
                        "step": step,
                        "model_state": model.state_dict(),
                        "optim_state": optim.state_dict(),
                        "subj_to_idx": subj_to_idx,
                        "config": cfg,
                        "seed": seed,
                        "batch_size": tuned_batch_size,
                    },
                )
                logger.info("Checkpointed: %s", ckpt_path)

        atomic_torch_save(
            ckpt_path,
            {
                "step": max_steps,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "subj_to_idx": subj_to_idx,
                "config": cfg,
                "seed": seed,
                "batch_size": tuned_batch_size,
            },
        )

        with torch.no_grad():
            effects = model.sample_mediation_effect(n=20000, device=device).detach().cpu().numpy()

        summary = {
            "n_subjects": len(subj_to_idx),
            "n_trials": int(n_total),
            "effect_mean": float(np.mean(effects)),
            "effect_ci95": [float(np.quantile(effects, 0.025)), float(np.quantile(effects, 0.975))],
            "run_id": run_id,
            "seed": seed,
            "amp_dtype": str(amp_dtype),
            "fp8_probe": fp8_probe.reason,
            "batch_size": int(tuned_batch_size),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("a_times_b", data=effects.astype(np.float32), compression="gzip", compression_opts=4, shuffle=True)
            for k, v in summary.items():
                f.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
            f.flush()

        logger.info("Wrote mediation summary: %s", summary_path)
        logger.info("Wrote mediation samples: %s", h5_path)

    finally:
        if gpu_logger is not None:
            gpu_logger.stop()


if __name__ == "__main__":
    main()
