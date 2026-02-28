#!/usr/bin/env python3
"""02_extract_features_CPU.py

Extract single-trial features from preprocessed epochs (CPU):

- P3b amplitude + latency (locked definition; Pz fallback)
- Pupillometry PDR (baseline-corrected pupil response)
- Trial covariates: memory_load, age, trial_order
- Optional behavioral outcomes: RT, accuracy

Outputs:
- One HDF5 file per subject-run in `--features_root`
  (fault-tolerant; atomic writes; safe for sharding)

Threading:
- Same 24-core constraints as preprocess step.
"""

# ----------------------------
# IMPORTANT: set thread env *before* importing NumPy/pandas/MNE.
# ----------------------------
from common.hardware import apply_cpu_thread_env, conservative_thread_count

apply_cpu_thread_env(threads=conservative_thread_count(), allow_override=True)

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from common.lawc_audit import (
    align_probe_events_to_epochs,
    load_lawc_event_map,
    prepare_probe_event_table,
    subject_key_from_entities,
)
from p3b_pipeline.bids_utils import BIDSRun
from p3b_pipeline.config import load_yaml
from p3b_pipeline.eeg import extract_p3_features
from p3b_pipeline.env import ThreadConfig, apply_thread_config
from p3b_pipeline.h5io import atomic_write_subject_h5
from p3b_pipeline.logging_utils import configure_logging
from p3b_pipeline.manifest import write_manifest
from p3b_pipeline.pupil import extract_pdr, find_eyetrack_file, load_pupil_timeseries


STERNBERG_DATASETS = {"ds005095", "ds003655", "ds004117"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids_root", type=Path, required=True)
    ap.add_argument("--deriv_root", type=Path, required=True)
    ap.add_argument("--features_root", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--subjects", type=str, nargs="*", default=None)
    ap.add_argument("--cohort", type=str, default="healthy", help="healthy | clinical | mechanism | other")
    ap.add_argument("--dataset_id", type=str, default=None, help="Dataset identifier (default: basename of bids_root)")
    ap.add_argument("--run_id", type=str, default=None)
    ap.add_argument("--lawc_event_map", type=Path, default=Path("configs/lawc_event_map.yaml"))
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel run workers. 0=auto.",
    )
    ap.add_argument(
        "--per_run_threads",
        type=int,
        default=0,
        help="OMP/MKL threads per worker process. 0=auto.",
    )
    return ap.parse_args()


def _parse_entities_from_epochs_filename(p: Path) -> BIDSRun:
    """Parse subject/task/run/session from our own epochs filename convention."""
    # Example:
    #   sub-01_ses-na_task-digitspan_run-1_desc-epo.fif.gz
    name = p.name

    def _get(prefix: str) -> str:
        import re

        m = re.search(prefix + r"-([A-Za-z0-9]+)", name)
        return m.group(1) if m else "na"

    subject = _get("sub")
    ses = _get("ses")
    task = _get("task")
    run = _get("run")

    return BIDSRun(
        subject=subject,
        task=None if task == "na" else task,
        run=None if run == "na" else run,
        session=None if ses == "na" else ses,
        eeg_path=Path(""),  # not needed here
        events_tsv=None,
    )


def _auto_workers() -> int:
    cores = os.cpu_count() or 8
    if cores <= 8:
        return 2
    return max(2, min(24, cores // 2))


def _auto_per_run_threads(workers: int) -> int:
    cores = os.cpu_count() or 8
    return max(1, min(conservative_thread_count(), cores // max(1, workers)))


def _find_events_tsv_for_run(bids_root: Path, run: BIDSRun) -> Optional[Path]:
    sub_dir = bids_root / f"sub-{run.subject}"
    if run.session is not None:
        sub_dir = sub_dir / f"ses-{run.session}"
    eeg_dir = sub_dir / "eeg"
    if not eeg_dir.exists():
        return None

    ents = [f"sub-{run.subject}"]
    if run.session is not None:
        ents.append(f"ses-{run.session}")
    if run.task is not None:
        ents.append(f"task-{run.task}")
    if run.run is not None:
        ents.append(f"run-{run.run}")
    stem = "_".join(ents) + "_events"

    exact = [eeg_dir / f"{stem}.tsv", eeg_dir / f"{stem}.tsv.gz"]
    for p in exact:
        if p.exists():
            return p

    fuzzy = sorted(list(eeg_dir.glob(f"sub-{run.subject}*_events.tsv")) + list(eeg_dir.glob(f"sub-{run.subject}*_events.tsv.gz")))
    if run.task is not None:
        fuzzy = [p for p in fuzzy if f"task-{run.task}" in p.name]
    if run.session is not None:
        fuzzy = [p for p in fuzzy if f"ses-{run.session}" in p.name]
    if run.run is not None:
        fuzzy = [p for p in fuzzy if f"run-{run.run}" in p.name]
    if len(fuzzy) == 1:
        return fuzzy[0]
    return None


def _h5_rt_counts(path: Path) -> Dict[str, Any]:
    try:
        with h5py.File(path, "r") as h:
            p3b_n = int(np.asarray(h["p3b_amp"]).shape[0]) if "p3b_amp" in h else 0
            rt = np.asarray(h["rt"], dtype=float) if "rt" in h else np.asarray([], dtype=float)
            rt_nonmissing = int(np.isfinite(rt).sum())
            pdr_finite = int(np.isfinite(np.asarray(h["pdr"], dtype=float)).sum()) if "pdr" in h else 0
            return {
                "p3b_n": p3b_n,
                "rt_nonmissing": rt_nonmissing,
                "pdr_finite": pdr_finite,
            }
    except Exception:
        return {"p3b_n": 0, "rt_nonmissing": 0, "pdr_finite": 0}


def _extract_one(
    *,
    epo: Path,
    run: BIDSRun,
    out_h5: Path,
    bids_root: Path,
    config: Dict[str, Any],
    cohort: str,
    dataset_id: str,
    config_path: Path,
    per_run_threads: int,
    lawc_event_map: Dict[str, Any],
) -> Dict[str, Any]:
    apply_cpu_thread_env(threads=per_run_threads, allow_override=False)
    apply_thread_config(ThreadConfig(), allow_override=True)

    if out_h5.exists():
        counts = _h5_rt_counts(out_h5)
        if cohort == "mechanism" and counts["pdr_finite"] == 0:
            try:
                out_h5.unlink()
            except Exception:
                return {"status": "error", "path": str(out_h5), "traceback": "Could not remove stale H5 file."}
        else:
            return {
                "status": "skip_exists",
                "path": str(out_h5),
                "pdr_finite": counts["pdr_finite"],
                "n_selected": counts["p3b_n"],
                "rt_nonmissing": counts["rt_nonmissing"],
            }

    try:
        if not dataset_id:
            raise RuntimeError("dataset_id is empty in feature extraction")

        feats = extract_p3_features(epo, config)
        if int(np.asarray(feats["p3b_amp"]).size) == 0:
            return {"status": "skip_empty", "path": str(out_h5)}

        # Pupil (best-effort): if not present, we still write EEG-only features.
        pdr = np.full_like(feats["p3b_amp"], np.nan, dtype=np.float32)
        eyetrack = find_eyetrack_file(
            bids_root=bids_root,
            subject=run.subject,
            task=run.task,
            run=run.run,
            session=run.session,
        )
        if eyetrack is not None:
            try:
                t_s, pup = load_pupil_timeseries(eyetrack, config)
                pdr = extract_pdr(onset_s=feats["onset_s"].astype(float), time_s=t_s, pupil=pup, cfg=config)
            except Exception:
                # Keep NaN pdr; caller logs as warning through status payload.
                pass
        elif cohort == "mechanism":
            return {"status": "skip_no_pupil", "path": str(out_h5)}

        pdr_finite = int(np.isfinite(pdr).sum())
        if cohort == "mechanism" and pdr_finite == 0:
            return {"status": "skip_no_pupil", "path": str(out_h5)}

        # Apply explicit Law-C event filter/load/RT mapping for Sternberg-family datasets.
        diag: Dict[str, Any] = {
            "rt_source": "none",
            "rt_columns_tried": [],
            "events_columns": [],
            "event_filter": "",
            "load_column_used": "",
            "rt_nonmissing_rate": float(np.isfinite(feats["rt"]).mean()) if "rt" in feats else float("nan"),
        }
        dataset_specs = (lawc_event_map.get("datasets", {}) or {})
        if dataset_id in dataset_specs:
            events_path = _find_events_tsv_for_run(bids_root, run)
            if events_path is None:
                raise RuntimeError(
                    f"Dataset {dataset_id}: could not locate events.tsv for sub-{run.subject} "
                    f"ses={run.session} task={run.task} run={run.run}"
                )

            probe_df, diag = prepare_probe_event_table(
                events_path=events_path,
                dataset_id=dataset_id,
                event_map=lawc_event_map,
                dataset_root=bids_root,
                bids_subject=run.subject,
                bids_task=run.task,
                bids_run=run.run,
                bids_session=run.session,
            )

            aligned = align_probe_events_to_epochs(
                epoch_onsets_s=np.asarray(feats["onset_s"], dtype=float),
                probe_df=probe_df,
                tolerance_s=0.012,
            )
            matched = aligned["matched"].to_numpy(dtype=bool)
            if int(matched.sum()) == 0:
                raise RuntimeError(
                    f"Dataset {dataset_id}: no epoch onsets matched probe events for sub-{run.subject} "
                    f"ses={run.session} task={run.task} run={run.run}"
                )

            for key in ["p3b_amp", "p3b_lat", "p3b_channel", "memory_load", "trial_order", "onset_s", "age", "rt", "accuracy"]:
                feats[key] = np.asarray(feats[key])[matched]
            pdr = np.asarray(pdr)[matched]

            # Overwrite trial metadata with fail-closed event mapping.
            feats["memory_load"] = aligned.loc[matched, "memory_load"].to_numpy(dtype=np.float32)
            feats["trial_order"] = aligned.loc[matched, "trial_order"].to_numpy(dtype=np.int32)
            feats["onset_s"] = aligned.loc[matched, "probe_onset_s"].to_numpy(dtype=np.float32)
            feats["rt"] = aligned.loc[matched, "rt"].to_numpy(dtype=np.float32)
            feats["accuracy"] = aligned.loc[matched, "accuracy"].to_numpy(dtype=np.float32)

        subject_key = subject_key_from_entities(dataset_id=dataset_id, bids_subject=run.subject, bids_session=run.session)

        n = int(np.asarray(feats["p3b_amp"]).shape[0])
        if n == 0:
            return {"status": "skip_empty", "path": str(out_h5)}

        arrays: Dict[str, Any] = {
            **{k: v for k, v in feats.items() if k not in {"p3b_channel"}},
            "p3b_channel": np.asarray(feats["p3b_channel"]).astype("U"),
            "pdr": np.asarray(pdr).astype(np.float32),
            "subject_key": np.asarray([subject_key] * n, dtype=object),
            "dataset_id": np.asarray([dataset_id] * n, dtype=object),
            "bids_subject": np.asarray([run.subject] * n, dtype=object),
            "bids_session": np.asarray([(run.session or "")] * n, dtype=object),
            "bids_run": np.asarray([(run.run or "")] * n, dtype=object),
        }

        attrs = {
            "subject": run.subject,
            "subject_key": subject_key,
            "bids_subject": run.subject,
            "bids_session": run.session,
            "bids_run": run.run,
            "task": run.task,
            "run": run.run,
            "session": run.session,
            "cohort": cohort,
            "dataset_id": dataset_id,
            "epochs_fif": str(epo),
            "bids_root": str(bids_root),
            "config": str(config_path),
            "lawc_event_filter": str(diag.get("event_filter", "")),
            "lawc_load_column": str(diag.get("load_column_used", "")),
            "lawc_load_sign": float(diag.get("load_sign", 1.0)),
            "lawc_rt_source": str(diag.get("rt_source", "none")),
            "lawc_rt_nonmissing_rate": float(diag.get("rt_nonmissing_rate", np.nan)),
        }

        atomic_write_subject_h5(out_h5, arrays=arrays, attrs=attrs)

        rt_nonmissing = int(np.isfinite(np.asarray(arrays["rt"], dtype=float)).sum()) if "rt" in arrays else 0
        return {
            "status": "done",
            "path": str(out_h5),
            "pdr_finite": int(np.isfinite(np.asarray(arrays["pdr"], dtype=float)).sum()),
            "n_selected": int(n),
            "rt_nonmissing": int(rt_nonmissing),
            "rt_source": str(diag.get("rt_source", "none")),
            "rt_columns_tried": list(diag.get("rt_columns_tried", [])),
            "events_columns": list(diag.get("events_columns", [])),
        }
    except Exception:
        tb = traceback.format_exc()
        low = tb.lower()
        # Fail-closed at run level: if mapped probe events are absent/invalid for a run/session,
        # skip that run and preserve traceback context in logs/artifacts.
        if (
            "event_filter selected zero rows" in low
            or "no epoch onsets matched probe events" in low
            or "sourcedata probe count mismatch" in low
        ):
            return {
                "status": "skip_mapping_empty",
                "path": str(out_h5),
                "traceback": tb,
            }
        return {
            "status": "error",
            "path": str(out_h5),
            "traceback": tb,
        }


def main() -> None:
    args = parse_args()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

    apply_thread_config(ThreadConfig(), allow_override=True)
    cfg = load_yaml(args.config)
    dataset_id = args.dataset_id or args.bids_root.name
    if not str(dataset_id).strip():
        raise RuntimeError("dataset_id resolved to empty string")

    lawc_event_map = load_lawc_event_map(args.lawc_event_map)
    rt_min_rate = float((lawc_event_map.get("defaults", {}) or {}).get("min_rt_nonmissing_rate", 0.5))

    log_dir = args.features_root / "logs"
    logger = configure_logging(log_dir=log_dir, run_id=run_id, name="02_extract_features_CPU")

    manifest_path = write_manifest(
        out_dir=log_dir,
        run_id=run_id,
        entrypoint="02_extract_features_CPU",
        args={k: str(v) for k, v in vars(args).items()},
        extra={"note": "feature extraction to per-subject HDF5"},
    )
    logger.info("Wrote manifest: %s", manifest_path)

    workers = int(args.workers) if int(args.workers) > 0 else _auto_workers()
    per_run_threads = int(args.per_run_threads) if int(args.per_run_threads) > 0 else _auto_per_run_threads(workers)
    logger.info(
        "Parallel config: workers=%d per_run_threads=%d (cpu_cores=%s)",
        workers,
        per_run_threads,
        os.cpu_count(),
    )

    epochs_root = args.deriv_root / "epochs"
    epochs_files = sorted(epochs_root.glob("sub-*/sub-*_desc-epo.fif.gz"))
    if not epochs_files:
        raise RuntimeError(f"No epochs files found under {epochs_root}")

    n_total = 0
    n_done = 0
    n_skip = 0
    n_err = 0
    n_mechanism_with_pdr = 0

    rt_trials_total = 0
    rt_nonmissing_total = 0
    rt_sources: Dict[str, int] = {}
    rt_columns_tried_union: set[str] = set()

    subject_filter = set(args.subjects) if args.subjects else None
    jobs: List[Dict[str, object]] = []
    for epo in epochs_files:
        run = _parse_entities_from_epochs_filename(epo)
        if subject_filter is not None and run.subject not in subject_filter:
            continue
        task = run.task or "na"
        run_ent = run.run or "na"
        ses = run.session or "na"
        out_dir = args.features_root / dataset_id / f"sub-{run.subject}"
        out_h5 = out_dir / f"sub-{run.subject}_ses-{ses}_task-{task}_run-{run_ent}_features.h5"
        jobs.append({"epo": epo, "run": run, "out_h5": out_h5})

    n_total = len(jobs)
    if n_total == 0:
        raise RuntimeError("No runs selected for feature extraction.")

    def _handle_result(job: Dict[str, object], res: Dict[str, Any]) -> None:
        nonlocal n_done, n_skip, n_err, n_mechanism_with_pdr
        nonlocal rt_trials_total, rt_nonmissing_total

        status = str(res.get("status", ""))
        if status == "done":
            n_done += 1
            if args.cohort == "mechanism" and int(res.get("pdr_finite", 0)) > 0:
                n_mechanism_with_pdr += 1
            logger.info("WROTE features: %s", res.get("path"))
        elif status in {"skip_exists", "skip_no_pupil", "skip_empty", "skip_mapping_empty"}:
            n_skip += 1
            if status == "skip_exists" and args.cohort == "mechanism" and int(res.get("pdr_finite", 0)) > 0:
                n_mechanism_with_pdr += 1
            if status == "skip_no_pupil":
                logger.warning("SKIP missing pupil/eyetrack for mechanism run: %s", res.get("path"))
            elif status == "skip_empty":
                logger.info("SKIP empty-epoch features: %s", res.get("path"))
            elif status == "skip_mapping_empty":
                logger.warning("SKIP mapped-probe-empty features: %s", res.get("path"))
        else:
            n_err += 1
            logger.error("ERROR extracting features from %s", job.get("epo"))
            logger.error("%s", res.get("traceback", "<no traceback>"))

        if dataset_id in STERNBERG_DATASETS:
            rt_trials_total += int(res.get("n_selected", 0))
            rt_nonmissing_total += int(res.get("rt_nonmissing", 0))
            src = str(res.get("rt_source", "none"))
            rt_sources[src] = rt_sources.get(src, 0) + 1
            for c in res.get("rt_columns_tried", []) or []:
                rt_columns_tried_union.add(str(c))

    if workers <= 1:
        logger.info("Running sequential feature extraction (%d run(s))", n_total)
        for job in jobs:
            run = job["run"]  # type: ignore[assignment]
            res = _extract_one(
                epo=job["epo"],  # type: ignore[arg-type]
                run=run,  # type: ignore[arg-type]
                out_h5=job["out_h5"],  # type: ignore[arg-type]
                bids_root=args.bids_root,
                config=cfg,
                cohort=args.cohort,
                dataset_id=dataset_id,
                config_path=args.config,
                per_run_threads=per_run_threads,
                lawc_event_map=lawc_event_map,
            )
            _handle_result(job, res)
    else:
        logger.info("Running parallel feature extraction for %d run(s) with %d workers", n_total, workers)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut_to_job = {
                ex.submit(
                    _extract_one,
                    epo=job["epo"],  # type: ignore[arg-type]
                    run=job["run"],  # type: ignore[arg-type]
                    out_h5=job["out_h5"],  # type: ignore[arg-type]
                    bids_root=args.bids_root,
                    config=cfg,
                    cohort=args.cohort,
                    dataset_id=dataset_id,
                    config_path=args.config,
                    per_run_threads=per_run_threads,
                    lawc_event_map=lawc_event_map,
                ): job
                for job in jobs
            }
            for fut in as_completed(fut_to_job):
                job = fut_to_job[fut]
                try:
                    res = fut.result()
                except Exception:
                    n_err += 1
                    logger.exception("ERROR extracting features from %s (worker crash)", job.get("epo"))
                    continue
                _handle_result(job, res)

    logger.info("Feature extraction complete. total=%d done=%d skip=%d err=%d", n_total, n_done, n_skip, n_err)
    if args.cohort == "mechanism" and n_mechanism_with_pdr == 0:
        raise RuntimeError("Mechanism cohort produced zero feature files with finite pupil PDR; cannot proceed.")

    if dataset_id in STERNBERG_DATASETS and rt_trials_total > 0:
        rt_rate = float(rt_nonmissing_total) / float(max(rt_trials_total, 1))
        logger.info(
            "RT completeness for %s: nonmissing=%d total=%d rate=%.3f",
            dataset_id,
            rt_nonmissing_total,
            rt_trials_total,
            rt_rate,
        )
        if rt_rate < rt_min_rate:
            raise RuntimeError(
                "RT fail-closed: insufficient non-missing RT for Law-C probe trials. "
                f"dataset={dataset_id} rate={rt_rate:.3f} threshold={rt_min_rate:.3f}; "
                f"rt_sources={json.dumps(rt_sources)}; rt_columns_tried={sorted(rt_columns_tried_union)}"
            )

    if n_err > 0:
        raise RuntimeError(f"Feature extraction failed for {n_err} run(s). See logs in {log_dir}")


if __name__ == "__main__":
    main()
