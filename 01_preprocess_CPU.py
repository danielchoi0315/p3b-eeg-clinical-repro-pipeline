#!/usr/bin/env python3
"""01_preprocess_CPU.py

CPU-bound preprocessing + epoching (MNE) with strict reproducibility.

Outputs:
- Per run: derivatives/epochs/.../*.fif (MNE Epochs with metadata)
- Logs + run manifest for auditability

Threading:
- We set OMP/MKL/OpenBLAS threads to 24 by default (no oversubscription).
"""

# ----------------------------
# IMPORTANT: set thread env *before* importing NumPy/MNE.
# ----------------------------
from common.hardware import apply_cpu_thread_env, conservative_thread_count

apply_cpu_thread_env(threads=conservative_thread_count(), allow_override=True)

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import os
import re
import time
import traceback
from typing import Dict, Optional

from p3b_pipeline.env import ThreadConfig, apply_thread_config
from p3b_pipeline.config import load_yaml
from p3b_pipeline.logging_utils import configure_logging
from p3b_pipeline.manifest import write_manifest
from p3b_pipeline.bids_utils import BIDSRun, iter_eeg_runs, load_participants, participant_age
from p3b_pipeline.eeg import preprocess_and_epoch_eeg


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids_root", type=Path, required=True)
    ap.add_argument("--deriv_root", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--subjects", type=str, nargs="*", default=None, help="Optional subject list (e.g., 01 02 03)")
    ap.add_argument("--run_id", type=str, default=None)
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
    ap.add_argument(
        "--mne_n_jobs",
        type=int,
        default=0,
        help="n_jobs passed to MNE filter/notch per run. 0=auto.",
    )
    return ap.parse_args()


def _auto_workers() -> int:
    cores = os.cpu_count() or 8
    if cores <= 8:
        return 2
    return max(2, min(12, cores // 4))


def _auto_per_run_threads(workers: int) -> int:
    cores = os.cpu_count() or 8
    return max(1, min(conservative_thread_count(), cores // max(1, workers)))


def _auto_mne_n_jobs(per_run_threads: int) -> int:
    # MNE filters scale to a handful of workers for these channel counts.
    return max(1, min(4, per_run_threads))


def _run_output_path(deriv_root: Path, run: BIDSRun) -> Path:
    out_dir = deriv_root / "epochs" / f"sub-{run.subject}"
    task = run.task or "na"
    run_ent = run.run or "na"
    ses = run.session or "na"
    return out_dir / f"sub-{run.subject}_ses-{ses}_task-{task}_run-{run_ent}_desc-epo.fif.gz"


def _process_one_run(
    *,
    run: BIDSRun,
    bids_root: Path,
    deriv_root: Path,
    cfg: Dict,
    age_years: Optional[float],
    per_run_threads: int,
    mne_n_jobs: int,
) -> Dict[str, str]:
    apply_cpu_thread_env(threads=per_run_threads, allow_override=False)
    apply_thread_config(ThreadConfig(), allow_override=True)

    out_epochs = _run_output_path(deriv_root, run)
    if out_epochs.exists():
        return {"status": "skip_exists", "path": str(out_epochs)}

    try:
        out = preprocess_and_epoch_eeg(
            bids_root=bids_root,
            run=run,
            cfg=cfg,
            age_years=age_years,
            out_epochs_fif=out_epochs,
            n_jobs=mne_n_jobs,
        )
        if out is None:
            return {"status": "skip", "path": str(out_epochs)}
        return {"status": "done", "path": str(out)}
    except Exception:
        tb = traceback.format_exc()
        # Known MNE EEGLAB parser incompatibility for some public datasets.
        # Skip run-level payloads that cannot be read instead of failing the whole dataset.
        if re.search(r"chaninfo.*no attribute 'get'|read_raw_eeglab", tb, flags=re.IGNORECASE):
            return {
                "status": "skip",
                "path": str(out_epochs),
                "traceback": tb,
                "skip_reason": "eeg_read_incompatible",
            }
        return {
            "status": "error",
            "path": str(out_epochs),
            "traceback": tb,
        }


def main():
    args = parse_args()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

    apply_thread_config(ThreadConfig(), allow_override=True)

    cfg = load_yaml(args.config)

    log_dir = args.deriv_root / "logs"
    logger = configure_logging(log_dir=log_dir, run_id=run_id, name="01_preprocess_CPU")

    manifest_path = write_manifest(
        out_dir=log_dir,
        run_id=run_id,
        entrypoint="01_preprocess_CPU",
        args={k: str(v) for k, v in vars(args).items()},
        extra={"note": "CPU preprocessing/epoching"},
    )
    logger.info("Wrote manifest: %s", manifest_path)

    workers = int(args.workers) if int(args.workers) > 0 else _auto_workers()
    per_run_threads = int(args.per_run_threads) if int(args.per_run_threads) > 0 else _auto_per_run_threads(workers)
    mne_n_jobs = int(args.mne_n_jobs) if int(args.mne_n_jobs) > 0 else _auto_mne_n_jobs(per_run_threads)
    # Guard against nested parallel deadlocks: ProcessPool workers + MNE joblib
    # inner pools can hang at teardown on long EEG batches.
    if workers > 1 and mne_n_jobs > 1:
        logger.warning(
            "workers=%d with mne_n_jobs=%d can deadlock via nested joblib; forcing mne_n_jobs=1",
            workers,
            mne_n_jobs,
        )
        mne_n_jobs = 1
    logger.info(
        "Parallel config: workers=%d per_run_threads=%d mne_n_jobs=%d (cpu_cores=%s)",
        workers,
        per_run_threads,
        mne_n_jobs,
        os.cpu_count(),
    )

    participants = load_participants(args.bids_root)
    ages_by_subject: Dict[str, Optional[float]] = {}
    for run in iter_eeg_runs(args.bids_root):
        if run.subject not in ages_by_subject:
            ages_by_subject[run.subject] = participant_age(participants, run.subject)

    subject_filter = set(args.subjects) if args.subjects else None
    runs = [r for r in iter_eeg_runs(args.bids_root) if (subject_filter is None or r.subject in subject_filter)]
    if not runs:
        raise RuntimeError(f"No EEG runs found under {args.bids_root}")

    n_total = len(runs)
    n_done = 0
    n_skip = 0
    n_err = 0

    if workers <= 1:
        logger.info("Running sequential preprocessing (%d run(s))", n_total)
        for run in runs:
            res = _process_one_run(
                run=run,
                bids_root=args.bids_root,
                deriv_root=args.deriv_root,
                cfg=cfg,
                age_years=ages_by_subject.get(run.subject),
                per_run_threads=per_run_threads,
                mne_n_jobs=mne_n_jobs,
            )
            status = res["status"]
            if status == "done":
                n_done += 1
                logger.info("DONE epochs: %s", res["path"])
            elif status in {"skip", "skip_exists"}:
                n_skip += 1
                why = str(res.get("skip_reason", "missing/unsupported annotations, non-memory task, or already complete"))
                logger.info(
                    "SKIP sub=%s task=%s run=%s (%s)",
                    run.subject,
                    run.task,
                    run.run,
                    why,
                )
            else:
                n_err += 1
                logger.error("ERROR preprocessing sub=%s task=%s run=%s", run.subject, run.task, run.run)
                logger.error("%s", res.get("traceback", "<no traceback>"))
    else:
        logger.info("Running parallel preprocessing for %d run(s) with %d workers", n_total, workers)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut_to_run = {
                ex.submit(
                    _process_one_run,
                    run=run,
                    bids_root=args.bids_root,
                    deriv_root=args.deriv_root,
                    cfg=cfg,
                    age_years=ages_by_subject.get(run.subject),
                    per_run_threads=per_run_threads,
                    mne_n_jobs=mne_n_jobs,
                ): run
                for run in runs
            }
            for fut in as_completed(fut_to_run):
                run = fut_to_run[fut]
                try:
                    res = fut.result()
                except Exception:
                    n_err += 1
                    logger.exception("ERROR preprocessing sub=%s task=%s run=%s (worker crash)", run.subject, run.task, run.run)
                    continue
                status = res["status"]
                if status == "done":
                    n_done += 1
                    logger.info("DONE epochs: %s", res["path"])
                elif status in {"skip", "skip_exists"}:
                    n_skip += 1
                    why = str(res.get("skip_reason", "missing/unsupported annotations, non-memory task, or already complete"))
                    logger.info(
                        "SKIP sub=%s task=%s run=%s (%s)",
                        run.subject,
                        run.task,
                        run.run,
                        why,
                    )
                else:
                    n_err += 1
                    logger.error("ERROR preprocessing sub=%s task=%s run=%s", run.subject, run.task, run.run)
                    logger.error("%s", res.get("traceback", "<no traceback>"))

    logger.info("Summary: total=%d done=%d skip=%d err=%d", n_total, n_done, n_skip, n_err)
    if n_err > 0:
        raise RuntimeError(f"Preprocessing failed for {n_err} run(s). See logs in {log_dir}")


if __name__ == "__main__":
    main()
