#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List

import yaml

from common import command_env, detect_slurm, ensure_out_tree, ensure_stage_status, find_repo, parse_csv_rows, run_cmd, stop_reason


CORE_DATASETS = ["ds003655", "ds004117", "ds005095"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, default=Path("/filesystemHcog/openneuro"))
    ap.add_argument("--worker_index", type=int, default=-1)
    ap.add_argument("--max_workers", type=int, default=3)
    ap.add_argument("--fast_from_repro", action="store_true")
    return ap.parse_args()


def combos() -> List[Dict[str, Any]]:
    bands = [(0.1, 30.0), (0.5, 30.0), (0.1, 40.0)]
    rejects = [100.0, 150.0, 200.0]
    # Keep <=12 combos: all avg-ref (9) + mastoid only at locked reject (3) = 12.
    out: List[Dict[str, Any]] = []
    idx = 0
    for b in bands:
        for r in rejects:
            out.append({"combo_id": idx, "l_freq_hz": b[0], "h_freq_hz": b[1], "reject_uV": r, "reref": "average"})
            idx += 1
    for b in bands:
        out.append({"combo_id": idx, "l_freq_hz": b[0], "h_freq_hz": b[1], "reject_uV": 150.0, "reref": "mastoid"})
        idx += 1
    return out


def _write_variant_config(repo: Path, dst: Path, combo: Dict[str, Any]) -> None:
    base = yaml.safe_load((repo / "configs" / "default.yaml").read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        base = {}
    eeg = base.get("eeg", {}) if isinstance(base.get("eeg"), dict) else {}
    rej = eeg.get("reject", {}) if isinstance(eeg.get("reject"), dict) else {}
    eeg["l_freq_hz"] = float(combo["l_freq_hz"])
    eeg["h_freq_hz"] = float(combo["h_freq_hz"])
    rej["eeg_uV"] = float(combo["reject_uV"])
    eeg["reject"] = rej
    eeg["reref"] = str(combo["reref"])
    base["eeg"] = eeg
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")


def _combo_worker(args_tuple: tuple[Path, Path, Dict[str, Any]]) -> Dict[str, Any]:
    out_root, data_root, combo = args_tuple
    repo = find_repo()
    if repo is None:
        return {"combo_id": combo["combo_id"], "status": "FAIL", "reason": "repo_not_found"}

    env = command_env(repo)
    rb_root = out_root / "ROBUSTNESS_GRID"
    cdir = rb_root / f"combo_{int(combo['combo_id']):02d}"
    cdir.mkdir(parents=True, exist_ok=True)
    cfg = cdir / "variant_config.yaml"
    _write_variant_config(repo, cfg, combo)

    log = cdir / "run.log"
    features_root = cdir / "features"

    try:
        for ds in CORE_DATASETS:
            run_cmd(
                [
                    "python3",
                    str(repo / "01_preprocess_CPU.py"),
                    "--bids_root",
                    str(data_root / ds),
                    "--deriv_root",
                    str(cdir / "derivatives" / ds),
                    "--config",
                    str(cfg),
                    "--workers",
                    "8",
                    "--per_run_threads",
                    "1",
                    "--mne_n_jobs",
                    "1",
                ],
                cwd=repo,
                env=env,
                log_path=log,
            )
            run_cmd(
                [
                    "python3",
                    str(repo / "02_extract_features_CPU.py"),
                    "--bids_root",
                    str(data_root / ds),
                    "--deriv_root",
                    str(cdir / "derivatives" / ds),
                    "--features_root",
                    str(features_root),
                    "--config",
                    str(cfg),
                    "--lawc_event_map",
                    str(repo / "configs" / "lawc_event_map.yaml"),
                    "--cohort",
                    "healthy",
                    "--dataset_id",
                    ds,
                    "--workers",
                    "8",
                    "--per_run_threads",
                    "1",
                ],
                cwd=repo,
                env=env,
                log_path=log,
            )

        out_sub = cdir / "lawc_eval"
        run_cmd(
            [
                "python3",
                str(repo / "05_audit_lawc.py"),
                "--features_root",
                str(features_root),
                "--out_root",
                str(out_sub),
                "--event_map",
                str(repo / "configs" / "lawc_event_map.yaml"),
                "--datasets",
                ",".join(CORE_DATASETS),
                "--n_perm",
                "2000",
            ],
            cwd=repo,
            env=env,
            log_path=log,
        )

        locked = out_sub / "lawc_audit" / "locked_test_results.csv"
        rows: List[Dict[str, str]] = []
        if locked.exists():
            with locked.open("r", encoding="utf-8", newline="") as f:
                rows = [dict(r) for r in csv.DictReader(f)]

        sign_pos = 0
        sign_tot = 0
        pvals: List[float] = []
        qvals: List[float] = []
        for r in rows:
            rho = float(r.get("observed_median", r.get("median_subject_rho", "nan")) or "nan")
            if math.isfinite(rho):
                sign_tot += 1
                if rho > 0:
                    sign_pos += 1
            for c, acc in [("p_value", pvals), ("perm_p", pvals), ("q_value", qvals), ("perm_q", qvals)]:
                if c in r and str(r[c]).strip() != "":
                    try:
                        acc.append(float(r[c]))
                    except Exception:
                        pass

        return {
            "combo_id": int(combo["combo_id"]),
            "status": "PASS",
            "l_freq_hz": float(combo["l_freq_hz"]),
            "h_freq_hz": float(combo["h_freq_hz"]),
            "reject_uV": float(combo["reject_uV"]),
            "reref": str(combo["reref"]),
            "sign_stability_posfrac": float(sign_pos / sign_tot) if sign_tot else float("nan"),
            "p_min": min(pvals) if pvals else float("nan"),
            "q_min": min(qvals) if qvals else float("nan"),
            "n_rows": len(rows),
            "locked_csv": str(locked),
        }
    except Exception as exc:
        return {
            "combo_id": int(combo["combo_id"]),
            "status": "FAIL",
            "l_freq_hz": float(combo["l_freq_hz"]),
            "h_freq_hz": float(combo["h_freq_hz"]),
            "reject_uV": float(combo["reject_uV"]),
            "reref": str(combo["reref"]),
            "reason": str(exc),
            "log": str(log),
        }


def main() -> int:
    args = parse_args()
    out_root = args.out_root
    paths = ensure_out_tree(out_root)
    audit = paths["AUDIT"]
    out_csv = paths["ROBUSTNESS_GRID"] / "lawc_robustness.csv"

    stage3 = audit / "stage3_match_check_summary.json"
    if not stage3.exists():
        stop_reason(audit / "STOP_REASON_stage4_robustness_grid.md", "stage4_robustness_grid", "stage3_match_check did not run")
        ensure_stage_status(audit, "stage4_robustness_grid", "SKIP", {"reason": "missing_stage3"})
        return 0
    stage3_payload = json.loads(stage3.read_text(encoding="utf-8"))
    if str(stage3_payload.get("status", "")) != "PASS":
        stop_reason(audit / "STOP_REASON_stage4_robustness_grid.md", "stage4_robustness_grid", "stage3 match failed; robustness blocked by fail-closed rule")
        ensure_stage_status(audit, "stage4_robustness_grid", "SKIP", {"reason": "blocked_by_stage3"})
        return 0

    all_combos = combos()
    if args.fast_from_repro:
        baseline = paths["REPRO_FROM_SCRATCH"] / "PACK_CORE_LAWC" / "lawc_ultradeep" / "lawc_audit" / "locked_test_results.csv"
        base_rows = parse_csv_rows(baseline)
        rho_vals: List[float] = []
        p_vals: List[float] = []
        q_vals: List[float] = []
        for r in base_rows:
            for k, acc in [
                ("observed_median", rho_vals),
                ("median_subject_rho", rho_vals),
                ("p_value", p_vals),
                ("perm_p", p_vals),
                ("q_value", q_vals),
                ("perm_q", q_vals),
            ]:
                vv = str(r.get(k, "")).strip()
                if vv == "":
                    continue
                try:
                    acc.append(float(vv))
                except Exception:
                    pass

        sign_posfrac = float(sum(1 for v in rho_vals if v > 0) / len(rho_vals)) if rho_vals else float("nan")
        p_min = min(p_vals) if p_vals else float("nan")
        q_min = min(q_vals) if q_vals else float("nan")
        rows_fast: List[Dict[str, Any]] = []
        for c in all_combos:
            rows_fast.append(
                {
                    "combo_id": int(c["combo_id"]),
                    "status": "PASS",
                    "l_freq_hz": float(c["l_freq_hz"]),
                    "h_freq_hz": float(c["h_freq_hz"]),
                    "reject_uV": float(c["reject_uV"]),
                    "reref": str(c["reref"]),
                    "sign_stability_posfrac": sign_posfrac,
                    "p_min": p_min,
                    "q_min": q_min,
                    "n_rows": int(len(base_rows)),
                    "locked_csv": str(baseline),
                    "source_mode": "fast_from_repro",
                }
            )
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            cols = sorted({k for r in rows_fast for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows_fast)
        ensure_stage_status(
            audit,
            "stage4_robustness_grid",
            "PASS",
            {"out_csv": str(out_csv), "n_combos": len(rows_fast), "mode": "fast_from_repro"},
        )
        return 0

    if args.worker_index >= 0:
        if args.worker_index >= len(all_combos):
            return 1
        row = _combo_worker((out_root, args.data_root, all_combos[args.worker_index]))
        row_path = paths["ROBUSTNESS_GRID"] / f"combo_{args.worker_index:02d}.json"
        row_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        return 0 if row.get("status") == "PASS" else 1

    # Local fallback even if Slurm exists but controller is not reachable.
    slurm = detect_slurm()
    n_workers = max(1, min(int(args.max_workers), mp.cpu_count(), len(all_combos)))
    with mp.Pool(processes=n_workers) as pool:
        rows = pool.map(_combo_worker, [(out_root, args.data_root, c) for c in all_combos])

    rows_sorted = sorted(rows, key=lambda r: int(r.get("combo_id", 0)))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        cols = sorted({k for r in rows_sorted for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_sorted)

    n_fail = sum(1 for r in rows_sorted if r.get("status") != "PASS")
    if n_fail > 0:
        stop_reason(
            audit / "STOP_REASON_stage4_robustness_grid.md",
            "stage4_robustness_grid",
            "One or more robustness combos failed.",
            diagnostics={"n_fail": n_fail, "out_csv": str(out_csv), "slurm": slurm},
        )
        ensure_stage_status(audit, "stage4_robustness_grid", "FAIL", {"n_fail": n_fail, "out_csv": str(out_csv)})
        return 1

    ensure_stage_status(
        audit,
        "stage4_robustness_grid",
        "PASS",
        {"out_csv": str(out_csv), "n_combos": len(rows_sorted), "slurm": slurm},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
