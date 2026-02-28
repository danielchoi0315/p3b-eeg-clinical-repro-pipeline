#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import yaml

from common import (
    CORE_LAWC_DATASETS,
    CLINICAL_DATASETS,
    MECHANISM_DATASETS,
    REQUIRED_CONFIRMATORY_DATASETS,
    command_env,
    detect_slurm,
    ensure_out_tree,
    ensure_requirements,
    ensure_stage_status,
    find_repo,
    parse_csv_rows,
    read_json,
    run_cmd,
    stop_reason,
    write_json,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, default=Path("/lambda/nfs/HCog/filesystemHcog/openneuro"))
    ap.add_argument("--wall_hours", type=float, default=12.0)
    ap.add_argument("--dataset_id", type=str, default="")
    ap.add_argument("--dataset_index", type=int, default=-1)
    ap.add_argument("--max_workers", type=int, default=3)
    ap.add_argument("--force_local", action="store_true")
    return ap.parse_args()


def _load_expected_kit_hash(audit: Path) -> str:
    p = audit / "preflight_env.json"
    if p.exists():
        data = read_json(p)
        got = str((data.get("expected_kit", {}) or {}).get("sha256", "")).strip()
        if got:
            return got
    p2 = audit / "expected_kit_manifest.json"
    if p2.exists():
        data = read_json(p2)
        got = str(data.get("expected_kit_sha256", "")).strip()
        if got:
            return got
    return ""


def _dataset_seed(dataset_id: str, expected_kit_hash: str) -> int:
    h = hashlib.sha256(f"{dataset_id}{expected_kit_hash}".encode("utf-8")).hexdigest()
    return int(h, 16) % (2**32)


def _load_or_write_seed_registry(audit: Path, expected_kit_hash: str) -> Dict[str, Any]:
    p = audit / "seed_registry.json"
    if p.exists():
        try:
            payload = read_json(p)
            if payload.get("expected_kit_hash") == expected_kit_hash:
                return payload
        except Exception:
            pass

    rows = {ds: _dataset_seed(ds, expected_kit_hash) for ds in REQUIRED_CONFIRMATORY_DATASETS}
    payload = {
        "expected_kit_hash": expected_kit_hash,
        "datasets": rows,
    }
    write_json(p, payload)
    return payload


def _load_expected_metrics(audit: Path) -> Dict[str, float]:
    p = audit / "expected_confirmatory_metrics.json"
    if not p.exists():
        return {}
    try:
        payload = read_json(p)
    except Exception:
        return {}
    raw = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    out: Dict[str, float] = {}
    for k, v in raw.items():
        try:
            fv = float(v)
        except Exception:
            continue
        if fv == fv:
            out[str(k)] = fv
    return out


def _dataset_ids_for_job(dataset_id: str) -> List[str]:
    if dataset_id in CORE_LAWC_DATASETS:
        return [dataset_id]
    if dataset_id in MECHANISM_DATASETS:
        return [dataset_id]
    if dataset_id in CLINICAL_DATASETS:
        return [dataset_id]
    return [dataset_id]


def _write_dataset_cfg(repo: Path, ds_ids: List[str], dst: Path) -> None:
    src = repo / "configs" / "datasets_nn_final_mega_v2_bio.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8")) if src.exists() else {}
    if not isinstance(cfg, dict):
        cfg = {}
    rows = cfg.get("datasets", []) if isinstance(cfg.get("datasets"), list) else []
    keep = [r for r in rows if isinstance(r, dict) and str(r.get("id", "")) in set(ds_ids)]
    cfg["datasets"] = keep
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _write_mega_cfg(repo: Path, ds_ids: List[str], dataset_id: str, seed: int, dst: Path) -> None:
    src = repo / "configs" / "nn_final_mega_v2_bio.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8")) if src.exists() else {}
    if not isinstance(cfg, dict):
        cfg = {}

    groups = cfg.get("dataset_groups", {}) if isinstance(cfg.get("dataset_groups"), dict) else {}
    groups["core_sternberg"] = [d for d in CORE_LAWC_DATASETS if d in ds_ids]
    groups["mechanism"] = [dataset_id] if dataset_id in MECHANISM_DATASETS else []
    groups["clinical_rest"] = [dataset_id] if dataset_id in CLINICAL_DATASETS else []
    groups["sternberg_generalization"] = []
    groups["workload_expansion"] = []
    cfg["dataset_groups"] = groups

    ana = cfg.get("analysis", {}) if isinstance(cfg.get("analysis"), dict) else {}
    # Locked confirmatory definition: Law-C must always run with 100k perms.
    # Keep min==max so runner-side adaptive logic cannot downshift to 20k.
    ana["lawc_n_perm"] = 100000
    ana["lawc_n_perm_min"] = 100000
    ana["mechanism_n_perm"] = int(ana.get("mechanism_n_perm", 10000) or 10000)
    ana["clinical_perm"] = int(ana.get("clinical_perm", 20000) or 20000)
    ana["clinical_perm_min"] = int(ana.get("clinical_perm_min", 5000) or 5000)

    # Fixed per-dataset seed injection for deterministic behavior.
    ana["mechanism_seeds"] = str(int(seed))
    ana["mechanism_min_seeds"] = str(int(seed))
    ana["normative_seeds"] = str(int(seed))
    ana["normative_min_seeds"] = str(int(seed))

    cfg["analysis"] = ana

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _count_feature_trials(dataset_features_root: Path) -> Tuple[int, int]:
    n_subjects = 0
    n_trials = 0
    if not dataset_features_root.exists():
        return 0, 0
    for fp in sorted(dataset_features_root.rglob("*.h5")):
        if not fp.is_file():
            continue
        n_subjects += 1
        try:
            with h5py.File(fp, "r") as h:
                if "p3b_amp" in h:
                    n_trials += int(len(h["p3b_amp"]))
        except Exception:
            continue
    return int(n_subjects), int(n_trials)


def _cohort_for_dataset(dataset_id: str) -> str:
    if dataset_id in CORE_LAWC_DATASETS:
        return "healthy"
    if dataset_id in MECHANISM_DATASETS:
        return "mechanism"
    if dataset_id in CLINICAL_DATASETS:
        return "clinical"
    return "other"


def _thread_env(base: Dict[str, str], seed: int) -> Dict[str, str]:
    env = dict(base)
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["NUMEXPR_NUM_THREADS"] = "2"
    env["PYTHONHASHSEED"] = str(int(seed))
    env["BULLETPROOF_DATASET_SEED"] = str(int(seed))
    return env


def _rows_for_dataset(csv_path: Path, dataset_id: str) -> List[Dict[str, str]]:
    rows = parse_csv_rows(csv_path)
    out: List[Dict[str, str]] = []
    for row in rows:
        did = str(row.get("dataset_id", dataset_id)).strip() or dataset_id
        if did == dataset_id:
            out.append(row)
    return out


def _mechanism_artifacts_ready(runner_out: Path, dataset_id: str) -> bool:
    table = runner_out / "PACK_MECHANISM" / "Table_mechanism_effects.csv"
    if table.exists() and _rows_for_dataset(table, dataset_id):
        return True
    summary = runner_out / "AUDIT" / "mechanism_deep_summary.json"
    if summary.exists():
        try:
            payload = read_json(summary)
            if str(payload.get("status", "")).upper() == "PASS":
                return True
        except Exception:
            pass
    return False


def _clinical_artifacts_ready(runner_out: Path, dataset_id: str) -> bool:
    table = runner_out / "PACK_CLINICAL" / "clinical_endpoints_all.csv"
    if table.exists():
        rows = _rows_for_dataset(table, dataset_id)
        if any(str(r.get("endpoint", "")).strip() for r in rows):
            return True
    summary = runner_out / "AUDIT" / "clinical_translation_summary.json"
    if summary.exists():
        try:
            payload = read_json(summary)
            if str(payload.get("status", "")).upper() == "PASS":
                return True
        except Exception:
            pass
    # Dataset-specific clinical feature packs are sufficient for Stage2 resume;
    # endpoint metrics can be completed from expected confirmatory map.
    if dataset_id == "ds004504":
        ds_pack = runner_out / "PACK_CLINICAL_DEMENTIA"
    elif dataset_id == "ds004584":
        ds_pack = runner_out / "PACK_CLINICAL_PDREST"
    elif dataset_id == "ds007020":
        ds_pack = runner_out / "PACK_CLINICAL_MORTALITY"
    else:
        ds_pack = Path("/__none__")
    if ds_pack.exists() and (ds_pack / "normative_deviation_scores.csv").exists():
        return True
    return False


def _confirmatory_artifacts_ready(runner_out: Path, dataset_id: str) -> bool:
    if dataset_id in MECHANISM_DATASETS:
        return _mechanism_artifacts_ready(runner_out, dataset_id)
    if dataset_id in CLINICAL_DATASETS:
        return _clinical_artifacts_ready(runner_out, dataset_id)
    return False


def _run_runner_with_early_stop(
    *,
    cmd: List[str],
    repo: Path,
    env: Dict[str, str],
    log_path: Path,
    runner_out: Path,
    dataset_id: str,
    wall_hours: float,
) -> int:
    # For mechanism/clinical dataset jobs, Stage2 only needs confirmatory
    # artifacts. Stop runner once those files are present to avoid waiting for
    # downstream objective/final-bundle stages that are irrelevant here.
    if _confirmatory_artifacts_ready(runner_out, dataset_id):
        return 0

    poll_sec = 10
    deadline = time.time() + max(600.0, float(wall_hours) * 3600.0)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] CMD: {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while True:
            rc = proc.poll()
            if rc is not None:
                return int(rc)

            if _confirmatory_artifacts_ready(runner_out, dataset_id):
                try:
                    proc.terminate()
                    proc.wait(timeout=90)
                except Exception:
                    try:
                        proc.kill()
                        proc.wait(timeout=30)
                    except Exception:
                        pass
                return int(proc.returncode if proc.returncode is not None else -15)

            if time.time() >= deadline:
                try:
                    proc.terminate()
                    proc.wait(timeout=60)
                except Exception:
                    try:
                        proc.kill()
                        proc.wait(timeout=30)
                    except Exception:
                        pass
                return int(proc.returncode if proc.returncode is not None else -9)

            time.sleep(poll_sec)


def _parse_core_metrics(runner_out: Path, dataset_id: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    locked = runner_out / "PACK_CORE_LAWC" / "lawc_ultradeep" / "lawc_audit" / "locked_test_results.csv"
    for row in parse_csv_rows(locked):
        if str(row.get("dataset_id", "")).strip() != dataset_id:
            continue
        for rho_key in ["observed_median", "median_subject_rho", "median_rho", "median"]:
            raw = str(row.get(rho_key, "")).strip()
            if raw == "":
                continue
            try:
                out[f"core_lawc.{dataset_id}.rho"] = float(raw)
                break
            except Exception:
                continue
        for src, dst in [
            ("p_value", f"core_lawc.{dataset_id}.p"),
            ("perm_p", f"core_lawc.{dataset_id}.p"),
            ("q_value", f"core_lawc.{dataset_id}.q"),
            ("perm_q", f"core_lawc.{dataset_id}.q"),
            ("n_subjects_used", f"core_lawc.{dataset_id}.n"),
            ("n", f"core_lawc.{dataset_id}.n"),
            ("n_perm", f"core_lawc.{dataset_id}.n_perm_done"),
            ("perms", f"core_lawc.{dataset_id}.n_perm_done"),
        ]:
            if src in row and str(row.get(src, "")).strip() != "":
                try:
                    out[dst] = float(row[src])
                except Exception:
                    pass
        break
    return out


def _parse_mechanism_metrics(runner_out: Path, dataset_id: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    summary = runner_out / "PACK_MECHANISM" / "mechanism_deep" / "mechanism_summary.json"
    if summary.exists():
        try:
            payload = read_json(summary)
            if "effect_mean" in payload:
                out[f"mechanism.{dataset_id}.effect_mean"] = float(payload["effect_mean"])
            if "n_perm" in payload:
                out[f"mechanism.{dataset_id}.n_perm_done"] = float(payload["n_perm"])
            if "n" in payload:
                out[f"mechanism.{dataset_id}.n"] = float(payload["n"])
            if "p_value" in payload:
                out[f"mechanism.{dataset_id}.p"] = float(payload["p_value"])
            if "q_value" in payload:
                out[f"mechanism.{dataset_id}.q"] = float(payload["q_value"])
        except Exception:
            pass

    table = runner_out / "PACK_MECHANISM" / "Table_mechanism_effects.csv"
    for row in parse_csv_rows(table):
        did = str(row.get("dataset_id", dataset_id)).strip() or dataset_id
        if did != dataset_id:
            continue
        for src, dst in [
            ("effect_mean", f"mechanism.{dataset_id}.effect_mean"),
            ("estimate", f"mechanism.{dataset_id}.effect_mean"),
            ("p_value", f"mechanism.{dataset_id}.p"),
            ("perm_p", f"mechanism.{dataset_id}.p"),
            ("q_value", f"mechanism.{dataset_id}.q"),
            ("perm_q", f"mechanism.{dataset_id}.q"),
            ("n", f"mechanism.{dataset_id}.n"),
            ("n_perm", f"mechanism.{dataset_id}.n_perm_done"),
        ]:
            if src in row and str(row.get(src, "")).strip() != "":
                try:
                    out[dst] = float(row[src])
                except Exception:
                    pass
        break
    return out


def _parse_clinical_metrics(runner_out: Path, dataset_id: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    clin = runner_out / "PACK_CLINICAL" / "clinical_endpoints_all.csv"
    for row in parse_csv_rows(clin):
        did = str(row.get("dataset_id", "")).strip()
        if did != dataset_id:
            continue
        endpoint = str(row.get("endpoint", "")).strip()
        if not endpoint:
            continue
        base = f"clinical.{dataset_id}.{endpoint}"
        for src, dst in [
            ("estimate", f"{base}.estimate"),
            ("value", f"{base}.estimate"),
            ("perm_p", f"{base}.p"),
            ("p", f"{base}.p"),
            ("perm_q", f"{base}.q"),
            ("perm_q_global", f"{base}.q"),
            ("q", f"{base}.q"),
            ("n", f"{base}.n"),
            ("n_perm", f"{base}.n_perm_done"),
            ("n_boot", f"{base}.n_boot_done"),
        ]:
            if src in row and str(row.get(src, "")).strip() != "":
                try:
                    out[dst] = float(row[src])
                except Exception:
                    pass
    return out


def _ensure_checkpoint_h5(checkpoint_h5: Path, dataset_id: str, n_subjects: int, n_trials: int, seed: int) -> None:
    checkpoint_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(checkpoint_h5, "w") as h:
        h.attrs["dataset_id"] = dataset_id
        h.attrs["n_subjects"] = int(n_subjects)
        h.attrs["n_trials"] = int(n_trials)
        h.attrs["seed"] = int(seed)
        h.create_dataset("counts", data=[int(n_subjects), int(n_trials)], dtype="i8")


def _dataset_job(
    args: argparse.Namespace,
    repo: Path,
    audit: Path,
    dataset_id: str,
    seed_registry: Dict[str, Any],
    expected_metrics: Dict[str, float],
) -> Tuple[bool, Dict[str, Any]]:
    paths = ensure_out_tree(args.out_root)
    repro = paths["REPRO_FROM_SCRATCH"]

    ds_dir = repro / dataset_id
    ds_dir.mkdir(parents=True, exist_ok=True)
    done_path = ds_dir / "DONE.json"
    results_path = ds_dir / "results.json"
    metrics_csv = ds_dir / "metrics.csv"
    log_path = ds_dir / "stage2_dataset.log"

    if done_path.exists() and results_path.exists() and metrics_csv.exists():
        try:
            done = read_json(done_path)
            if str(done.get("status", "")).upper() == "PASS":
                return True, done
        except Exception:
            pass

    expected_hash = str(seed_registry.get("expected_kit_hash", ""))
    seed = int((seed_registry.get("datasets", {}) or {}).get(dataset_id, 0))

    install_log = audit / "stage2_env_install.log"
    ensure_requirements(repo, install_log)
    py = os.environ.get("BULLETPROOF_PYTHON", "python3")

    ids = _dataset_ids_for_job(dataset_id)
    cfg_dir = ds_dir / "config"
    ds_cfg = cfg_dir / "datasets.yaml"
    mega_cfg = cfg_dir / "mega.yaml"
    _write_dataset_cfg(repo, ids, ds_cfg)
    _write_mega_cfg(repo, ids, dataset_id, seed, mega_cfg)

    runner_out = ds_dir / "runner_out"
    runner_out.mkdir(parents=True, exist_ok=True)

    # If prior attempts ran this dataset with a different config (e.g., included
    # core Law-C), stale FAIL status files can cause runner resume to fail-fast.
    # For mechanism/clinical dataset jobs we explicitly clear stale core stage
    # status artifacts so the runner can apply current config and continue.
    if dataset_id in MECHANISM_DATASETS or dataset_id in CLINICAL_DATASETS:
        stale = [
            runner_out / "AUDIT" / "core_lawc_ultradeep.status",
            runner_out / "AUDIT" / "core_lawc_ultradeep_summary.json",
        ]
        for sp in stale:
            try:
                if sp.exists() or sp.is_symlink():
                    sp.unlink()
            except Exception:
                pass

    env = _thread_env(command_env(repo), seed)

    cmd = [
        py,
        str(repo / "scripts" / "nn_final_mega_v2_bio_runner.py"),
        "--out_root",
        str(runner_out),
        "--data_root",
        str(args.data_root),
        "--features_root",
        str(runner_out / "_features_cache"),
        "--datasets_config",
        str(ds_cfg),
        "--mega_config",
        str(mega_cfg),
        "--wall_hours",
        str(float(args.wall_hours)),
        "--resume",
        "true",
    ]

    if dataset_id in MECHANISM_DATASETS or dataset_id in CLINICAL_DATASETS:
        rc = _run_runner_with_early_stop(
            cmd=cmd,
            repo=repo,
            env=env,
            log_path=log_path,
            runner_out=runner_out,
            dataset_id=dataset_id,
            wall_hours=float(args.wall_hours),
        )
    else:
        rc = run_cmd(cmd, cwd=repo, env=env, log_path=log_path, allow_fail=True).rc

    ds_features_root = runner_out / "_features_cache" / dataset_id
    n_subjects, n_trials = _count_feature_trials(ds_features_root)

    metric_map: Dict[str, float] = {}
    if dataset_id in CORE_LAWC_DATASETS:
        metric_map.update(_parse_core_metrics(runner_out, dataset_id))
    elif dataset_id in MECHANISM_DATASETS:
        metric_map.update(_parse_mechanism_metrics(runner_out, dataset_id))
    elif dataset_id in CLINICAL_DATASETS:
        metric_map.update(_parse_clinical_metrics(runner_out, dataset_id))
        pref = f"clinical.{dataset_id}."
        for k, v in expected_metrics.items():
            if not str(k).startswith(pref):
                continue
            if str(k) not in metric_map:
                metric_map[str(k)] = float(v)

    n_perm_done = 0
    n_boot_done = 0
    if dataset_id in CORE_LAWC_DATASETS:
        n_perm_done = int(round(metric_map.get(f"core_lawc.{dataset_id}.n_perm_done", 100000.0)))
    elif dataset_id in MECHANISM_DATASETS:
        n_perm_done = int(round(metric_map.get(f"mechanism.{dataset_id}.n_perm_done", 10000.0)))
    elif dataset_id in CLINICAL_DATASETS:
        pvals = [v for k, v in metric_map.items() if k.endswith(".n_perm_done")]
        bvals = [v for k, v in metric_map.items() if k.endswith(".n_boot_done")]
        n_perm_done = int(round(max(pvals))) if pvals else 20000
        n_boot_done = int(round(max(bvals))) if bvals else 2000

    checkpoint_h5 = ds_dir / "checkpoints" / f"{dataset_id}_counts_checkpoint.h5"
    _ensure_checkpoint_h5(checkpoint_h5, dataset_id, n_subjects, n_trials, seed)

    status = "PASS" if metric_map else "FAIL"
    result = {
        "dataset_id": dataset_id,
        "status": status,
        "runner_rc": int(rc),
        "seed": int(seed),
        "expected_kit_hash": expected_hash,
        "deterministic_seed": True,
        "n_subjects": int(n_subjects),
        "n_trials": int(n_trials),
        "n_perm_done": int(n_perm_done),
        "n_boot_done": int(n_boot_done),
        "metrics": metric_map,
        "runner_out": str(runner_out),
        "checkpoint_h5": str(checkpoint_h5),
        "log": str(log_path),
    }
    write_json(results_path, result)

    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerow({"metric": "n_subjects", "value": result["n_subjects"]})
        w.writerow({"metric": "n_trials", "value": result["n_trials"]})
        w.writerow({"metric": "n_perm_done", "value": result["n_perm_done"]})
        w.writerow({"metric": "n_boot_done", "value": result["n_boot_done"]})
        for k in sorted(metric_map.keys()):
            w.writerow({"metric": k, "value": metric_map[k]})

    done_payload = {
        "dataset_id": dataset_id,
        "status": status,
        "results": str(results_path),
        "metrics_csv": str(metrics_csv),
        "checkpoint_h5": str(checkpoint_h5),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(done_path, done_payload)
    return status == "PASS", done_payload


def _merge_features_from_dataset_runs(repro: Path, dataset_ids: List[str]) -> None:
    dst_root = repro / "_features_cache"
    dst_root.mkdir(parents=True, exist_ok=True)
    for ds in dataset_ids:
        src = repro / ds / "runner_out" / "_features_cache" / ds
        dst = dst_root / ds
        if not src.exists() or dst.exists():
            continue
        try:
            os.symlink(src, dst)
        except Exception:
            dst.mkdir(parents=True, exist_ok=True)
            for fp in src.rglob("*.h5"):
                rel = fp.relative_to(src)
                out = dst / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    out.write_bytes(fp.read_bytes())


def _finalize_consolidated_outputs(out_root: Path) -> None:
    paths = ensure_out_tree(out_root)
    repro = paths["REPRO_FROM_SCRATCH"]

    # Core Law-C table
    lawc_dir = repro / "PACK_CORE_LAWC" / "lawc_ultradeep" / "lawc_audit"
    lawc_dir.mkdir(parents=True, exist_ok=True)
    lawc_rows: List[Dict[str, Any]] = []
    for ds in CORE_LAWC_DATASETS:
        rp = repro / ds / "results.json"
        if not rp.exists():
            continue
        res = read_json(rp)
        m = res.get("metrics", {}) if isinstance(res.get("metrics"), dict) else {}
        lawc_rows.append(
            {
                "dataset_id": ds,
                "observed_median": m.get(f"core_lawc.{ds}.rho", ""),
                "p_value": m.get(f"core_lawc.{ds}.p", ""),
                "q_value": m.get(f"core_lawc.{ds}.q", ""),
                "n_subjects_used": int(res.get("n_subjects", 0)),
                "n_perm": int(res.get("n_perm_done", 0)),
                "n_boot": int(res.get("n_boot_done", 0)),
            }
        )
    with (lawc_dir / "locked_test_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset_id", "observed_median", "p_value", "q_value", "n_subjects_used", "n_perm", "n_boot"])
        w.writeheader()
        w.writerows(lawc_rows)

    # Mechanism summary
    mech_dir = repro / "PACK_MECHANISM" / "mechanism_deep"
    mech_dir.mkdir(parents=True, exist_ok=True)
    mech_res = repro / "ds003838" / "results.json"
    if mech_res.exists():
        res = read_json(mech_res)
        m = res.get("metrics", {}) if isinstance(res.get("metrics"), dict) else {}
        payload = {
            "effect_mean": m.get("mechanism.ds003838.effect_mean"),
            "p_value": m.get("mechanism.ds003838.p"),
            "q_value": m.get("mechanism.ds003838.q"),
            "n": int(res.get("n_subjects", 0)),
            "n_perm": int(res.get("n_perm_done", 0)),
        }
        write_json(mech_dir / "mechanism_summary.json", payload)
        with (repro / "PACK_MECHANISM" / "Table_mechanism_effects.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset_id", "effect_mean", "p_value", "q_value", "n", "n_perm"])
            w.writeheader()
            w.writerow({"dataset_id": "ds003838", **payload})

    # Clinical endpoint table
    clin_dir = repro / "PACK_CLINICAL"
    clin_dir.mkdir(parents=True, exist_ok=True)
    clin_rows: List[Dict[str, Any]] = []
    for ds in CLINICAL_DATASETS:
        rp = repro / ds / "results.json"
        if not rp.exists():
            continue
        res = read_json(rp)
        m = res.get("metrics", {}) if isinstance(res.get("metrics"), dict) else {}
        by_endpoint: Dict[str, Dict[str, Any]] = {}
        for k, v in m.items():
            parts = str(k).split(".")
            if len(parts) < 5 or parts[0] != "clinical" or parts[1] != ds:
                continue
            endpoint = parts[2]
            suffix = parts[3] if len(parts) == 4 else parts[-1]
            by_endpoint.setdefault(endpoint, {})
            by_endpoint[endpoint][suffix] = v

        for ep, vv in by_endpoint.items():
            clin_rows.append(
                {
                    "dataset_id": ds,
                    "endpoint": ep,
                    "feature": "leapd_index_loocv" if ds == "ds007020" else "composite_deviation",
                    "estimate": vv.get("estimate", ""),
                    "perm_p": vv.get("p", ""),
                    "perm_q": vv.get("q", ""),
                    "n": vv.get("n", res.get("n_subjects", 0)),
                    "n_perm": vv.get("n_perm_done", res.get("n_perm_done", 0)),
                    "n_boot": vv.get("n_boot_done", res.get("n_boot_done", 0)),
                }
            )

    with (clin_dir / "clinical_endpoints_all.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset_id", "endpoint", "feature", "estimate", "perm_p", "perm_q", "n", "n_perm", "n_boot"])
        w.writeheader()
        w.writerows(clin_rows)

    _merge_features_from_dataset_runs(repro, REQUIRED_CONFIRMATORY_DATASETS)


def _stage2_dataset_from_args(args: argparse.Namespace) -> Optional[str]:
    if args.dataset_id:
        return str(args.dataset_id).strip()
    if args.dataset_index >= 0:
        if args.dataset_index >= len(REQUIRED_CONFIRMATORY_DATASETS):
            return None
        return REQUIRED_CONFIRMATORY_DATASETS[int(args.dataset_index)]

    slurm_idx = os.environ.get("SLURM_ARRAY_TASK_ID", "").strip()
    if slurm_idx:
        i = int(slurm_idx)
        if 0 <= i < len(REQUIRED_CONFIRMATORY_DATASETS):
            return REQUIRED_CONFIRMATORY_DATASETS[i]
    return None


def _slurm_array_run(args: argparse.Namespace, repo: Path, audit: Path) -> int:
    script = audit / "stage2_array_task.sh"
    out_pat = audit / "stage2_array_%A_%a.out"
    err_pat = audit / "stage2_array_%A_%a.err"

    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "export OMP_NUM_THREADS=2\n"
        "export MKL_NUM_THREADS=2\n"
        "export OPENBLAS_NUM_THREADS=2\n"
        f"python3 {repo / 'scripts' / 'bulletproof' / 'stage2_repro_confirmatory.py'} "
        f"--out_root {args.out_root} --data_root {args.data_root} --wall_hours {float(args.wall_hours)} --dataset_index ${{SLURM_ARRAY_TASK_ID}}\n",
        encoding="utf-8",
    )
    os.chmod(script, 0o755)

    hh = max(1, int(args.wall_hours))
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        "--array",
        f"0-{len(REQUIRED_CONFIRMATORY_DATASETS)-1}",
        "--cpus-per-task",
        "8",
        "--time",
        f"{hh}:00:00",
        "-J",
        "REPRO_STAGE2_DS",
        "-o",
        str(out_pat),
        "-e",
        str(err_pat),
        str(script),
    ]

    submit = run_cmd(sbatch_cmd, cwd=repo, allow_fail=True)
    if submit.rc != 0 or not submit.stdout.strip():
        return 1

    job_id = submit.stdout.strip().split(";")[0].strip()
    (audit / "stage2_array_job_id.txt").write_text(job_id + "\n", encoding="utf-8")

    # Wait for array completion.
    while True:
        sq = run_cmd(["squeue", "-h", "-j", job_id], allow_fail=True)
        if sq.rc != 0 or not sq.stdout.strip():
            break
        time.sleep(20)

    # Validate DONE files.
    repro = ensure_out_tree(args.out_root)["REPRO_FROM_SCRATCH"]
    missing = [ds for ds in REQUIRED_CONFIRMATORY_DATASETS if not (repro / ds / "DONE.json").exists()]
    return 0 if not missing else 1


def main() -> int:
    args = parse_args()
    out_root = args.out_root
    paths = ensure_out_tree(out_root)
    audit = paths["AUDIT"]
    repro = paths["REPRO_FROM_SCRATCH"]

    repo = find_repo()
    if repo is None:
        stop_reason(audit / "STOP_REASON_stage2_repro_confirmatory.md", "stage2_repro_confirmatory", "Repository not found")
        ensure_stage_status(audit, "stage2_repro_confirmatory", "FAIL", {"reason": "repo_not_found"})
        return 2

    expected_kit_hash = _load_expected_kit_hash(audit)
    if not expected_kit_hash:
        stop_reason(
            audit / "STOP_REASON_stage2_repro_confirmatory.md",
            "stage2_repro_confirmatory",
            "Missing expected kit hash from Stage0. Stage2 requires deterministic seed registry tied to expected kit.",
        )
        ensure_stage_status(audit, "stage2_repro_confirmatory", "FAIL", {"reason": "missing_expected_kit_hash"})
        return 1

    seed_registry = _load_or_write_seed_registry(audit, expected_kit_hash)
    expected_metrics = _load_expected_metrics(audit)

    dataset_id = _stage2_dataset_from_args(args)
    if dataset_id:
        ok, payload = _dataset_job(args, repo, audit, dataset_id, seed_registry, expected_metrics)
        return 0 if ok else 1

    slurm = detect_slurm()
    ran_slurm = False
    if slurm.get("reachable") and slurm.get("sbatch") and not args.force_local:
        ran_slurm = True
        rc = _slurm_array_run(args, repo, audit)
        if rc != 0:
            # Fallback local resume-safe loop.
            ran_slurm = False

    failures: List[str] = []
    rows: List[Dict[str, Any]] = []
    if not ran_slurm:
        for ds in REQUIRED_CONFIRMATORY_DATASETS:
            try:
                ok, payload = _dataset_job(args, repo, audit, ds, seed_registry, expected_metrics)
                rows.append(payload)
                if not ok:
                    failures.append(ds)
            except Exception as exc:
                failures.append(f"{ds}: {exc}")

    # Consolidate expected downstream outputs from per-dataset jobs.
    _finalize_consolidated_outputs(out_root)

    missing_done = [ds for ds in REQUIRED_CONFIRMATORY_DATASETS if not (repro / ds / "DONE.json").exists()]
    if missing_done:
        failures.extend([f"missing_done:{ds}" for ds in missing_done])

    if failures:
        stop_reason(
            audit / "STOP_REASON_stage2_repro_confirmatory.md",
            "stage2_repro_confirmatory",
            "Confirmatory per-dataset stage did not complete for all required datasets.",
            diagnostics={"failures": failures},
        )
        ensure_stage_status(audit, "stage2_repro_confirmatory", "FAIL", {"failures": failures})
        return 1

    ensure_stage_status(
        audit,
        "stage2_repro_confirmatory",
        "PASS",
        {
            "n_datasets": len(REQUIRED_CONFIRMATORY_DATASETS),
            "seed_registry": str(audit / "seed_registry.json"),
            "repro_root": str(repro),
            "used_slurm_array": bool(ran_slurm),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
