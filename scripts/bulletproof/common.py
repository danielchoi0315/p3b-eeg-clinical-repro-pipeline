#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

CORE_LAWC_DATASETS = ["ds003655", "ds004117", "ds005095"]
MECHANISM_DATASETS = ["ds003838"]
CLINICAL_DATASETS = ["ds004504", "ds004584", "ds007020"]
REQUIRED_CONFIRMATORY_DATASETS = CORE_LAWC_DATASETS + MECHANISM_DATASETS + CLINICAL_DATASETS
OPTIONAL_DATASETS = ["ds004796", "ds007262", "ds004752"]

EXPECTED_KIT_SEARCH_ROOTS = [
    Path("/lambda/nfs/HCog/filesystemHcog/runs"),
    Path("/filesystemHcog/runs"),
]
EXPECTED_KIT_PATTERNS = [
    "**/OUTZIP/MANUSCRIPT_KIT*.zip",
    "**/OUTZIP/*MANUSCRIPT*.zip",
    "**/MANUSCRIPT_KIT*.zip",
]
DEFAULT_MASTER_RUN_LABEL = "REPRO_BULLETPROOF_MASTER_RUN"


@dataclass
class ExecResult:
    rc: int
    stdout: str
    stderr: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ts_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout_sec: Optional[int] = None,
    log_path: Optional[Path] = None,
    allow_fail: bool = False,
) -> ExecResult:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(f"[{iso_now()}] CMD: {' '.join(cmd)}\n")
            lf.flush()
            proc = subprocess.run(
                list(cmd),
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        rc = int(proc.returncode)
        if rc != 0 and not allow_fail:
            raise RuntimeError(f"command failed rc={rc}: {' '.join(cmd)}")
        return ExecResult(rc=rc, stdout="", stderr="")

    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    rc = int(proc.returncode)
    if rc != 0 and not allow_fail:
        raise RuntimeError(f"command failed rc={rc}: {' '.join(cmd)}\n{proc.stderr}")
    return ExecResult(rc=rc, stdout=proc.stdout, stderr=proc.stderr)


def run_shell(cmd: str, *, cwd: Optional[Path] = None, log_path: Optional[Path] = None, allow_fail: bool = False) -> int:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(f"[{iso_now()}] SH: {cmd}\n")
            lf.flush()
            proc = subprocess.run(
                ["bash", "-lc", cmd],
                cwd=str(cwd) if cwd else None,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        rc = int(proc.returncode)
    else:
        proc = subprocess.run(["bash", "-lc", cmd], cwd=str(cwd) if cwd else None, check=False)
        rc = int(proc.returncode)
    if rc != 0 and not allow_fail:
        raise RuntimeError(f"shell command failed rc={rc}: {cmd}")
    return rc


def find_repo() -> Optional[Path]:
    env_repo = os.environ.get("REPO_ROOT", "").strip()
    if env_repo:
        p = Path(env_repo).expanduser().resolve()
        if (p / "scripts" / "bulletproof" / "master.py").exists():
            return p

    roots: List[Path] = [Path(os.environ.get("HOME", "/home/ubuntu"))]
    if Path("/lambda/nfs/HCog").exists():
        roots.append(Path("/lambda/nfs/HCog"))

    cwd = Path.cwd().resolve()
    if (cwd / "scripts" / "bulletproof" / "master.py").exists():
        return cwd
    if (cwd.parent / "scripts" / "bulletproof" / "master.py").exists():
        return cwd.parent

    cands: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for marker in root.rglob("scripts/bulletproof/master.py"):
            p = marker.parents[2]
            if p.is_dir():
                cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda p: len(str(p)))
    for p in cands:
        if (p / "README.md").exists() and (p / "scripts").exists():
            return p
    return cands[0]


def find_kit_paths() -> List[Path]:
    roots: List[Path] = [Path(os.environ.get("HOME", "/home/ubuntu")), Path("/scratch"), Path("/filesystemHcog")]
    if not Path("/filesystemHcog").exists() and Path("/lambda/nfs/HCog/filesystemHcog").exists():
        roots.append(Path("/lambda/nfs/HCog/filesystemHcog"))

    hits: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("MANUSCRIPT_KIT.zip"):
            if p.is_file():
                hits.append(p)
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits


def find_expected_kit_candidates() -> List[Path]:
    cands: List[Path] = []
    seen: set[str] = set()
    for root in EXPECTED_KIT_SEARCH_ROOTS:
        if not root.exists():
            continue
        for pat in EXPECTED_KIT_PATTERNS:
            for p in root.glob(pat):
                if not p.is_file():
                    continue
                s = str(p.resolve())
                if s in seen:
                    continue
                seen.add(s)
                cands.append(p.resolve())
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands


def resolve_expected_kit(explicit_path: Optional[Path]) -> Tuple[Optional[Path], List[Path]]:
    if explicit_path is not None:
        p = explicit_path.expanduser().resolve()
        return (p if p.exists() and p.is_file() else None, [p])
    cands = find_expected_kit_candidates()
    if not cands:
        return None, []
    return cands[0], cands


def unpack_zip(zip_path: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)


def find_expected_dataset_hashes_file(kit_ref: Path) -> Optional[Path]:
    preferred = [
        kit_ref / "PROVENANCE" / "dataset_hashes.json",
        kit_ref / "dataset_hashes.json",
    ]
    for p in preferred:
        if p.exists():
            return p
    cands = sorted(kit_ref.rglob("dataset_hashes.json"))
    return cands[0] if cands else None


def find_expected_results_table_file(kit_ref: Path) -> Optional[Path]:
    preferred = [
        kit_ref / "FINAL_RESULTS_TABLE.csv",
        kit_ref / "RESULTS" / "FINAL_RESULTS_TABLE.csv",
    ]
    for p in preferred:
        if p.exists():
            return p
    cands = sorted([p for p in kit_ref.rglob("*.csv") if "result" in p.name.lower() and "table" in p.name.lower()])
    return cands[0] if cands else None


def parse_dataset_hashes_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    rows = payload.get("datasets", []) if isinstance(payload.get("datasets"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        did = str(row.get("dataset_id", "")).strip()
        commit = str(
            row.get("expected_commit")
            or row.get("checked_out_commit")
            or row.get("pinned_hash")
            or row.get("pinned_hash_used")
            or ""
        ).strip()
        if did and commit:
            out[did] = commit

    if not out:
        for k, v in payload.items():
            if k in REQUIRED_CONFIRMATORY_DATASETS and isinstance(v, str) and v.strip():
                out[k] = v.strip()
    return out


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _is_number(v: Any) -> bool:
    try:
        f = float(v)
    except Exception:
        return False
    return f == f and f not in (float("inf"), float("-inf"))


def canonicalize_metric_key(raw_key: str) -> str:
    k = str(raw_key).strip().lower()
    if not k:
        return ""

    if k.startswith("lawc."):
        k = "core_lawc." + k[len("lawc.") :]

    k = k.replace("observed_median", "rho")
    k = k.replace("median_subject_rho", "rho")
    k = k.replace("median_rho", "rho")

    k = k.replace(".perm_p", ".p")
    k = k.replace(".perm_q_global", ".q")
    k = k.replace(".perm_q", ".q")

    # Use token-boundary replacements to avoid duplicating suffixes like
    # ".n_perm_done" -> ".n_perm_done_done".
    k = re.sub(r"\.perms\b", ".n_perm_done", k)
    k = re.sub(r"\.n_perm\b", ".n_perm_done", k)
    k = re.sub(r"\.perm_n\b", ".n_perm_done", k)

    k = re.sub(r"\.n_boot\b", ".n_boot_done", k)
    k = re.sub(r"\.boot_n\b", ".n_boot_done", k)

    k = re.sub(r"\.+", ".", k).strip(".")
    return k


def _extract_dataset_id_from_key(key: str) -> str:
    parts = key.split(".")
    for p in parts:
        if re.fullmatch(r"ds\d{6}", p):
            return p
    return ""


def is_confirmatory_metric_key(key: str) -> bool:
    k = canonicalize_metric_key(key)
    if not k:
        return False
    did = _extract_dataset_id_from_key(k)
    if did and did in OPTIONAL_DATASETS:
        return False
    if k.startswith("core_lawc.") and did in CORE_LAWC_DATASETS:
        return True
    if k.startswith("mechanism."):
        if did and did not in MECHANISM_DATASETS:
            return False
        return True
    if k.startswith("clinical.") and did in CLINICAL_DATASETS:
        return True
    return False


def parse_expected_confirmatory_metrics(table_path: Path) -> Dict[str, float]:
    rows = parse_csv_rows(table_path)
    out: Dict[str, float] = {}

    for row in rows:
        section = str(row.get("section", "")).strip().lower()
        if section:
            ds = str(row.get("dataset_id", "")).strip().lower()
            endpoint = str(row.get("endpoint", "")).strip()
            est = row.get("estimate", row.get("value", ""))
            perm_p = row.get("perm_p", row.get("p", ""))
            perm_q = row.get("perm_q", row.get("perm_q_within", row.get("perm_q_global", row.get("q", ""))))
            n = row.get("n", "")
            n_perm = row.get("n_perm_done", row.get("n_perm", ""))
            n_boot = row.get("n_boot_done", row.get("n_boot", ""))

            if section == "core_lawc" and ds in CORE_LAWC_DATASETS:
                cand = {
                    f"core_lawc.{ds}.rho": est,
                    f"core_lawc.{ds}.p": perm_p,
                    f"core_lawc.{ds}.q": perm_q,
                    f"core_lawc.{ds}.n": n,
                    f"core_lawc.{ds}.n_perm_done": n_perm,
                    f"core_lawc.{ds}.n_boot_done": n_boot,
                }
                for k, v in cand.items():
                    if _is_number(v):
                        out[k] = float(v)
                continue

            if section == "mechanism" and ds in MECHANISM_DATASETS:
                cand = {
                    f"mechanism.{ds}.effect_mean": est,
                    f"mechanism.{ds}.p": perm_p,
                    f"mechanism.{ds}.q": perm_q,
                    f"mechanism.{ds}.n": n,
                    f"mechanism.{ds}.n_perm_done": n_perm,
                    f"mechanism.{ds}.n_boot_done": n_boot,
                }
                for k, v in cand.items():
                    if _is_number(v):
                        out[k] = float(v)
                continue

            if section == "clinical" and ds in CLINICAL_DATASETS and endpoint:
                cand = {
                    f"clinical.{ds}.{endpoint}.estimate": est,
                    f"clinical.{ds}.{endpoint}.p": perm_p,
                    f"clinical.{ds}.{endpoint}.q": perm_q,
                    f"clinical.{ds}.{endpoint}.n": n,
                    f"clinical.{ds}.{endpoint}.n_perm_done": n_perm,
                    f"clinical.{ds}.{endpoint}.n_boot_done": n_boot,
                }
                for k, v in cand.items():
                    if _is_number(v):
                        out[k] = float(v)
                continue

        if "metric" in row:
            raw_key = str(row.get("metric", "")).strip()
            key = canonicalize_metric_key(raw_key)
            if not is_confirmatory_metric_key(key):
                continue
            val = row.get("value", row.get("estimate", row.get("observed", "")))
            if _is_number(val):
                out[key] = float(val)
            continue

        ds = str(row.get("dataset_id", "")).strip().lower()
        endpoint = str(row.get("endpoint", "")).strip()

        if ds in CORE_LAWC_DATASETS:
            cand = {
                f"core_lawc.{ds}.rho": row.get("observed_median", row.get("median_subject_rho", row.get("rho", ""))),
                f"core_lawc.{ds}.p": row.get("p_value", row.get("perm_p", row.get("p", ""))),
                f"core_lawc.{ds}.q": row.get("q_value", row.get("perm_q", row.get("q", ""))),
                f"core_lawc.{ds}.n": row.get("n_subjects_used", row.get("n_used", row.get("n", ""))),
                f"core_lawc.{ds}.n_perm_done": row.get("n_perm", row.get("perms", "")),
            }
            for k, v in cand.items():
                if _is_number(v):
                    out[k] = float(v)
            continue

        if ds in CLINICAL_DATASETS and endpoint:
            cand = {
                f"clinical.{ds}.{endpoint}.estimate": row.get("estimate", row.get("value", "")),
                f"clinical.{ds}.{endpoint}.p": row.get("perm_p", row.get("p", "")),
                f"clinical.{ds}.{endpoint}.q": row.get("perm_q", row.get("perm_q_global", row.get("q", ""))),
                f"clinical.{ds}.{endpoint}.n": row.get("n", ""),
                f"clinical.{ds}.{endpoint}.n_perm_done": row.get("n_perm", ""),
                f"clinical.{ds}.{endpoint}.n_boot_done": row.get("n_boot", ""),
            }
            for k, v in cand.items():
                if _is_number(v):
                    out[k] = float(v)
            continue

        if ds in MECHANISM_DATASETS:
            cand = {
                f"mechanism.{ds}.effect_mean": row.get("effect_mean", row.get("estimate", row.get("value", ""))),
                f"mechanism.{ds}.p": row.get("perm_p", row.get("p", "")),
                f"mechanism.{ds}.q": row.get("perm_q", row.get("q", "")),
                f"mechanism.{ds}.n": row.get("n", ""),
                f"mechanism.{ds}.n_perm_done": row.get("n_perm", row.get("perms", "")),
            }
            for k, v in cand.items():
                if _is_number(v):
                    out[k] = float(v)

    return out


def best_scratch_base() -> Path:
    slurm_tmp = os.environ.get("SLURM_TMPDIR", "").strip()
    if slurm_tmp and Path(slurm_tmp).exists():
        return Path(slurm_tmp)
    if Path("/filesystemHcog").exists():
        return Path("/filesystemHcog")
    if Path("/lambda/nfs/HCog/filesystemHcog").exists():
        return Path("/lambda/nfs/HCog/filesystemHcog")
    return Path(os.environ.get("HOME", "/home/ubuntu"))


def out_root_default() -> Path:
    base = best_scratch_base() / "runs"
    return base / f"{ts_slug()}_{DEFAULT_MASTER_RUN_LABEL}"


def ensure_out_tree(out_root: Path) -> Dict[str, Path]:
    paths = {
        "AUDIT": out_root / "AUDIT",
        "REPRO_FROM_SCRATCH": out_root / "REPRO_FROM_SCRATCH",
        "ROBUSTNESS_GRID": out_root / "ROBUSTNESS_GRID",
        "RELIABILITY": out_root / "RELIABILITY",
        "CLINICAL_STABILITY": out_root / "CLINICAL_STABILITY",
        "MANUSCRIPT_KIT_UPDATED": out_root / "MANUSCRIPT_KIT_UPDATED",
        "OVERLEAF_UPDATED": out_root / "OVERLEAF_UPDATED",
        "PRISM_ARTIST_PACK": out_root / "PRISM_ARTIST_PACK",
        "OUTZIP": out_root / "OUTZIP",
        "TARBALLS": out_root / "TARBALLS",
        "PACK_BIO": out_root / "PACK_BIO",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def detect_slurm() -> Dict[str, Any]:
    sbatch = shutil.which("sbatch")
    sinfo = shutil.which("sinfo")
    reachable = False
    err = ""
    if sbatch and sinfo:
        r = subprocess.run(["sinfo", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        reachable = r.returncode == 0
        err = (r.stderr or "").strip()
    return {
        "sbatch": sbatch or "",
        "sinfo": sinfo or "",
        "reachable": bool(reachable),
        "error": err,
    }


def stop_reason(path: Path, title: str, why: str, diagnostics: Optional[Dict[str, Any]] = None) -> None:
    lines = [f"# STOP_REASON {title}", "", "## Why", why]
    if diagnostics:
        lines.extend(["", "## Diagnostics", "```json", json.dumps(diagnostics, indent=2), "```"])
    write_text(path, "\n".join(lines) + "\n")


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def stage_dataset(
    *,
    dataset_id: str,
    git_url: str,
    data_root: Path,
    pinned_hash: Optional[str],
    datalad_bin: Optional[str],
    log_path: Path,
) -> Dict[str, Any]:
    ds_root = data_root / dataset_id
    ds_root.parent.mkdir(parents=True, exist_ok=True)

    if not ds_root.exists():
        if datalad_bin:
            run_cmd([datalad_bin, "clone", git_url, str(ds_root)], cwd=data_root.parent, log_path=log_path)
        else:
            run_cmd(["git", "clone", git_url, str(ds_root)], cwd=data_root.parent, log_path=log_path)

    run_cmd(["git", "fetch", "origin", "--tags", "--prune"], cwd=ds_root, log_path=log_path, allow_fail=True)

    used_hash = ""
    pin = (pinned_hash or "").strip()
    if pin and pin.lower() not in {"null", "none"}:
        run_cmd(["git", "checkout", pin], cwd=ds_root, log_path=log_path)
        used_hash = pin
    else:
        run_cmd(["git", "checkout", "HEAD"], cwd=ds_root, log_path=log_path)

    if datalad_bin:
        run_cmd([datalad_bin, "get", "-n", "."], cwd=ds_root, log_path=log_path, allow_fail=True)
        run_cmd(
            [datalad_bin, "get", "dataset_description.json", "participants.tsv"],
            cwd=ds_root,
            log_path=log_path,
            allow_fail=True,
        )
        run_shell(
            f"shopt -s globstar nullglob; {datalad_bin} get -r sub-*/**/eeg/*_events.tsv sub-*/**/eeg/*_events.tsv.gz "
            "sub-*/**/eeg/*.vhdr sub-*/**/eeg/*.set sub-*/**/eeg/*.edf sub-*/**/eeg/*.bdf sub-*/**/eeg/*.fif sub-*/**/eeg/*.gdf",
            cwd=ds_root,
            log_path=log_path,
            allow_fail=True,
        )

    head = run_cmd(["git", "rev-parse", "HEAD"], cwd=ds_root).stdout.strip()
    if not used_hash:
        used_hash = head

    desc = ds_root / "dataset_description.json"
    participants = ds_root / "participants.tsv"
    eeg_count = (
        sum(1 for _ in ds_root.rglob("*.vhdr"))
        + sum(1 for _ in ds_root.rglob("*.set"))
        + sum(1 for _ in ds_root.rglob("*.edf"))
        + sum(1 for _ in ds_root.rglob("*.bdf"))
    )
    event_count = sum(1 for _ in ds_root.rglob("*_events.tsv")) + sum(1 for _ in ds_root.rglob("*_events.tsv.gz"))

    return {
        "dataset_id": dataset_id,
        "dataset_root": str(ds_root),
        "git_url": git_url,
        "pinned_hash_requested": pin,
        "checked_out_commit": head,
        "pinned_hash_used": used_hash,
        "dataset_description_exists": desc.exists(),
        "participants_exists": participants.exists(),
        "eeg_header_count": int(eeg_count),
        "n_event_files": int(event_count),
        "status": "PASS",
        "reason": "",
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def zip_dir(src_dir: Path, zip_path: Path) -> int:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(src_dir))
                n += 1
    return n


def tar_results_only(out_root: Path, tar_path: Path) -> int:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tar_abs = tar_path.resolve()
    exclude_tokens = [
        "openneuro",
        "raw",
        "annex",
        "runner_out",
        "_features_cache",
        "nvidia_smi_1hz.csv",
    ]

    def _include(p: Path) -> bool:
        try:
            if p.resolve() == tar_abs:
                return False
        except Exception:
            pass
        s = str(p).lower()
        return not any(tok in s for tok in exclude_tokens)

    with tarfile.open(tar_path, "w:gz") as tf:
        for p in sorted(out_root.rglob("*")):
            if not p.is_file():
                continue
            if not _include(p):
                continue
            tf.add(p, arcname=str(p.relative_to(out_root)))
            n += 1
    return n


def copytree_files(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
    return n


def parse_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        return [dict(r) for r in rdr]


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def ensure_stage_status(audit_dir: Path, stage: str, status: str, extra: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {
        "stage": stage,
        "status": status,
        "timestamp_utc": iso_now(),
    }
    if extra:
        payload.update(extra)
    write_json(audit_dir / f"{stage}_summary.json", payload)


def command_env(repo_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    src = str(repo_root / "src")
    py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}:{py}" if py else src
    return env


def ensure_requirements(repo_root: Path, audit_log: Path) -> None:
    venv = repo_root / ".venv_bulletproof"
    req = repo_root / "requirements.txt"
    marker = venv / ".deps_ok"
    if not venv.exists():
        run_cmd(["python3", "-m", "venv", str(venv)], cwd=repo_root, log_path=audit_log)
    if marker.exists() and req.exists() and marker.stat().st_mtime >= req.stat().st_mtime:
        os.environ["BULLETPROOF_PYTHON"] = str(venv / "bin" / "python")
        return
    pip = venv / "bin" / "pip"
    python = venv / "bin" / "python"
    run_cmd([str(pip), "install", "--upgrade", "pip", "setuptools", "wheel"], cwd=repo_root, log_path=audit_log)
    run_cmd([str(pip), "install", "-r", str(req)], cwd=repo_root, log_path=audit_log)
    marker.write_text(iso_now() + "\n", encoding="utf-8")
    os.environ["BULLETPROOF_PYTHON"] = str(python)


def bulletproof_python() -> str:
    return os.environ.get("BULLETPROOF_PYTHON", sys.executable)
