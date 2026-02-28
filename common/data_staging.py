"""DataLad-based dataset staging and BIDS integrity validation."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    git_url: str
    pinned_hash: str


_DEFAULT_OPENNEURO_ROOT = Path("/filesystemHcog/openneuro")
_DEFAULT_DATASETS: Tuple[DatasetSpec, ...] = (
    DatasetSpec(id="ds003838", git_url="https://github.com/OpenNeuroDatasets/ds003838.git", pinned_hash="2b29295"),
    DatasetSpec(id="ds005095", git_url="https://github.com/OpenNeuroDatasets/ds005095.git", pinned_hash="51de64a"),
    DatasetSpec(id="ds003655", git_url="https://github.com/OpenNeuroDatasets/ds003655.git", pinned_hash="4807ef6"),
    DatasetSpec(id="ds004117", git_url="https://github.com/OpenNeuroDatasets/ds004117.git", pinned_hash="065e388"),
)


def _run(cmd: Sequence[str], *, cwd: Optional[Path] = None, stream: bool = False) -> str:
    where = cwd.as_posix() if cwd is not None else os.getcwd()
    if not stream:
        p = subprocess.run(
            list(cmd),
            cwd=where,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if p.returncode != 0:
            raise RuntimeError(
                f"Command failed (exit={p.returncode}) in {where}: {' '.join(cmd)}\n{p.stdout}"
            )
        return p.stdout.strip()

    proc = subprocess.Popen(
        list(cmd),
        cwd=where,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: deque[str] = deque(maxlen=400)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        tail.append(line)

    rc = proc.wait()
    out = "\n".join(tail).strip()
    if rc != 0:
        raise RuntimeError(
            f"Command failed (exit={rc}) in {where}: {' '.join(cmd)}\n{out}"
        )
    return out


def _datalad_exe() -> str:
    return str(os.environ.get("DATALAD_BIN", "datalad"))


def _datalad_jobs() -> str:
    env_jobs = os.environ.get("DATALAD_GET_JOBS")
    if env_jobs:
        return str(env_jobs)
    n = os.cpu_count() or 8
    return str(max(8, min(64, int(n))))


def _datalad_source() -> str:
    return str(os.environ.get("DATALAD_GET_SOURCE", "s3-PUBLIC"))


def ensure_datalad_available() -> str:
    exe = _datalad_exe()
    datalad = shutil.which(exe)
    if datalad is None:
        raise RuntimeError(
            f"datalad is required but not found in PATH (DATALAD_BIN={exe}). Install datalad/git-annex before staging."
        )
    return _run([exe, "--version"])


def _git_head(repo_dir: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=repo_dir)


def _git_checkout(repo_dir: Path, rev: str) -> None:
    # OpenNeuro mirrors often include non-git remotes (e.g., s3-PUBLIC); fetch origin only.
    try:
        _run(["git", "fetch", "origin", "--tags"], cwd=repo_dir)
    except Exception:
        # If network fetch fails but rev is already present locally, checkout can still succeed.
        pass
    _run(["git", "checkout", rev], cwd=repo_dir)


def _clone_if_needed(spec: DatasetSpec, root: Path) -> Path:
    ds_dir = root / spec.id
    if not ds_dir.exists():
        _run([_datalad_exe(), "clone", spec.git_url, ds_dir.as_posix()], stream=True)
    elif not (ds_dir / ".git").exists():
        raise RuntimeError(f"Expected git dataset directory but '.git' missing: {ds_dir}")
    return ds_dir


def _datalad_get_all(ds_dir: Path) -> None:
    jobs = _datalad_jobs()
    source = _datalad_source()
    batch_size = int(os.environ.get("DATALAD_GET_BATCH_SIZE", "64"))

    # OpenNeuro annex trees can deadlock with `datalad get .` on old git-annex builds.
    # Use explicit missing-key batches; this is still resumable/idempotent via git-annex.
    missing_raw = _run(["git", "annex", "find", "--not", "--in=here"], cwd=ds_dir)
    missing = [x.strip() for x in missing_raw.splitlines() if x.strip()]
    total = len(missing)
    if total == 0:
        print(f"[stage_data] no missing annex keys in {ds_dir}", flush=True)
        return

    jobs_i = int(jobs)
    mode = f"-J {jobs_i}" if jobs_i > 1 else "default single-job mode"
    print(
        f"[stage_data] fetching {total} annex file(s) from {source} "
        f"in batches of {batch_size} using {mode}",
        flush=True,
    )

    for i in range(0, total, batch_size):
        batch = missing[i : i + batch_size]
        batch_no = (i // batch_size) + 1
        n_batches = (total + batch_size - 1) // batch_size
        print(f"[stage_data] batch {batch_no}/{n_batches}: {len(batch)} file(s)", flush=True)
        cmd = ["git", "annex", "get", "--from", source]
        if jobs_i > 1:
            cmd += ["-J", str(jobs_i)]
        cmd += ["--", *batch]
        try:
            _run(cmd, cwd=ds_dir, stream=True)
        except Exception as e:
            print(
                f"[stage_data] source-pinned annex get failed ({e}); retrying batch without --from",
                flush=True,
            )
            fallback_cmd = ["git", "annex", "get"]
            if jobs_i > 1:
                fallback_cmd += ["-J", str(jobs_i)]
            fallback_cmd += ["--", *batch]
            _run(fallback_cmd, cwd=ds_dir, stream=True)

    missing_after_raw = _run(["git", "annex", "find", "--not", "--in=here"], cwd=ds_dir)
    missing_after = [x.strip() for x in missing_after_raw.splitlines() if x.strip()]
    if missing_after:
        preview = "\n".join(missing_after[:20])
        raise RuntimeError(
            f"Annex retrieval incomplete in {ds_dir}: {len(missing_after)} file(s) still missing.\n"
            f"First missing:\n{preview}"
        )


def _files_and_bytes(root: Path) -> Tuple[int, int]:
    n_files = 0
    n_bytes = 0
    for p in root.rglob("*"):
        if p.is_file():
            n_files += 1
            try:
                n_bytes += p.stat().st_size
            except OSError:
                pass
    return n_files, n_bytes


def _read_tsv_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, [])
    # Handle UTF-8 BOM that appears in some OpenNeuro TSV headers.
    return [str(x).strip().lstrip("\ufeff") for x in header]


def _find_eeg_files(dataset_dir: Path) -> List[Path]:
    eeg_exts = (
        ".edf",
        ".edf.gz",
        ".bdf",
        ".bdf.gz",
        ".set",
        ".fdt",
        ".vhdr",
        ".fif",
        ".eeg",
        ".cnt",
        ".gdf",
    )
    excluded_dirs = {"code", "derivatives", "sourcedata"}
    out: List[Path] = []
    for p in dataset_dir.rglob("*_eeg*"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(dataset_dir).parts
        if any(part in excluded_dirs for part in rel_parts):
            continue
        if "eeg" not in rel_parts:
            continue
        if not any(part.startswith("sub-") for part in rel_parts):
            continue
        name = p.name.lower()
        if name.endswith(".json") or name.endswith(".tsv") or name.endswith(".tsv.gz"):
            continue
        if any(name.endswith(ext) for ext in eeg_exts):
            out.append(p)
    return sorted(set(out))


def _find_events_files(dataset_dir: Path) -> List[Path]:
    files = list(dataset_dir.rglob("*_events.tsv")) + list(dataset_dir.rglob("*_events.tsv.gz"))
    return sorted(set([p for p in files if p.is_file()]))


def _find_eyetrack_files(dataset_dir: Path) -> List[Path]:
    files = (
        list(dataset_dir.rglob("*_eyetrack.tsv"))
        + list(dataset_dir.rglob("*_eyetrack.tsv.gz"))
        + list(dataset_dir.rglob("*_pupil.tsv"))
        + list(dataset_dir.rglob("*_pupil.tsv.gz"))
    )
    return sorted(set([p for p in files if p.is_file()]))


def _matching_events_for_eeg(eeg_path: Path) -> List[Path]:
    name = eeg_path.name
    if "_eeg" not in name:
        return []
    prefix = name.split("_eeg", 1)[0]
    return [
        eeg_path.with_name(prefix + "_events.tsv"),
        eeg_path.with_name(prefix + "_events.tsv.gz"),
    ]


def _is_memory_task_event(path: Path) -> bool:
    name = path.name.lower()
    m = re.search(r"_task-([a-z0-9]+)_", name)
    if not m:
        return True
    task = m.group(1)
    keywords = ("memory", "sternberg", "workingmemory", "wm", "nback")
    return any(k in task for k in keywords)


def _has_load_mapping_columns(
    *,
    spec: DatasetSpec,
    event_path: Path,
    cols: Sequence[str],
    load_mapping_columns: Sequence[str],
) -> bool:
    colset = set(cols)
    if any(c in colset for c in load_mapping_columns):
        return True

    # ds003838 encodes memory load in trigger/value columns.
    if spec.id == "ds003838" and {"value", "trial_type"} & colset:
        return True
    return False


def validate_dataset(
    *,
    spec: DatasetSpec,
    dataset_dir: Path,
    load_columns_priority: Sequence[str],
    load_mapping_columns: Sequence[str],
    pupil_columns_priority: Sequence[str],
    require_mechanism_pupil: bool,
) -> Dict[str, Any]:
    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset directory does not exist: {dataset_dir}")
    if not (dataset_dir / "dataset_description.json").exists():
        raise RuntimeError(f"Missing BIDS dataset_description.json in {dataset_dir}")

    eeg_files = _find_eeg_files(dataset_dir)
    events_files = _find_events_files(dataset_dir)

    if not eeg_files:
        raise RuntimeError(f"Dataset {spec.id}: no EEG files found.")
    if not events_files:
        raise RuntimeError(f"Dataset {spec.id}: no events.tsv files found.")

    # Per-run fail-fast: each EEG file must have a matching events file in same directory.
    missing_events: List[str] = []
    matched_events: List[Path] = []
    for eeg in eeg_files:
        candidates = _matching_events_for_eeg(eeg)
        existing = [c for c in candidates if c.exists()]
        if not existing:
            missing_events.append(str(eeg))
            continue
        matched_events.append(existing[0])
    if missing_events:
        preview = "\n".join(missing_events[:10])
        raise RuntimeError(
            f"Dataset {spec.id}: missing matching *_events.tsv for {len(missing_events)} EEG recording(s)."
            f"\nFirst missing:\n{preview}"
        )

    # Column integrity: validate matched EEG events only.
    # Require load columns for memory tasks, allow config-mapping fallback columns.
    bad_events: List[str] = []
    for ev in sorted(set(matched_events)):
        cols = set(_read_tsv_header(ev))
        if ("onset" not in cols) and ("sample" not in cols):
            bad_events.append(f"{ev} (missing timing column: onset or sample)")
            continue
        if not _is_memory_task_event(ev):
            continue
        if any(c in cols for c in load_columns_priority):
            continue
        if _has_load_mapping_columns(
            spec=spec,
            event_path=ev,
            cols=tuple(cols),
            load_mapping_columns=load_mapping_columns,
        ):
            continue
        bad_events.append(
            f"{ev} (missing any load columns {list(load_columns_priority)} "
            f"or mapping columns {list(load_mapping_columns)})"
        )
    if bad_events:
        preview = "\n".join(bad_events[:10])
        raise RuntimeError(
            f"Dataset {spec.id}: required event/load columns missing in {len(bad_events)} matched events file(s)."
            f"\nFirst failures:\n{preview}"
        )

    # Sanity check for full event inventory.
    if not events_files:
        raise RuntimeError(f"Dataset {spec.id}: no events.tsv files found.")

    eyetrack_files = _find_eyetrack_files(dataset_dir)
    if spec.id == "ds003838" and require_mechanism_pupil:
        if not eyetrack_files:
            raise RuntimeError(
                "Dataset ds003838 must include eyetracking/pupil files for LC-NE mechanism module, but none were found."
            )
        # Ensure at least one eyetrack file has a known pupil column.
        has_pupil_col = False
        for ey in eyetrack_files[:20]:
            cols = set(_read_tsv_header(ey))
            if any(c in cols for c in pupil_columns_priority):
                has_pupil_col = True
                break
        if not has_pupil_col:
            raise RuntimeError(
                "Dataset ds003838 eyetrack files found, but no known pupil column was detected in sampled headers."
            )

    n_files, n_bytes = _files_and_bytes(dataset_dir)
    return {
        "id": spec.id,
        "path": dataset_dir.as_posix(),
        "eeg_file_count": len(eeg_files),
        "events_file_count": len(events_files),
        "matched_events_file_count": len(set(matched_events)),
        "eyetrack_file_count": len(eyetrack_files),
        "file_count": n_files,
        "disk_usage_bytes": n_bytes,
    }


def load_dataset_config(config_path: Optional[Path]) -> Tuple[Path, List[DatasetSpec]]:
    if config_path is None:
        return _DEFAULT_OPENNEURO_ROOT, list(_DEFAULT_DATASETS)

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("datasets config root must be a mapping")

    root = Path(cfg.get("openneuro_root", _DEFAULT_OPENNEURO_ROOT.as_posix()))
    entries = cfg.get("datasets", None)
    if not isinstance(entries, list) or not entries:
        raise ValueError("datasets config must contain a non-empty 'datasets' list")

    out: List[DatasetSpec] = []
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError("each datasets entry must be a mapping")
        for req in ["id", "git_url", "pinned_hash"]:
            if req not in row:
                raise ValueError(f"dataset entry missing '{req}': {row}")
        out.append(DatasetSpec(id=str(row["id"]), git_url=str(row["git_url"]), pinned_hash=str(row["pinned_hash"])))

    return root, out


def stage_datasets(
    *,
    openneuro_root: Path,
    datasets: Sequence[DatasetSpec],
    manifest_out: Path,
    load_columns_priority: Sequence[str],
    load_mapping_columns: Sequence[str],
    pupil_columns_priority: Sequence[str],
    require_mechanism_pupil: bool,
) -> Dict[str, Any]:
    version = ensure_datalad_available()
    print(f"[stage_data] datalad version: {version}", flush=True)
    print(f"[stage_data] openneuro_root={openneuro_root}", flush=True)

    openneuro_root.mkdir(parents=True, exist_ok=True)
    staged: List[Dict[str, Any]] = []

    for spec in datasets:
        print(f"[stage_data] === dataset {spec.id} ===", flush=True)
        print(f"[stage_data] source={spec.git_url}", flush=True)
        print(f"[stage_data] pinned_hash={spec.pinned_hash}", flush=True)
        ds_dir = _clone_if_needed(spec, openneuro_root)
        _git_checkout(ds_dir, spec.pinned_hash)
        _datalad_get_all(ds_dir)

        checked_out = _git_head(ds_dir)
        if not checked_out.startswith(spec.pinned_hash):
            raise RuntimeError(
                f"Dataset {spec.id}: checkout mismatch. Expected prefix {spec.pinned_hash}, got {checked_out}"
            )

        integrity = validate_dataset(
            spec=spec,
            dataset_dir=ds_dir,
            load_columns_priority=load_columns_priority,
            load_mapping_columns=load_mapping_columns,
            pupil_columns_priority=pupil_columns_priority,
            require_mechanism_pupil=require_mechanism_pupil,
        )

        staged.append(
            {
                "id": spec.id,
                "git_url": spec.git_url,
                "pinned_hash": spec.pinned_hash,
                "checked_out_hash": checked_out,
                "staged_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **integrity,
            }
        )
        print(f"[stage_data] complete {spec.id} @ {checked_out}", flush=True)

    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "openneuro_root": openneuro_root.as_posix(),
        "datalad_version": version,
        "datasets": staged,
    }

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_out.with_suffix(manifest_out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(manifest_out)
    return payload
