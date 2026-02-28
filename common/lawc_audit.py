"""Shared Law-C event mapping, RT extraction, and audit helpers."""

from __future__ import annotations

import gzip
import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import yaml


RT_COLUMNS_PRIORITY: List[str] = [
    "response_time",
    "reaction_time",
    "rt",
    "RT",
    "response.rt",
    "resp_rt",
    "responseTime",
]

ACCURACY_COLUMNS_PRIORITY: List[str] = [
    "correct",
    "accuracy",
    "is_correct",
]


def load_lawc_event_map(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Law-C event map missing: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Law-C event map must be a mapping: {path}")
    defaults = cfg.get("defaults", {})
    datasets = cfg.get("datasets", {})
    if not isinstance(defaults, dict):
        raise ValueError("lawc_event_map defaults must be a mapping")
    if not isinstance(datasets, dict):
        raise ValueError("lawc_event_map datasets must be a mapping")
    return {"defaults": defaults, "datasets": datasets}


def subject_key_from_entities(dataset_id: str, bids_subject: str, bids_session: Optional[str]) -> str:
    dataset_id = str(dataset_id or "").strip()
    bids_subject = str(bids_subject or "").strip()
    if not dataset_id:
        raise ValueError("dataset_id is empty; cannot build subject_key")
    if not bids_subject:
        raise ValueError("bids_subject is empty; cannot build subject_key")
    if bids_session and str(bids_session).strip().lower() not in {"", "na", "none"}:
        return f"{dataset_id}:sub-{bids_subject}:ses-{bids_session}"
    return f"{dataset_id}:sub-{bids_subject}"


def _read_tsv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return pd.read_csv(f, sep="\t")
    return pd.read_csv(path, sep="\t")


def _safe_query(df: pd.DataFrame, query: str, context: str) -> pd.DataFrame:
    try:
        return df.query(query, engine="python")
    except Exception as exc:
        raise RuntimeError(f"Failed query for {context}: {query}\n{exc}") from exc


def _dedupe_events(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        c
        for c in ["onset", "sample", "trial_type", "value", "event_type", "task_role", "trial", "memory_cond"]
        if c in df.columns
    ]
    if not keys:
        return df.copy()
    return df.drop_duplicates(subset=keys, keep="first").copy()


def _infer_rt_seconds(rt: np.ndarray) -> Tuple[np.ndarray, str]:
    out = np.asarray(rt, dtype=float)
    finite = out[np.isfinite(out)]
    if finite.size == 0:
        return out, "no-finite-rt"
    med = float(np.median(finite))
    if med > 50.0:
        return out / 1000.0, "ms->s"
    return out, "s"


def _parse_accuracy_from_response_row(row: pd.Series) -> float:
    if "task_role" in row.index:
        role = str(row.get("task_role", "")).lower()
        if "correct" in role:
            return 1.0
        if "incorrect" in role:
            return 0.0

    for c in ACCURACY_COLUMNS_PRIORITY:
        if c in row.index:
            v = pd.to_numeric(pd.Series([row[c]]), errors="coerce").to_numpy(dtype=float)[0]
            if np.isfinite(v):
                return float(v)

    if "trial_type" in row.index:
        tt = str(row.get("trial_type", "")).lower()
        if "correct" in tt:
            return 1.0
        if "error" in tt or "incorrect" in tt:
            return 0.0

    return float("nan")


def _derive_rt_next_response_any(
    probe_df: pd.DataFrame,
    events_df: pd.DataFrame,
    response_filter: Optional[str],
    response_accuracy_map: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    if response_filter:
        resp = _safe_query(events_df, response_filter, "response_filter")
    else:
        if "value" in events_df.columns:
            resp = events_df[pd.to_numeric(events_df["value"], errors="coerce").isin([10, 11])]
        else:
            resp = events_df.iloc[0:0]
    if resp.empty:
        return np.full(len(probe_df), np.nan), np.full(len(probe_df), np.nan)

    resp = resp.copy()
    resp["onset_s"] = pd.to_numeric(resp.get("onset"), errors="coerce")
    resp = resp[np.isfinite(resp["onset_s"])].sort_values("onset_s")

    rt = np.full(len(probe_df), np.nan, dtype=float)
    acc = np.full(len(probe_df), np.nan, dtype=float)

    response_field = "value" if "value" in resp.columns else None

    for i, prow in probe_df.reset_index(drop=True).iterrows():
        onset = float(prow["onset_s"])
        rr = resp[resp["onset_s"] >= onset]
        if rr.empty:
            continue
        r0 = rr.iloc[0]
        dt = float(r0["onset_s"]) - onset
        if dt < 0.05 or dt > 20.0:
            continue
        rt[i] = dt
        if response_accuracy_map and response_field is not None:
            raw_val = r0[response_field]
            parsed = pd.to_numeric(pd.Series([raw_val]), errors="coerce").iloc[0]
            if not np.isfinite(parsed):
                m = re.search(r"(\d+)", str(raw_val))
                if m is not None:
                    parsed = float(m.group(1))
            key = str(int(parsed)) if np.isfinite(parsed) else "-1"
            if key in response_accuracy_map:
                acc[i] = float(response_accuracy_map[key])
        if not np.isfinite(acc[i]):
            acc[i] = _parse_accuracy_from_response_row(r0)

    return rt, acc


def _derive_rt_next_response_regex(
    probe_df: pd.DataFrame,
    events_df: pd.DataFrame,
    response_filter: Optional[str],
    probe_match_regex: str,
    response_match_regex: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if response_filter:
        resp = _safe_query(events_df, response_filter, "response_filter")
    else:
        resp = events_df.iloc[0:0]
    if resp.empty or "trial_type" not in resp.columns or "trial_type" not in probe_df.columns:
        return np.full(len(probe_df), np.nan), np.full(len(probe_df), np.nan)

    resp = resp.copy()
    resp["onset_s"] = pd.to_numeric(resp.get("onset"), errors="coerce")
    resp = resp[np.isfinite(resp["onset_s"])].sort_values("onset_s")

    rt = np.full(len(probe_df), np.nan, dtype=float)
    acc = np.full(len(probe_df), np.nan, dtype=float)

    probe_re = re.compile(probe_match_regex)
    resp_re = re.compile(response_match_regex)

    for i, prow in probe_df.reset_index(drop=True).iterrows():
        onset = float(prow["onset_s"])
        pm = probe_re.search(str(prow.get("trial_type", "")))
        if pm is None:
            continue
        pload = pm.groupdict().get("load", "") or (pm.group(1) if pm.groups() else "")
        pcond = pm.groupdict().get("cond", "")

        rr = resp[resp["onset_s"] >= onset]
        if rr.empty:
            continue

        for _, rrow in rr.iterrows():
            rm = resp_re.search(str(rrow.get("trial_type", "")))
            if rm is None:
                continue
            rload = rm.groupdict().get("load", "") or (rm.group(1) if rm.groups() else "")
            rcond = rm.groupdict().get("cond", "")
            if str(rload) != str(pload):
                continue
            if pcond and rcond and str(pcond) != str(rcond):
                continue
            dt = float(rrow["onset_s"]) - onset
            if dt < 0.05 or dt > 20.0:
                continue
            rt[i] = dt
            racc = rm.groupdict().get("acc", "")
            if str(racc).lower() == "correct":
                acc[i] = 1.0
            elif str(racc).lower() in {"error", "incorrect"}:
                acc[i] = 0.0
            else:
                acc[i] = _parse_accuracy_from_response_row(rrow)
            break

    return rt, acc


def _derive_rt_same_trial_click(
    probe_df: pd.DataFrame,
    events_df: pd.DataFrame,
    response_filter: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    if "trial" not in probe_df.columns:
        return np.full(len(probe_df), np.nan), np.full(len(probe_df), np.nan)
    if response_filter:
        resp = _safe_query(events_df, response_filter, "response_filter")
    else:
        resp = events_df.iloc[0:0]
    if resp.empty or "trial" not in resp.columns:
        return np.full(len(probe_df), np.nan), np.full(len(probe_df), np.nan)

    resp = resp.copy()
    resp["onset_s"] = pd.to_numeric(resp.get("onset"), errors="coerce")
    resp = resp[np.isfinite(resp["onset_s"])].sort_values(["trial", "onset_s"])

    rt = np.full(len(probe_df), np.nan, dtype=float)
    acc = np.full(len(probe_df), np.nan, dtype=float)

    for i, prow in probe_df.reset_index(drop=True).iterrows():
        trial = prow.get("trial")
        onset = float(prow["onset_s"])
        rr = resp[(resp["trial"] == trial) & (resp["onset_s"] >= onset)]
        if rr.empty:
            continue
        r0 = rr.iloc[0]
        dt = float(r0["onset_s"]) - onset
        if dt < 0.05 or dt > 20.0:
            continue
        rt[i] = dt
        acc[i] = _parse_accuracy_from_response_row(r0)

    return rt, acc


def _candidate_behavior_files(dataset_root: Path, subject: str, task: Optional[str], run: Optional[str], session: Optional[str]) -> List[Path]:
    subj_token = f"sub-{subject}"
    sess_token = f"ses-{session}" if session and str(session).lower() not in {"", "na", "none"} else None
    task_token = f"task-{task}" if task and str(task).lower() not in {"", "na", "none"} else None
    run_token = f"run-{run}" if run and str(run).lower() not in {"", "na", "none"} else None

    out: List[Path] = []
    patterns = ["*_beh.tsv", "*_beh.tsv.gz", "*_events.tsv", "*_events.tsv.gz"]
    search_roots = [dataset_root]
    ddir = dataset_root / "derivatives"
    if ddir.exists():
        search_roots.append(ddir)

    for root in search_roots:
        for pat in patterns:
            for p in root.rglob(pat):
                name = p.name
                if subj_token not in name:
                    continue
                if sess_token and sess_token not in name:
                    continue
                if task_token and task_token not in name:
                    continue
                if run_token and run_token not in name:
                    continue
                out.append(p)
    return sorted(set(out))


def _attempt_rt_from_behavior_files(
    probe_df: pd.DataFrame,
    dataset_root: Path,
    subject: str,
    task: Optional[str],
    run: Optional[str],
    session: Optional[str],
) -> Tuple[np.ndarray, str, List[str]]:
    candidates = _candidate_behavior_files(dataset_root, subject=subject, task=task, run=run, session=session)
    checked: List[str] = [str(p) for p in candidates]
    for p in candidates:
        try:
            bdf = _read_tsv(p)
        except Exception:
            continue
        for c in RT_COLUMNS_PRIORITY:
            if c not in bdf.columns:
                continue
            rt = pd.to_numeric(bdf[c], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(rt).mean() < 0.5:
                continue
            if len(rt) < len(probe_df):
                continue
            rt = rt[: len(probe_df)]
            rt, rt_units = _infer_rt_seconds(rt)
            return rt, f"beh-file:{p}:{c}({rt_units})", checked
    return np.full(len(probe_df), np.nan), "none", checked


def _locate_sourcedata_sternberg_log(
    dataset_root: Path,
    subject: str,
    session: Optional[str],
) -> Optional[Path]:
    sid = str(subject).strip()
    if not sid:
        return None
    subj_tokens = [sid]
    if sid.isdigit():
        subj_tokens.append(sid.zfill(2))
        subj_tokens.append(sid.zfill(3))
    subj_tokens = list(dict.fromkeys(subj_tokens))

    candidates: List[Path] = []
    for tok in subj_tokens:
        sub_dir = dataset_root / "sourcedata" / f"sub-{tok}" / "logfiles"
        if not sub_dir.exists():
            continue
        for pat in [
            f"sub-{tok}_task-sternberg_events.txt",
            f"sub-{tok}_task-sternberg*events*.txt",
            f"*task-sternberg*events*.txt",
        ]:
            candidates.extend(sorted(sub_dir.glob(pat)))

    # Session-specific logs (rare in ds004796, but handled for completeness)
    if session and str(session).strip().lower() not in {"", "na", "none"}:
        sess = str(session).strip()
        for tok in subj_tokens:
            sub_dir = dataset_root / "sourcedata" / f"sub-{tok}" / f"ses-{sess}" / "logfiles"
            if not sub_dir.exists():
                continue
            for pat in [
                f"sub-{tok}_ses-{sess}_task-sternberg_events.txt",
                f"sub-{tok}_ses-{sess}_task-sternberg*events*.txt",
                f"*task-sternberg*events*.txt",
            ]:
                candidates.extend(sorted(sub_dir.glob(pat)))

    if not candidates:
        return None
    # Prefer the canonical merged sternberg log if present.
    for p in candidates:
        if p.name.endswith("_task-sternberg_events.txt"):
            return p
    return candidates[0]


def _parse_sourcedata_sternberg_log(log_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    try:
        df = pd.read_csv(log_path, sep="\t")
    except Exception as exc:
        raise RuntimeError(f"failed to read sourcedata sternberg log: {log_path} ({exc})") from exc

    if df.empty:
        raise RuntimeError(f"sourcedata sternberg log is empty: {log_path}")

    df.columns = [str(c).strip() for c in df.columns]
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = df.dropna(how="all").reset_index(drop=True)

    if "Event Type" not in df.columns or "Code" not in df.columns:
        raise RuntimeError(
            f"sourcedata sternberg log missing required columns Event Type/Code: {log_path} cols={list(df.columns)}"
        )

    for c in ["Trial", "Time", "TTime"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    records: List[Dict[str, Any]] = []
    current_mem_letters: Optional[List[str]] = None

    for _, row in df.iterrows():
        etype = str(row.get("Event Type", "")).strip().lower()
        code_raw = str(row.get("Code", "")).strip()

        if etype == "picture":
            letters = [x.upper() for x in re.findall(r"[A-Za-z]", code_raw)]
            # Memory-set row carries multiple letters (often with spaces/+ delimiters).
            if len(letters) >= 2 and ("+" in code_raw or " " in code_raw):
                current_mem_letters = letters
                continue

            # Probe row is a single letter.
            if re.fullmatch(r"[A-Za-z]", code_raw):
                if not current_mem_letters:
                    continue
                probe_letter = code_raw.upper()
                records.append(
                    {
                        "load": float(len(current_mem_letters)),
                        "probe_letter": probe_letter,
                        "is_target": float(probe_letter in current_mem_letters),
                        "response_code": np.nan,
                        "rt": np.nan,
                    }
                )
                continue

        if etype == "response":
            if not records:
                continue
            # Attach response to the most recent probe without an assigned response.
            idx = len(records) - 1
            while idx >= 0 and np.isfinite(float(records[idx].get("response_code", np.nan))):
                idx -= 1
            if idx < 0:
                continue

            parsed = pd.to_numeric(pd.Series([row.get("Code")]), errors="coerce").iloc[0]
            if not np.isfinite(parsed):
                m = re.search(r"(\d+)", str(row.get("Code", "")))
                if m is not None:
                    parsed = float(m.group(1))
            if np.isfinite(parsed):
                records[idx]["response_code"] = float(parsed)

            ttime = pd.to_numeric(pd.Series([row.get("TTime")]), errors="coerce").iloc[0]
            if np.isfinite(ttime):
                records[idx]["rt"] = float(ttime)

    if not records:
        raise RuntimeError(f"sourcedata sternberg log yielded zero probe records: {log_path}")

    out = pd.DataFrame(records)
    out = out[np.isfinite(pd.to_numeric(out["load"], errors="coerce"))].reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"sourcedata sternberg log produced zero finite-load probes: {log_path}")

    # Infer response-key semantics (yes/no) from memory-membership labels when possible.
    response_codes = sorted(set(pd.to_numeric(out["response_code"], errors="coerce").dropna().astype(int).tolist()))
    yes_code: Optional[int] = None
    yes_code_score = float("nan")
    if len(response_codes) >= 2:
        fit = out.dropna(subset=["response_code", "is_target"]).copy()
        fit["response_code"] = pd.to_numeric(fit["response_code"], errors="coerce")
        if not fit.empty and fit["is_target"].nunique() >= 2:
            scores: List[Tuple[int, float]] = []
            for c in response_codes:
                pred_yes = fit["response_code"].astype(int) == int(c)
                acc = float((pred_yes.to_numpy(dtype=bool) == fit["is_target"].to_numpy(dtype=bool)).mean())
                scores.append((int(c), acc))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            if len(scores) >= 2:
                best_code, best_acc = scores[0]
                second_acc = scores[1][1]
                if best_acc >= 0.55 and (best_acc - second_acc) >= 0.05:
                    yes_code = int(best_code)
                    yes_code_score = float(best_acc)

    acc = np.full(len(out), np.nan, dtype=float)
    if yes_code is not None:
        resp = pd.to_numeric(out["response_code"], errors="coerce").to_numpy(dtype=float)
        tgt = pd.to_numeric(out["is_target"], errors="coerce").to_numpy(dtype=float)
        pred_yes = resp == float(yes_code)
        good = np.isfinite(resp) & np.isfinite(tgt)
        acc[good] = (pred_yes[good] == (tgt[good] > 0.5)).astype(float)
    out["accuracy"] = acc

    rt_raw = pd.to_numeric(out["rt"], errors="coerce").to_numpy(dtype=float)
    rt_s, rt_units = _infer_rt_seconds(rt_raw)
    out["rt"] = rt_s

    diag = {
        "log_path": str(log_path),
        "n_probe_records": int(len(out)),
        "load_levels": sorted(set(pd.to_numeric(out["load"], errors="coerce").dropna().astype(int).tolist())),
        "response_codes": response_codes,
        "inferred_yes_code": None if yes_code is None else int(yes_code),
        "inferred_yes_code_accuracy": yes_code_score,
        "rt_units": rt_units,
        "rt_nonmissing_rate": float(np.isfinite(pd.to_numeric(out["rt"], errors="coerce")).mean()),
        "accuracy_nonmissing_rate": float(np.isfinite(pd.to_numeric(out["accuracy"], errors="coerce")).mean()),
    }
    return out, diag


def prepare_probe_event_table(
    *,
    events_path: Path,
    dataset_id: str,
    event_map: Dict[str, Any],
    dataset_root: Optional[Path] = None,
    bids_subject: Optional[str] = None,
    bids_task: Optional[str] = None,
    bids_run: Optional[str] = None,
    bids_session: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    defaults = dict(event_map.get("defaults", {}))
    spec = dict((event_map.get("datasets", {}) or {}).get(dataset_id, {}))

    events_df = _read_tsv(events_path)
    events_df = _dedupe_events(events_df)

    diag: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "events_path": str(events_path),
        "events_columns": list(events_df.columns),
        "event_filter": spec.get("event_filter"),
        "load_column_requested": spec.get("load_column"),
        "load_sign": float(spec.get("load_sign", 1.0)),
        "rt_column_requested": spec.get("rt_column"),
        "rt_strategy": spec.get("rt_strategy", "auto"),
        "rt_columns_tried": [],
        "behavior_candidates_checked": [],
        "load_source": str(spec.get("load_source", "")),
        "accuracy_source": str(spec.get("accuracy_source", "")),
    }

    event_filter = spec.get("event_filter")
    if not event_filter:
        # Fail-closed unless inspection has generated explicit mapping.
        raise RuntimeError(
            f"Dataset {dataset_id}: no event_filter defined in lawc_event_map; run scripts/inspect_events.py and set an explicit mapping."
        )

    probe_df = _safe_query(events_df, str(event_filter), f"{dataset_id}.event_filter").copy()
    if probe_df.empty:
        raise RuntimeError(
            f"Dataset {dataset_id}: event_filter selected zero rows. filter={event_filter} events={events_path}"
        )

    probe_df["onset_s"] = pd.to_numeric(probe_df.get("onset"), errors="coerce")
    probe_df = probe_df[np.isfinite(probe_df["onset_s"])].copy()
    if probe_df.empty:
        raise RuntimeError(f"Dataset {dataset_id}: selected events have no finite onset values: {events_path}")

    # Memory-load extraction
    load_col = spec.get("load_column")
    if not load_col:
        candidates = [
            c
            for c in ["memory_load", "set_size", "load", "memory_cond", "value", "trial_type"]
            if c in probe_df.columns
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Dataset {dataset_id}: ambiguous load column candidates={candidates}. Set load_column in lawc_event_map."
            )
        load_col = candidates[0]
    if load_col not in probe_df.columns:
        raise RuntimeError(
            f"Dataset {dataset_id}: load_column '{load_col}' missing from events columns {list(probe_df.columns)}"
        )

    load_transform = str(spec.get("load_transform", "")).strip().lower()
    load_source = str(spec.get("load_source", "")).strip().lower()
    load_regex = spec.get("load_regex")
    load_value_map = spec.get("load_value_map")
    sourcedata_trials: Optional[pd.DataFrame] = None

    if load_source == "sourcedata_sternberg_log":
        if dataset_root is None or bids_subject is None:
            raise RuntimeError(
                f"Dataset {dataset_id}: load_source=sourcedata_sternberg_log requires dataset_root and bids_subject context."
            )
        log_path = _locate_sourcedata_sternberg_log(dataset_root, subject=bids_subject, session=bids_session)
        if log_path is None:
            raise RuntimeError(
                f"Dataset {dataset_id}: sourcedata sternberg log not found for sub-{bids_subject}."
            )
        sourcedata_trials, sd_diag = _parse_sourcedata_sternberg_log(log_path)
        diag["sourcedata_log"] = sd_diag
        if len(sourcedata_trials) != len(probe_df):
            raise RuntimeError(
                f"Dataset {dataset_id}: sourcedata probe count mismatch for sub-{bids_subject}. "
                f"log_n={len(sourcedata_trials)} events_probe_n={len(probe_df)} log={log_path}"
            )
        load = pd.to_numeric(sourcedata_trials["load"], errors="coerce").to_numpy(dtype=float)

    elif isinstance(load_value_map, dict) and load_value_map:
        # Auditable categorical->numeric remap for clinical datasets where
        # condition labels are encoded as strings.
        def _norm_text(x: Any) -> str:
            s = str(x).strip().lower()
            return re.sub(r"\s+", " ", s)

        map_norm: Dict[str, float] = {}
        for k, v in load_value_map.items():
            vv = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
            if np.isfinite(vv):
                map_norm[_norm_text(k)] = float(vv)

        src = probe_df[load_col].astype(str).to_numpy(dtype=object)
        load = np.full(len(src), np.nan, dtype=float)
        for i, x in enumerate(src.tolist()):
            key = _norm_text(x)
            if key in map_norm:
                load[i] = map_norm[key]
    elif load_transform == "sternberg_code_tens":
        raw = pd.to_numeric(probe_df[load_col], errors="coerce").to_numpy(dtype=float)
        load = np.floor(raw / 10.0)
    elif load_regex:
        # str.extract may return a DataFrame when regex has >1 capture group.
        ex = probe_df[load_col].astype(str).str.extract(str(load_regex), expand=True)
        if isinstance(ex, pd.DataFrame):
            if ex.shape[1] == 0:
                ex_series = pd.Series(np.nan, index=probe_df.index)
            else:
                # Prefer first non-null capture across groups, preserving row order.
                ex_series = ex.bfill(axis=1).iloc[:, 0]
        else:
            ex_series = ex
        load = pd.to_numeric(ex_series, errors="coerce").to_numpy(dtype=float)
    else:
        load = pd.to_numeric(probe_df[load_col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(load).mean() < 0.8:
            parsed = (
                probe_df[load_col]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
            )
            parsed_num = pd.to_numeric(parsed, errors="coerce").to_numpy(dtype=float)
            if np.isfinite(parsed_num).mean() > np.isfinite(load).mean():
                load = parsed_num

    if np.isfinite(load).mean() < 0.8:
        raise RuntimeError(
            f"Dataset {dataset_id}: memory_load extraction failed (<80% finite). load_column={load_col}"
        )

    load_sign = float(spec.get("load_sign", 1.0))
    if bool(spec.get("invert_load", False)):
        load_sign *= -1.0
    if not np.isfinite(load_sign) or float(load_sign) == 0.0:
        raise RuntimeError(f"Dataset {dataset_id}: invalid load_sign={load_sign!r} in lawc_event_map")
    if float(load_sign) != 1.0:
        load = load * float(load_sign)

    probe_df = probe_df.copy()
    probe_df["memory_load"] = load

    # RT extraction
    rt = np.full(len(probe_df), np.nan, dtype=float)
    acc = np.full(len(probe_df), np.nan, dtype=float)
    rt_source = str(spec.get("rt_source", "")).strip().lower()
    accuracy_source = str(spec.get("accuracy_source", "")).strip().lower()

    if rt_source == "sourcedata_sternberg_log" and sourcedata_trials is not None:
        rt = pd.to_numeric(sourcedata_trials["rt"], errors="coerce").to_numpy(dtype=float)
        diag["rt_source"] = "sourcedata_sternberg_log"

    if accuracy_source == "sourcedata_sternberg_log" and sourcedata_trials is not None:
        acc = pd.to_numeric(sourcedata_trials["accuracy"], errors="coerce").to_numpy(dtype=float)
        diag["accuracy_source"] = "sourcedata_sternberg_log"

    rt_cols = []
    if spec.get("rt_column"):
        rt_cols.append(str(spec["rt_column"]))
    rt_cols.extend([c for c in defaults.get("rt_columns_priority", RT_COLUMNS_PRIORITY)])

    seen_cols = set()
    for c in rt_cols:
        if c in seen_cols:
            continue
        seen_cols.add(c)
        diag["rt_columns_tried"].append(c)
        if c not in probe_df.columns:
            continue
        vals = pd.to_numeric(probe_df[c], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(vals).mean() <= 0.0:
            continue
        rt, rt_units = _infer_rt_seconds(vals)
        diag["rt_source"] = f"events:{c}({rt_units})"
        break

    if not np.isfinite(rt).any():
        if {"response_onset", "probe_onset"}.issubset(probe_df.columns):
            vals = pd.to_numeric(probe_df["response_onset"], errors="coerce") - pd.to_numeric(
                probe_df["probe_onset"], errors="coerce"
            )
            rt, rt_units = _infer_rt_seconds(vals.to_numpy(dtype=float))
            diag["rt_source"] = f"events:response_onset-probe_onset({rt_units})"

    if not np.isfinite(rt).any():
        strategy = str(spec.get("rt_strategy", "")).strip().lower()
        response_filter = spec.get("response_filter")
        if strategy == "next_response_any":
            rt, acc = _derive_rt_next_response_any(
                probe_df=probe_df,
                events_df=events_df,
                response_filter=str(response_filter) if response_filter else None,
                response_accuracy_map={str(k): float(v) for k, v in (spec.get("response_accuracy_map") or {}).items()},
            )
            rt, rt_units = _infer_rt_seconds(rt)
            diag["rt_source"] = f"derived:next_response_any({rt_units})"
        elif strategy == "next_response_regex_match":
            probe_regex = str(spec.get("probe_match_regex", r"probe:\\s*(?P<load>\\d+)(?P<cond>[A-Za-z])"))
            resp_regex = str(spec.get("response_match_regex", r"Response:\\s*(?P<load>\\d+)(?P<cond>[A-Za-z]):\\s*(?P<acc>correct|error)"))
            rt, acc = _derive_rt_next_response_regex(
                probe_df=probe_df,
                events_df=events_df,
                response_filter=str(response_filter) if response_filter else None,
                probe_match_regex=probe_regex,
                response_match_regex=resp_regex,
            )
            rt, rt_units = _infer_rt_seconds(rt)
            diag["rt_source"] = f"derived:next_response_regex_match({rt_units})"
        elif strategy == "same_trial_click":
            rt, acc = _derive_rt_same_trial_click(
                probe_df=probe_df,
                events_df=events_df,
                response_filter=str(response_filter) if response_filter else None,
            )
            rt, rt_units = _infer_rt_seconds(rt)
            diag["rt_source"] = f"derived:same_trial_click({rt_units})"

    if not np.isfinite(rt).any() and dataset_root is not None and bids_subject is not None:
        rt_b, source, checked = _attempt_rt_from_behavior_files(
            probe_df=probe_df,
            dataset_root=dataset_root,
            subject=bids_subject,
            task=bids_task,
            run=bids_run,
            session=bids_session,
        )
        if np.isfinite(rt_b).any():
            rt = rt_b
            diag["rt_source"] = source
        diag["behavior_candidates_checked"] = checked

    if not np.isfinite(acc).any():
        for c in ACCURACY_COLUMNS_PRIORITY:
            if c in probe_df.columns:
                acc = pd.to_numeric(probe_df[c], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(acc).any():
                    break

    probe_df["rt"] = rt
    probe_df["accuracy"] = acc

    probe_df = probe_df.sort_values("onset_s").reset_index(drop=True)
    probe_df["trial_order"] = np.arange(1, len(probe_df) + 1, dtype=np.int32)

    diag["load_column_used"] = load_col
    diag["load_sign"] = float(load_sign)
    diag["n_events_total"] = int(len(events_df))
    diag["n_probe_selected"] = int(len(probe_df))
    diag["memory_load_nonmissing_rate"] = float(np.isfinite(probe_df["memory_load"]).mean())
    diag["rt_nonmissing_rate"] = float(np.isfinite(probe_df["rt"]).mean())
    diag.setdefault("rt_source", "none")

    keep_cols = ["onset_s", "memory_load", "rt", "accuracy", "trial_order"]
    for extra in ["trial_type", "event_type", "task_role", "value", "trial", "memory_cond"]:
        if extra in probe_df.columns:
            keep_cols.append(extra)
    return probe_df[keep_cols].copy(), diag


def align_probe_events_to_epochs(
    *,
    epoch_onsets_s: np.ndarray,
    probe_df: pd.DataFrame,
    tolerance_s: float = 0.01,
) -> pd.DataFrame:
    ep = pd.DataFrame(
        {
            "epoch_idx": np.arange(len(epoch_onsets_s), dtype=np.int64),
            "onset_s": pd.to_numeric(pd.Series(epoch_onsets_s), errors="coerce").to_numpy(dtype=float),
        }
    )

    pr = probe_df.copy()
    pr = pr[np.isfinite(pd.to_numeric(pr["onset_s"], errors="coerce"))].copy()
    pr["probe_onset_s"] = pd.to_numeric(pr["onset_s"], errors="coerce").to_numpy(dtype=float)
    pr = pr.sort_values("probe_onset_s").reset_index(drop=True)

    ep_sorted = ep.sort_values("onset_s").reset_index(drop=True)
    merged = pd.merge_asof(
        ep_sorted,
        pr,
        left_on="onset_s",
        right_on="probe_onset_s",
        direction="nearest",
        tolerance=float(tolerance_s),
        suffixes=("", "_probe"),
    )

    merged = merged.sort_values("epoch_idx").reset_index(drop=True)
    merged["matched"] = merged["probe_onset_s"].notna()
    return merged


def collect_lawc_trials_from_features(features_root: Path, dataset_id: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for fp in sorted(features_root.rglob("*.h5")):
        if not fp.is_file():
            continue
        with h5py.File(fp, "r") as h:
            attrs = {k: h.attrs[k] for k in h.attrs.keys()}
            ds_attr = str(attrs.get("dataset_id", ""))
            if ds_attr != dataset_id:
                continue
            if "p3b_amp" not in h or "memory_load" not in h:
                continue

            p3b = np.asarray(h["p3b_amp"], dtype=float)
            load = np.asarray(h["memory_load"], dtype=float)
            rt = np.asarray(h["rt"], dtype=float) if "rt" in h else np.full(len(p3b), np.nan)

            ch = np.asarray(h["p3b_channel"]).astype(str) if "p3b_channel" in h else np.asarray([""] * len(p3b))
            skey = str(attrs.get("subject_key", ""))
            if not skey:
                subj = str(attrs.get("subject", ""))
                ses = str(attrs.get("session", ""))
                skey = subject_key_from_entities(dataset_id=dataset_id, bids_subject=subj, bids_session=ses)

            n = len(p3b)
            rows.append(
                pd.DataFrame(
                    {
                        "subject_key": [skey] * n,
                        "dataset_id": [dataset_id] * n,
                        "p3b_amp": p3b,
                        "memory_load": load,
                        "rt": rt,
                        "p3b_channel": ch,
                    }
                )
            )

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, axis=0, ignore_index=True)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    xx = pd.Series(x[m]).rank(method="average").to_numpy(dtype=float)
    yy = pd.Series(y[m]).rank(method="average").to_numpy(dtype=float)
    if np.std(xx) <= 0 or np.std(yy) <= 0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def subject_level_rhos(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject_key",
    x_col: str = "p3b_amp",
    y_col: str = "memory_load",
    min_trials: int = 20,
) -> Dict[str, Any]:
    rhos: List[float] = []
    per_subject: List[Dict[str, Any]] = []

    for sid, g in df.groupby(subject_col):
        gg = g[[x_col, y_col]].dropna()
        if len(gg) < int(min_trials):
            continue
        rho = _spearman_rho(gg[x_col].to_numpy(dtype=float), gg[y_col].to_numpy(dtype=float))
        if np.isfinite(rho):
            rhos.append(float(rho))
            per_subject.append({"subject_key": str(sid), "n_trials": int(len(gg)), "rho": float(rho)})

    med = float(np.median(rhos)) if rhos else float("nan")
    return {
        "n_subjects_used": len(rhos),
        "median_rho": med,
        "subject_rhos": rhos,
        "per_subject": per_subject,
    }


def permutation_test_median_rho(
    df: pd.DataFrame,
    *,
    n_perm: int,
    seed: int,
    mode: str,
    subject_col: str = "subject_key",
    x_col: str = "p3b_amp",
    y_col: str = "memory_load",
    min_trials: int = 20,
) -> Dict[str, Any]:
    groups: List[Tuple[np.ndarray, np.ndarray]] = []
    for _, g in df.groupby(subject_col):
        gg = g[[x_col, y_col]].dropna()
        if len(gg) < int(min_trials):
            continue
        x = gg[x_col].to_numpy(dtype=float)
        y = gg[y_col].to_numpy(dtype=float)
        if np.std(x) <= 0 or np.std(y) <= 0:
            continue
        groups.append((x, y))

    if not groups:
        return {
            "n_subjects_used": 0,
            "observed_median_rho": float("nan"),
            "p_value": float("nan"),
            "perm_stats": [],
        }

    obs_rhos = [_spearman_rho(x, y) for x, y in groups]
    obs = float(np.nanmedian(np.asarray(obs_rhos, dtype=float)))

    rng = np.random.default_rng(int(seed))
    perm_stats = np.full(int(n_perm), np.nan, dtype=float)

    for p in range(int(n_perm)):
        rhos = []
        for x, y in groups:
            if mode == "x_shuffle":
                xp = rng.permutation(x)
                yp = y
            elif mode == "y_shuffle":
                xp = x
                yp = rng.permutation(y)
            else:
                raise ValueError(f"Unknown permutation mode: {mode}")
            rhos.append(_spearman_rho(xp, yp))
        perm_stats[p] = float(np.nanmedian(np.asarray(rhos, dtype=float)))

    finite_perm = perm_stats[np.isfinite(perm_stats)]
    if finite_perm.size == 0 or not np.isfinite(obs):
        p_value = float("nan")
    else:
        p_value = float((1.0 + np.sum(finite_perm >= obs)) / (1.0 + finite_perm.size))

    return {
        "n_subjects_used": len(groups),
        "observed_median_rho": obs,
        "p_value": p_value,
        "perm_stats": finite_perm.tolist(),
    }


def bh_fdr(pvals: Sequence[float]) -> List[float]:
    arr = np.asarray(pvals, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)

    finite_idx = np.where(np.isfinite(arr))[0]
    if finite_idx.size == 0:
        return out.tolist()

    pv = arr[finite_idx]
    order = np.argsort(pv)
    ranked = pv[order]
    m = float(len(ranked))
    q = np.empty_like(ranked)
    prev = 1.0
    for i in range(len(ranked) - 1, -1, -1):
        rank = i + 1.0
        val = min(prev, ranked[i] * m / rank)
        q[i] = val
        prev = val

    q_unsorted = np.empty_like(q)
    q_unsorted[order] = q
    out[finite_idx] = q_unsorted
    return out.tolist()


def checksum_array_sha256(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(arr))
    h = hashlib.sha256()
    h.update(a.tobytes(order="C"))
    return h.hexdigest()
