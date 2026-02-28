"""EEG preprocessing and P3b feature extraction (MNE).

This module implements the locked ERP definition used in your Law C runs:
- epoch: t = [-0.2, 0.8] s
- baseline: [-0.2, 0] s
- P3b amplitude: mean voltage in [0.35, 0.60] s at Pz (fallback CPz->Cz)

We intentionally keep preprocessing minimal and transparent. Anything that could
meaningfully change the results should be config-driven and logged.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import cfg_get
from .bids_utils import BIDSRun, load_events


def _read_raw_with_fallback(bids_root: Path, run: BIDSRun):
    """Read raw EEG using MNE-BIDS if possible, else fall back to MNE readers."""
    import mne

    def _read_brainvision_with_header_repair(vhdr_path: Path):
        """Read BrainVision data, repairing broken DataFile/MarkerFile pointers when possible."""

        def _try_read(path: Path):
            return mne.io.read_raw_brainvision(path, preload=True, verbose="ERROR")

        def _repair_header_if_needed(path: Path) -> Optional[Path]:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return None

            data_match = re.search(r"(?m)^DataFile=(.+)$", text)
            marker_match = re.search(r"(?m)^MarkerFile=(.+)$", text)
            if not data_match and not marker_match:
                return None

            changed = False
            if data_match:
                current_data = data_match.group(1).strip()
                current_data_path = (path.parent / current_data).resolve()
                canonical_data = path.with_suffix(".eeg")
                if (not current_data_path.exists()) and canonical_data.exists():
                    text = re.sub(
                        r"(?m)^DataFile=.+$",
                        f"DataFile={canonical_data.name}",
                        text,
                        count=1,
                    )
                    changed = True

            if marker_match:
                current_marker = marker_match.group(1).strip()
                current_marker_path = (path.parent / current_marker).resolve()
                canonical_marker = path.with_suffix(".vmrk")
                if (not current_marker_path.exists()) and canonical_marker.exists():
                    text = re.sub(
                        r"(?m)^MarkerFile=.+$",
                        f"MarkerFile={canonical_marker.name}",
                        text,
                        count=1,
                    )
                    changed = True

            if not changed:
                return None

            # Keep temp header in same directory so relative sidecar paths resolve.
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".vhdr",
                prefix=f"{path.stem}.autofix.",
                dir=path.parent,
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp.write(text)
                return Path(tmp.name)

        try:
            return _try_read(vhdr_path)
        except FileNotFoundError:
            repaired = _repair_header_if_needed(vhdr_path)
            if repaired is None:
                raise
            try:
                return _try_read(repaired)
            finally:
                repaired.unlink(missing_ok=True)

    def _direct_read():
        ext = run.eeg_path.suffix.lower()
        if ext in [".fif", ".gz"]:
            return mne.io.read_raw_fif(run.eeg_path, preload=True, verbose="ERROR")
        if ext == ".edf":
            return mne.io.read_raw_edf(run.eeg_path, preload=True, verbose="ERROR")
        if ext == ".bdf":
            return mne.io.read_raw_bdf(run.eeg_path, preload=True, verbose="ERROR")
        if ext == ".set":
            return mne.io.read_raw_eeglab(run.eeg_path, preload=True, verbose="ERROR")
        if ext == ".vhdr":
            return _read_brainvision_with_header_repair(run.eeg_path)
        if ext == ".eeg":
            # BrainVision datasets often include *_eeg.eeg data-part files with
            # the corresponding *_eeg.vhdr header in the same directory.
            vhdr = run.eeg_path.with_suffix(".vhdr")
            if vhdr.exists():
                return _read_brainvision_with_header_repair(vhdr)
            raise RuntimeError(f"Unsupported .eeg without companion .vhdr: {run.eeg_path}")
        if ext == ".gdf":
            return mne.io.read_raw_gdf(run.eeg_path, preload=True, verbose="ERROR")
        raise RuntimeError(f"Unsupported EEG file extension: {ext} ({run.eeg_path})")

    # Preferred: MNE-BIDS (respects sidecars, etc.)
    try:
        from mne_bids import BIDSPath, read_raw_bids

        bp = BIDSPath(
            root=bids_root.as_posix(),
            subject=run.subject,
            task=run.task,
            run=run.run,
            session=run.session,
            datatype="eeg",
            suffix="eeg",
            extension=run.eeg_path.suffix,
        )
        raw = read_raw_bids(bp, verbose="ERROR")
        n_eeg = len(mne.pick_types(raw.info, eeg=True, meg=False, seeg=True, ecog=True, dbs=True))
        if n_eeg > 0:
            return raw
    except Exception:
        pass

    # Fallback: direct reader by extension.
    return _direct_read()


def _select_trial_events(events_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    selector_col = cfg_get(cfg, "events.selector.column", "trial_type")
    include = cfg_get(cfg, "events.selector.include", [])
    if selector_col not in events_df.columns:
        fallback_cols = cfg_get(
            cfg,
            "events.selector.fallback_columns",
            ["task_role", "event_type", "value", "condition", "label"],
        )
        selector_col = next((c for c in fallback_cols if c in events_df.columns), "")
        if not selector_col:
            raise ValueError(
                f"events.tsv missing selector column '{cfg_get(cfg, 'events.selector.column', 'trial_type')}' "
                f"and no fallback column available. Present columns: {list(events_df.columns)}"
            )

    src = events_df[selector_col].astype(str)
    df = events_df.copy()
    if include:
        include_s = [str(x) for x in include]
        exact = src.isin(include_s)
        if exact.any():
            df = events_df.loc[exact].copy()
        else:
            # Fallback to substring match for datasets with verbose trial_type labels.
            low = src.str.lower()
            mask = pd.Series(False, index=events_df.index)
            for tok in include_s:
                tok_l = tok.strip().lower()
                if tok_l:
                    mask = mask | low.str.contains(tok_l, regex=False, na=False)
            if mask.any():
                df = events_df.loc[mask].copy()

    # Drop obvious non-trial status markers that pollute OpenNeuro event files.
    if selector_col in df.columns:
        lab = df[selector_col].astype(str).str.lower().str.strip()
        bad = lab.eq("status") | lab.str.contains("boundary", regex=False, na=False)
        if "value" in df.columns:
            v = df["value"].astype(str).str.lower().str.strip()
            bad = bad | v.eq("boundary") | v.eq("status")
        df = df.loc[~bad].copy()

    # Trial order = row index among selected events (1-indexed for interpretability)
    df = df.reset_index(drop=True)
    df["trial_order"] = np.arange(1, len(df) + 1, dtype=int)
    return df


def _extract_memory_load(df: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    cols = cfg_get(cfg, "events.load_columns_priority", ["memory_load", "set_size", "load"])
    for c in cols:
        if c in df.columns:
            # Try numeric
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().mean() > 0.8:
                return x.to_numpy(dtype=float)
    # Fallback: parse from mapping/text columns.
    import re

    mapping_cols = cfg_get(
        cfg,
        "events.load_mapping_columns",
        ["trial_type", "memory_cond", "value", "condition", "task_role", "event_type", "label"],
    )

    def parse_one(text: str) -> float:
        s = str(text).strip().lower()
        if not s:
            return np.nan
        if s in {"n/a", "na", "none", "nan", "status", "boundary"}:
            return np.nan

        # Common categorical encodings (e.g., ds004117 WM/nonWM).
        if any(tok in s for tok in ("nonwm", "non_wm", "non-wm", "to_ignore", "ignored")):
            return 0.0
        if any(tok in s for tok in ("work_memory", "to_remember", "remembered")):
            return 1.0
        if s == "wm":
            return 1.0

        m = re.search(r"\b\d+\s*/\s*(\d+)\b", s)
        if m:
            return float(m.group(1))

        m = re.search(r"(?:in|load)[^0-9]{0,6}(\d+)\s*digit", s)
        if m:
            return float(m.group(1))

        m = re.search(r"set[_\\s-]*size[_\\s-]*(\d+)", s)
        if m:
            return float(m.group(1))

        nums = re.findall(r"\d+", s)
        if nums:
            # Use max numeric token to avoid taking trial index from strings like 01/13.
            return float(max(int(x) for x in nums))
        return np.nan

    for c in mapping_cols:
        if c not in df.columns:
            continue
        out = [parse_one(x) for x in df[c].astype(str).tolist()]
        arr = np.asarray(out, dtype=float)
        if np.isfinite(arr).mean() > 0.6:
            return arr

    raise ValueError("Could not determine memory load from events.tsv (no numeric column and no parseable trial_type).")


def _is_memory_like_task(task: Optional[str]) -> bool:
    if task is None:
        return False
    t = str(task).strip().lower()
    return any(k in t for k in ("memory", "sternberg", "workingmemory", "wm", "nback"))


def _extract_optional_column(df: pd.DataFrame, cols_priority: List[str]) -> Optional[np.ndarray]:
    for c in cols_priority:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().mean() > 0.5:
                return x.to_numpy(dtype=float)
    return None


def _coerce_aux_channel_types(raw) -> None:
    """Retype obvious non-scalp channels so EEG rejection stays meaningful."""
    import re

    mapping: Dict[str, str] = {}
    for ch in list(raw.ch_names):
        name = str(ch).strip()
        upper = name.upper()
        if not upper:
            continue

        # EOG/HEOG/VEOG naming conventions in EEGLAB/OpenNeuro exports.
        if re.search(r"(^|[^A-Z])(VEOG|HEOG|EOG)([^A-Z]|$)", upper):
            mapping[name] = "eog"
            continue
        # ECG/EKG channels are frequently typed as EEG in source files.
        if re.search(r"(^|[^A-Z])(ECG|EKG)([^A-Z]|$)", upper):
            mapping[name] = "ecg"
            continue
        # EMG channels should not contribute to EEG peak-to-peak rejection.
        if re.search(r"(^|[^A-Z])EMG([^A-Z]|$)", upper):
            mapping[name] = "emg"
            continue

    if mapping:
        raw.set_channel_types(mapping, on_unit_change="ignore", verbose="ERROR")


def preprocess_and_epoch_eeg(
    *,
    bids_root: Path,
    run: BIDSRun,
    cfg: Dict[str, Any],
    age_years: Optional[float],
    out_epochs_fif: Path,
    n_jobs: int = 24,
) -> Optional[Path]:
    """CPU preprocessing + epoch saving.

    Output is a compressed MNE Epochs FIF with metadata attached, so that
    feature extraction is cheap and deterministic.

    Returns:
        Path to epochs FIF, or None if the run is skipped (e.g., missing events.tsv).
    """
    import mne

    if run.events_tsv is None:
        # Fail-closed: no auditable stimulus onsets.
        return None

    events_df = load_events(run.events_tsv)
    trials_df = _select_trial_events(events_df, cfg)

    if trials_df.empty:
        # Non-memory tasks (e.g., resting-state) are not used for load-linked ERP modeling.
        if not _is_memory_like_task(run.task):
            return None
        raise ValueError(f"No usable trial events after selection/filtering: {run.events_tsv}")

    memory_load = _extract_memory_load(trials_df, cfg)
    rt = _extract_optional_column(trials_df, cfg_get(cfg, "events.rt_columns_priority", []))
    acc = _extract_optional_column(trials_df, cfg_get(cfg, "events.acc_columns_priority", []))

    raw = _read_raw_with_fallback(bids_root, run)
    raw.load_data()
    _coerce_aux_channel_types(raw)

    # Minimal channel selection: keep EEG (and EOG if present for later QA).
    raw.pick_types(eeg=True, eog=True, misc=True, stim=False)
    data_picks = mne.pick_types(raw.info, eeg=True, meg=False, seeg=True, ecog=True, dbs=True)
    if len(data_picks) == 0:
        # Some auxiliary recordings in clinical cohorts contain no usable EEG data channels.
        # Skip these runs (fail-closed at run level) rather than crashing the dataset stage.
        return None

    # Filtering
    l_freq = cfg_get(cfg, "eeg.l_freq_hz", 0.1)
    h_freq = cfg_get(cfg, "eeg.h_freq_hz", 30.0)
    notch = cfg_get(cfg, "eeg.notch_hz", [])
    if notch:
        try:
            raw.notch_filter(freqs=list(notch), n_jobs=n_jobs, verbose="ERROR")
        except ValueError as exc:
            if "yielded no channels" in str(exc):
                return None
            raise
    raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs, verbose="ERROR")

    # Re-reference (optional)
    reref = cfg_get(cfg, "eeg.reref", "average")
    if reref == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    elif reref and reref != "none":
        raw.set_eeg_reference([str(reref)], projection=False, verbose="ERROR")

    # Resample (optional; can speed downstream extraction)
    rs = cfg_get(cfg, "eeg.resample_hz", None)
    if rs:
        raw.resample(float(rs), npad="auto", verbose="ERROR")

    sfreq = float(raw.info["sfreq"])

    # Build MNE events: all selected trials share a single event code.
    # The per-trial memory_load is stored in metadata.
    onsets_s = pd.to_numeric(trials_df["onset"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(onsets_s).all():
        raise ValueError(f"Non-finite onsets in events.tsv: {run.events_tsv}")
    onsets_samples = np.round(onsets_s * sfreq).astype(int)

    # Some datasets include repeated/duplicate onset rows; keep first occurrence
    # so event vectors and metadata remain 1:1 and auditable.
    dup_mask = pd.Series(onsets_samples).duplicated(keep="first").to_numpy()
    if dup_mask.any():
        keep = ~dup_mask
        onsets_samples = onsets_samples[keep]
        onsets_s = onsets_s[keep]
        memory_load = memory_load[keep]
        trials_df = trials_df.iloc[keep].reset_index(drop=True)
        if rt is not None:
            rt = rt[keep]
        if acc is not None:
            acc = acc[keep]

    events = np.zeros((len(onsets_samples), 3), dtype=int)
    events[:, 0] = onsets_samples
    events[:, 2] = 1  # event code

    # Assemble metadata: this is stored inside the Epochs object and round-trips.
    metadata = pd.DataFrame(
        {
            "subject": run.subject,
            "task": run.task or "",
            "run": run.run or "",
            "session": run.session or "",
            "memory_load": memory_load,
            "trial_order": trials_df["trial_order"].to_numpy(dtype=int),
            "onset_s": onsets_s,
        }
    )
    if age_years is not None:
        metadata["age"] = float(age_years)
    if rt is not None:
        metadata["rt"] = rt
    if acc is not None:
        metadata["accuracy"] = acc

    # Epoch params (locked)
    tmin = float(cfg_get(cfg, "eeg.epoch.tmin_s", -0.2))
    tmax = float(cfg_get(cfg, "eeg.epoch.tmax_s", 0.8))
    baseline = cfg_get(cfg, "eeg.epoch.baseline_s", [-0.2, 0.0])
    baseline = (float(baseline[0]), float(baseline[1]))

    reject_uV = cfg_get(cfg, "eeg.reject.eeg_uV", None)
    reject = None
    if reject_uV:
        reject = {"eeg": float(reject_uV) * 1e-6}  # uV -> V

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={"trial": 1},
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        reject=reject,
        event_repeated="drop",
        metadata=metadata,
        verbose="ERROR",
    )

    out_epochs_fif.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(out_epochs_fif.as_posix(), overwrite=True)
    return out_epochs_fif


def extract_p3_features(
    epochs_fif: Path,
    cfg: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Extract single-trial P3b amplitude and latency from a saved epochs FIF."""
    import mne

    epochs = mne.read_epochs(epochs_fif.as_posix(), preload=True, verbose="ERROR")
    md = epochs.metadata
    if md is None:
        raise ValueError(f"Epochs missing metadata: {epochs_fif}")

    # Select channel (case-insensitive + normalized-name fallback)
    chan_priority = cfg_get(cfg, "eeg.p3b.channel_priority", ["Pz", "CPz", "Cz"])

    def _norm_chan(name: str) -> str:
        return "".join(ch for ch in str(name).upper() if ch.isalnum())

    # Normalize all available names once so variants like "Pz", "PZ", "Pz-REF" can map.
    chan_map: Dict[str, str] = {}
    for ch_name in epochs.ch_names:
        chan_map.setdefault(_norm_chan(ch_name), ch_name)

    picks = None
    picked = None
    for ch in chan_priority:
        found = chan_map.get(_norm_chan(ch))
        if found is not None:
            picks = [found]
            picked = found
            break

    if picks is None:
        fallback_priority = cfg_get(
            cfg,
            "eeg.p3b.channel_fallback_priority",
            ["POz", "P1", "P2", "P3", "P4", "CP1", "CP2", "C1", "C2"],
        )
        for ch in fallback_priority:
            found = chan_map.get(_norm_chan(ch))
            if found is not None:
                picks = [found]
                picked = found
                break

    if picks is None:
        eeg_chs = epochs.copy().pick_types(eeg=True).ch_names
        if not eeg_chs:
            raise ValueError(f"No EEG channels found in epochs: {epochs_fif}")
        # Last-resort fallback avoids hard failure on non-standard montages.
        picks = [eeg_chs[0]]
        picked = eeg_chs[0]

    # Get data: shape (n_epochs, 1, n_times)
    data = epochs.get_data(picks=picks)

    times = epochs.times  # seconds relative to event
    amp_w = cfg_get(cfg, "eeg.p3b.amp_window_s", [0.35, 0.60])
    lat_w = cfg_get(cfg, "eeg.p3b.lat_window_s", [0.30, 0.70])
    amp_mask = (times >= float(amp_w[0])) & (times <= float(amp_w[1]))
    lat_mask = (times >= float(lat_w[0])) & (times <= float(lat_w[1]))

    # Amplitude: mean voltage in window
    amp = data[:, 0, :][:, amp_mask].mean(axis=1)

    # Latency: time of peak (max) within window
    lat_idx = np.argmax(data[:, 0, :][:, lat_mask], axis=1)
    lat_times = times[lat_mask][lat_idx]

    n = len(amp)

    def _col_float(name: str, default: float = np.nan) -> np.ndarray:
        if name in md.columns:
            return pd.to_numeric(md[name], errors="coerce").to_numpy(dtype=np.float32)
        return np.full(n, default, dtype=np.float32)

    def _col_int(name: str) -> np.ndarray:
        if name in md.columns:
            vals = pd.to_numeric(md[name], errors="coerce").to_numpy(dtype=np.float64)
            if np.isnan(vals).all():
                return np.arange(1, n + 1, dtype=np.int32)
            vals = np.nan_to_num(vals, nan=0.0)
            return vals.astype(np.int32)
        return np.arange(1, n + 1, dtype=np.int32)

    return {
        "p3b_amp": amp.astype(np.float32),
        "p3b_lat": lat_times.astype(np.float32),
        "p3b_channel": np.asarray([picked] * len(amp)),
        # Metadata columns we expect:
        "memory_load": _col_float("memory_load"),
        "trial_order": _col_int("trial_order"),
        "onset_s": _col_float("onset_s"),
        "age": _col_float("age"),
        "rt": _col_float("rt"),
        "accuracy": _col_float("accuracy"),
    }
