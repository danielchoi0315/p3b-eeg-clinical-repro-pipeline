"""BIDS utilities.

Design principles:
- Fail-closed: if events or entities are ambiguous, we *skip* rather than guess.
- Avoid implicit magic. Everything important is logged by the callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class BIDSRun:
    """A single subject/task/run recording."""

    subject: str
    task: Optional[str]
    run: Optional[str]
    session: Optional[str]
    eeg_path: Path
    events_tsv: Optional[Path]


def build_layout(bids_root: Path):
    """Create a PyBIDS layout.

    OpenNeuro mirrors occasionally contain malformed JSON sidecar fields that can
    trigger metadata-index conflicts in pybids. We disable metadata indexing here
    because this pipeline only needs filename entities + TSV paths.
    """
    from bids import BIDSLayout  # pybids
    from bids.layout import BIDSLayoutIndexer

    return BIDSLayout(
        bids_root.as_posix(),
        validate=False,
        derivatives=False,
        indexer=BIDSLayoutIndexer(index_metadata=False),
    )


def list_subjects(bids_root: Path) -> List[str]:
    layout = build_layout(bids_root)
    subs = layout.get_subjects()
    return sorted(subs)


def load_participants(bids_root: Path) -> pd.DataFrame:
    """Load participants.tsv if present."""
    layout = build_layout(bids_root)
    files = layout.get(suffix="participants", extension=".tsv", return_type="filename")
    if not files:
        return pd.DataFrame()
    # BIDS expects a single participants.tsv at root; pick the first.
    df = pd.read_csv(files[0], sep="\t")
    return df


def participant_age(participants_df: pd.DataFrame, subject: str) -> Optional[float]:
    """Return age for subject if available.

    We accept common column names: age, Age, participant_age.
    """
    if participants_df.empty:
        return None
    col = None
    for c in ["age", "Age", "participant_age", "age_years"]:
        if c in participants_df.columns:
            col = c
            break
    if col is None:
        return None
    if "participant_id" not in participants_df.columns:
        return None
    pid = f"sub-{subject}"
    rows = participants_df.loc[participants_df["participant_id"] == pid]
    if rows.empty:
        return None
    try:
        return float(rows.iloc[0][col])
    except Exception:
        return None


def iter_eeg_runs(bids_root: Path) -> Iterable[BIDSRun]:
    """Iterate all EEG runs in a BIDS dataset.

    We use `suffix='eeg'` and let PyBIDS find matching files.
    """
    layout = build_layout(bids_root)
    eeg_files = layout.get(
        suffix="eeg",
        # Prefer primary header/container files. Excluding `.eeg` avoids duplicate
        # BrainVision data-part files when `.vhdr` exists.
        extension=[".edf", ".edf.gz", ".bdf", ".bdf.gz", ".set", ".vhdr", ".fif", ".cnt", ".gdf"],
        return_type="filename",
    )
    for f in sorted(eeg_files):
        eeg_path = Path(f)
        entities = layout.parse_file_entities(f)
        subject = entities.get("subject")
        if subject is None:
            continue
        task = entities.get("task")
        run = entities.get("run")
        session = entities.get("session")

        # Find the matching events.tsv for this recording.
        events = layout.get(
            subject=subject,
            task=task,
            run=run,
            session=session,
            datatype="eeg",
            suffix="events",
            extension=[".tsv", ".tsv.gz"],
            return_type="filename",
        )
        events_path: Optional[Path] = None
        if events:
            same_dir = [Path(p) for p in events if Path(p).parent == eeg_path.parent]
            if same_dir:
                events_path = same_dir[0]
            else:
                events_path = Path(events[0])

        if events_path is None:
            # Fallback to direct BIDS filename pairing when index queries are ambiguous.
            name = eeg_path.name
            if "_eeg" in name:
                base = name.split("_eeg", 1)[0]
                for cand in (
                    eeg_path.with_name(base + "_events.tsv"),
                    eeg_path.with_name(base + "_events.tsv.gz"),
                ):
                    if cand.exists():
                        events_path = cand
                        break

        yield BIDSRun(
            subject=str(subject),
            task=str(task) if task is not None else None,
            run=str(run) if run is not None else None,
            session=str(session) if session is not None else None,
            eeg_path=eeg_path,
            events_tsv=events_path,
        )


def load_events(events_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(events_tsv, sep="\t")
    # BIDS requires onset, duration columns.
    if "onset" not in df.columns:
        raise ValueError(f"events.tsv missing required column 'onset': {events_tsv}")
    if "duration" not in df.columns:
        # Some datasets omit duration; default to 0 for point events.
        df["duration"] = 0.0
    return df
