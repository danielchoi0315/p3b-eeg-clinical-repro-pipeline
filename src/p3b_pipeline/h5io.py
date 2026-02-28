"""HDF5 IO utilities using h5py.

We use **one HDF5 file per subject-run** to avoid concurrent-writer corruption.
Each file contains:
- datasets for numeric arrays (p3b_amp, p3b_lat, pdr, ...)
- lightweight attributes (subject, task, run, session, cohort, etc.)

This design is:
- append-safe (new subject file = new write)
- crash-tolerant (atomic rename)
- easy to shard across machines if needed
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np


def _h5_string_dtype():
    return h5py.string_dtype(encoding="utf-8")


def atomic_write_subject_h5(path: Path, *, arrays: Dict[str, np.ndarray], attrs: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    # Ensure deterministic dtypes to avoid downstream surprises
    cleaned: Dict[str, np.ndarray] = {}
    for k, v in arrays.items():
        if isinstance(v, np.ndarray):
            cleaned[k] = v
        else:
            cleaned[k] = np.asarray(v)

    with h5py.File(tmp, "w") as f:
        # Attributes (small JSON-serializable data)
        for k, v in attrs.items():
            if v is None:
                continue
            if isinstance(v, (dict, list)):
                f.attrs[k] = json.dumps(v)
            else:
                try:
                    f.attrs[k] = v
                except TypeError:
                    f.attrs[k] = str(v)

        # Datasets
        for name, arr in cleaned.items():
            if arr.dtype.kind in {"U", "S", "O"}:
                # Store as UTF-8 variable-length string dataset
                dt = _h5_string_dtype()
                as_obj = np.asarray(arr, dtype=object)
                if as_obj.ndim == 0:
                    as_obj = np.asarray([str(as_obj.item())], dtype=object)
                else:
                    as_obj = as_obj.astype(str).astype(object)
                f.create_dataset(name, data=as_obj, dtype=dt)
            else:
                f.create_dataset(
                    name,
                    data=arr,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    fletcher32=True,
                )

        f.flush()

    tmp.replace(path)


def read_subject_h5(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    arrays: Dict[str, np.ndarray] = {}
    attrs: Dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        # attrs
        for k, v in f.attrs.items():
            # Try to decode JSON
            if isinstance(v, (bytes, str)):
                s = v.decode("utf-8") if isinstance(v, bytes) else v
                try:
                    attrs[k] = json.loads(s)
                except Exception:
                    attrs[k] = s
            else:
                attrs[k] = v

        # datasets
        for k in f.keys():
            d = f[k][()]
            arrays[k] = d
    return arrays, attrs


def iter_subject_feature_files(features_root: Path) -> List[Path]:
    return sorted([p for p in features_root.rglob("*.h5") if p.is_file()])
