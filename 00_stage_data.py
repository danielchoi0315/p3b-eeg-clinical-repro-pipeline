#!/usr/bin/env python3
"""Stage OpenNeuro datasets with DataLad and fail-fast BIDS validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from common.data_staging import (
    load_dataset_config,
    stage_datasets,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/datasets.yaml"), help="Dataset list/hash config")
    ap.add_argument("--openneuro_root", type=Path, default=None, help="Override OpenNeuro root path")
    ap.add_argument(
        "--manifest_out",
        type=Path,
        default=None,
        help="Machine-readable staging manifest path (default: <openneuro_root>/STAGING_MANIFEST.json)",
    )
    ap.add_argument(
        "--load_columns",
        type=str,
        default="memory_load,set_size,setsize,load,n_items",
        help="Comma-separated event load columns accepted by fail-fast validator",
    )
    ap.add_argument(
        "--load_mapping_columns",
        type=str,
        default="trial_type,value,condition,stim_file,label",
        help="Comma-separated fallback columns that can encode load via dataset-specific mapping",
    )
    ap.add_argument(
        "--pupil_columns",
        type=str,
        default="pupil,pupil_size,pupil_diameter,pupil_area,diameter,diameter_3d",
        help="Comma-separated pupil columns accepted in eyetrack TSV",
    )
    ap.add_argument(
        "--allow_missing_mechanism_pupil",
        action="store_true",
        help="Disable strict ds003838 pupil-file requirement (not recommended)",
    )
    return ap.parse_args()


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def main() -> None:
    args = parse_args()

    openneuro_root, datasets = load_dataset_config(args.config)

    if args.openneuro_root is not None:
        openneuro_root = args.openneuro_root

    manifest_out = args.manifest_out or (openneuro_root / "STAGING_MANIFEST.json")

    payload = stage_datasets(
        openneuro_root=openneuro_root,
        datasets=datasets,
        manifest_out=manifest_out,
        load_columns_priority=_split_csv(args.load_columns),
        load_mapping_columns=_split_csv(args.load_mapping_columns),
        pupil_columns_priority=_split_csv(args.pupil_columns),
        require_mechanism_pupil=not args.allow_missing_mechanism_pupil,
    )

    print(f"Staged {len(payload['datasets'])} dataset(s).")
    print(f"Manifest: {manifest_out}")


if __name__ == "__main__":
    main()
