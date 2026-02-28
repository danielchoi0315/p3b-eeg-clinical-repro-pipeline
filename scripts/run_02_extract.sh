#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

BIDS_ROOT="${1:?Usage: bash scripts/run_02_extract.sh /path/to/BIDS /path/to/derivatives /path/to/features [configs/default.yaml] [cohort] [dataset_id]}"
DERIV_ROOT="${2:?Usage: bash scripts/run_02_extract.sh /path/to/BIDS /path/to/derivatives /path/to/features [configs/default.yaml] [cohort] [dataset_id]}"
FEATURES_ROOT="${3:?Usage: bash scripts/run_02_extract.sh /path/to/BIDS /path/to/derivatives /path/to/features [configs/default.yaml] [cohort] [dataset_id]}"
CONFIG="${4:-configs/default.yaml}"
COHORT="${5:-healthy}"
DATASET_ID="${6:-$(basename "$BIDS_ROOT")}"

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
python 02_extract_features_CPU.py \
  --bids_root "$BIDS_ROOT" \
  --deriv_root "$DERIV_ROOT" \
  --features_root "$FEATURES_ROOT" \
  --config "$CONFIG" \
  --cohort "$COHORT" \
  --dataset_id "$DATASET_ID"
