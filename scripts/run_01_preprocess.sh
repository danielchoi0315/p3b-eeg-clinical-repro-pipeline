#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

BIDS_ROOT="${1:?Usage: bash scripts/run_01_preprocess.sh /path/to/BIDS /path/to/derivatives [configs/default.yaml]}"
DERIV_ROOT="${2:?Usage: bash scripts/run_01_preprocess.sh /path/to/BIDS /path/to/derivatives [configs/default.yaml]}"
CONFIG="${3:-configs/default.yaml}"

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
python 01_preprocess_CPU.py --bids_root "$BIDS_ROOT" --deriv_root "$DERIV_ROOT" --config "$CONFIG"
