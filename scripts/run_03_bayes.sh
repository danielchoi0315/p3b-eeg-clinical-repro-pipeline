#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

FEATURES_ROOT="${1:?Usage: bash scripts/run_03_bayes.sh /path/to/features /path/to/outputs [configs/default.yaml] [seed]}"
OUT_ROOT="${2:?Usage: bash scripts/run_03_bayes.sh /path/to/features /path/to/outputs [configs/default.yaml] [seed]}"
CONFIG="${3:-configs/default.yaml}"
SEED="${4:-0}"

bash scripts/run_module.sh \
  --module 03 \
  --features_root "$FEATURES_ROOT" \
  --out_root "$OUT_ROOT" \
  --config "$CONFIG" \
  --seeds "$SEED"
