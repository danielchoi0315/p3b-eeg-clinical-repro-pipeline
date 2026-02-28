#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

FEATURES_ROOT="${1:?Usage: bash scripts/run_04_normative.sh /path/to/features /path/to/outputs [configs/default.yaml] [severity_csv] [seed]}"
OUT_ROOT="${2:?Usage: bash scripts/run_04_normative.sh /path/to/features /path/to/outputs [configs/default.yaml] [severity_csv] [seed]}"
CONFIG="${3:-configs/default.yaml}"
SEVERITY_CSV="${4:-}"
SEED="${5:-0}"

cmd=(
  bash scripts/run_module.sh
  --module 04
  --features_root "$FEATURES_ROOT"
  --out_root "$OUT_ROOT"
  --config "$CONFIG"
  --seeds "$SEED"
)
if [[ -n "$SEVERITY_CSV" ]]; then
  cmd+=(--severity_csv "$SEVERITY_CSV")
fi
"${cmd[@]}"
