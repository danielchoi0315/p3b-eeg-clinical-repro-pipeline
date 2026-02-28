#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/datasets.yaml"
OPENNEURO_ROOT="/filesystemHcog/openneuro"
MANIFEST_OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --openneuro_root) OPENNEURO_ROOT="$2"; shift 2 ;;
    --manifest_out) MANIFEST_OUT="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: bash 00_stage_data.sh [--config configs/datasets.yaml] [--openneuro_root /filesystemHcog/openneuro] [--manifest_out /path/to/manifest.json]"
      exit 1
      ;;
  esac
done

if [[ -n "$MANIFEST_OUT" ]]; then
  python 00_stage_data.py --config "$CONFIG" --openneuro_root "$OPENNEURO_ROOT" --manifest_out "$MANIFEST_OUT"
else
  python 00_stage_data.py --config "$CONFIG" --openneuro_root "$OPENNEURO_ROOT"
fi
