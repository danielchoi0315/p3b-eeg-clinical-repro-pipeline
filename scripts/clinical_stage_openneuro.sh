#!/usr/bin/env bash
set -euo pipefail

CLINICAL_BIDS_ROOT="/filesystemHcog/clinical_bids"
AUDIT_JSON=""
DATASETS="ds003523,ds005114"
OPENNEURO_BUCKET="s3://openneuro.org"

usage() {
  cat <<EOF
Usage: bash scripts/clinical_stage_openneuro.sh [options]

Options:
  --clinical_bids_root <path>   Destination root (default: /filesystemHcog/clinical_bids)
  --audit_json <path>           Output JSON with dataset hashes (required)
  --datasets <csv>              Dataset IDs (default: ds003523,ds005114)
  --openneuro_bucket <s3://...> OpenNeuro bucket (default: s3://openneuro.org)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clinical_bids_root) CLINICAL_BIDS_ROOT="$2"; shift 2 ;;
    --audit_json) AUDIT_JSON="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --openneuro_bucket) OPENNEURO_BUCKET="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$AUDIT_JSON" ]]; then
  echo "ERROR: --audit_json is required" >&2
  exit 1
fi

mkdir -p "$CLINICAL_BIDS_ROOT"
mkdir -p "$(dirname "$AUDIT_JSON")"

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

get_url() {
  local ds="$1"
  case "$ds" in
    ds003523) echo "https://github.com/OpenNeuroDatasets/ds003523.git" ;;
    ds005114) echo "https://github.com/OpenNeuroDatasets/ds005114.git" ;;
    *) echo "" ;;
  esac
}

count_event_files() {
  local root="$1"
  find "$root" -type f \( -name '*_events.tsv' -o -name '*_events.tsv.gz' \) | wc -l | xargs
}

count_eeg_regular_files() {
  local root="$1"
  find "$root" -type f | rg -N '_eeg\.(edf|bdf|set|vhdr|eeg|fif|gdf|fdt)(\.gz)?$' | wc -l | xargs || true
}

count_eeg_symlink_files() {
  local root="$1"
  find "$root" -type l | rg -N '_eeg\.(edf|bdf|set|vhdr|eeg|fif|gdf|fdt)(\.gz)?$' | wc -l | xargs || true
}

count_broken_eeg_symlinks() {
  local root="$1"
  python - "$root" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
pat = re.compile(r"_eeg\.(edf|bdf|set|vhdr|eeg|fif|gdf|fdt)(\.gz)?$", re.IGNORECASE)
n = 0
for p in root.rglob("*"):
    try:
        if p.is_symlink() and pat.search(p.name) and not p.exists():
            n += 1
    except Exception:
        continue
print(n)
PY
}

dataset_ready() {
  local root="$1"
  [[ -f "$root/participants.tsv" ]] || return 1
  local n_events n_eeg
  n_events="$(count_event_files "$root")"
  n_eeg="$(count_eeg_regular_files "$root")"
  [[ "$n_events" -gt 0 ]] || return 1
  [[ "$n_eeg" -gt 0 ]] || return 1
  return 0
}

json_escape() {
  python - "$1" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1]))
PY
}

IFS=',' read -r -a ds_arr <<< "$DATASETS"

declare -a dataset_json_rows=()

tmp_json="${AUDIT_JSON}.tmp"
: > "$tmp_json"

for raw in "${ds_arr[@]}"; do
  ds="$(echo "$raw" | xargs)"
  [[ -z "$ds" ]] && continue
  url="$(get_url "$ds")"
  if [[ -z "$url" ]]; then
    echo "ERROR: Unsupported dataset id: $ds" >&2
    exit 1
  fi

  dest="$CLINICAL_BIDS_ROOT/$ds"
  method="existing"
  fallback_used="false"
  had_existing="false"
  if [[ -d "$dest" ]]; then
    had_existing="true"
  fi

  # If existing data is incomplete (e.g., datalad clone with broken annex links),
  # re-stage cleanly.
  if [[ -d "$dest" ]] && ! dataset_ready "$dest"; then
    echo "[clinical_stage] existing dataset incomplete, clearing: $dest"
    rm -rf "$dest"
  fi

  if [[ -d "$dest" ]] && dataset_ready "$dest"; then
    echo "[clinical_stage] dataset ready: $ds ($dest)"
  else
    staged_ok="false"

    if has_cmd datalad; then
      echo "[clinical_stage] trying datalad clone/get: $ds -> $dest"
      if datalad clone "$url" "$dest"; then
        (
          cd "$dest"
          # Do not fail fast here; validate readiness explicitly below.
          datalad get -n . || true
          datalad get -r . || true
        )
        if dataset_ready "$dest"; then
          staged_ok="true"
          method="datalad"
        fi
      fi
    fi

    if [[ "$staged_ok" != "true" ]]; then
      fallback_used="true"
      method="aws_s3_sync"
      echo "[clinical_stage] falling back to aws s3 sync: $ds"
      rm -rf "$dest"
      mkdir -p "$dest"
      aws s3 sync "${OPENNEURO_BUCKET}/${ds}/" "$dest/" --no-sign-request --only-show-errors
      if dataset_ready "$dest"; then
        staged_ok="true"
      fi
    fi

    if [[ "$staged_ok" != "true" ]]; then
      echo "ERROR: failed to stage usable dataset payload for $ds at $dest" >&2
      exit 1
    fi
  fi

  git_head="$(git -C "$dest" rev-parse HEAD 2>/dev/null || true)"
  remote_head="$(git ls-remote "$url" HEAD 2>/dev/null | awk '{print $1}' | head -n 1 || true)"
  if [[ -z "$git_head" ]]; then
    if [[ -n "$remote_head" ]]; then
      git_head="$remote_head"
    else
      git_head="<unavailable>"
    fi
  fi
  has_participants="false"
  [[ -f "$dest/participants.tsv" ]] && has_participants="true"

  n_events="$(count_event_files "$dest")"
  n_eeg="$(count_eeg_regular_files "$dest")"
  n_eeg_links="$(count_eeg_symlink_files "$dest")"
  n_broken_links="$(count_broken_eeg_symlinks "$dest")"

  dataset_json_rows+=(
    "{\
\"dataset_id\": $(json_escape "$ds"), \
\"git_url\": $(json_escape "$url"), \
\"path\": $(json_escape "$dest"), \
\"checked_out_commit\": $(json_escape "$git_head"), \
\"remote_head_commit\": $(json_escape "${remote_head:-}"), \
\"staging_method\": $(json_escape "$method"), \
\"fallback_used\": ${fallback_used}, \
\"had_existing_dir\": ${had_existing}, \
\"has_participants_tsv\": ${has_participants}, \
\"n_event_files\": ${n_events}, \
\"n_eeg_files\": ${n_eeg}, \
\"n_eeg_symlink_files\": ${n_eeg_links}, \
\"n_broken_eeg_symlinks\": ${n_broken_links}\
}"
  )
done

{
  echo "{"
  echo "  \"timestamp_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"," 
  echo "  \"clinical_bids_root\": \"$CLINICAL_BIDS_ROOT\"," 
  echo "  \"openneuro_bucket\": \"$OPENNEURO_BUCKET\"," 
  echo "  \"datasets\": ["
  for i in "${!dataset_json_rows[@]}"; do
    row="${dataset_json_rows[$i]}"
    if [[ "$i" -lt $((${#dataset_json_rows[@]} - 1)) ]]; then
      echo "    $row,"
    else
      echo "    $row"
    fi
  done
  echo "  ]"
  echo "}"
} > "$tmp_json"

mv "$tmp_json" "$AUDIT_JSON"
echo "[clinical_stage] wrote hashes: $AUDIT_JSON"
