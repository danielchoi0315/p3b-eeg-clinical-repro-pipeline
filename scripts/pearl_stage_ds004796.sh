#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/filesystemHcog/openneuro"
DATASET_ID="ds004796"
REPO_URL="https://github.com/OpenNeuroDatasets/ds004796.git"
OPENNEURO_BUCKET="s3://openneuro.org"
OUT_MANIFEST=""

usage() {
  cat <<EOF
Usage: bash scripts/pearl_stage_ds004796.sh [options]

Options:
  --data_root <path>      Destination root (default: /filesystemHcog/openneuro)
  --dataset_id <id>       Dataset id (default: ds004796)
  --repo_url <url>        Dataset git URL
  --openneuro_bucket <s3> OpenNeuro S3 bucket prefix (default: s3://openneuro.org)
  --out_manifest <path>   Output manifest JSON (required)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --dataset_id) DATASET_ID="$2"; shift 2 ;;
    --repo_url) REPO_URL="$2"; shift 2 ;;
    --openneuro_bucket) OPENNEURO_BUCKET="$2"; shift 2 ;;
    --out_manifest) OUT_MANIFEST="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${OUT_MANIFEST}" ]]; then
  echo "ERROR: --out_manifest is required" >&2
  exit 1
fi

DATASET_ROOT="${DATA_ROOT}/${DATASET_ID}"
mkdir -p "${DATA_ROOT}"
mkdir -p "$(dirname "${OUT_MANIFEST}")"

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

count_files() {
  local root="$1"
  local pat="$2"
  find "${root}" -type f -name "${pat}" | wc -l | xargs
}

count_matching_regex() {
  local root="$1"
  local regex="$2"
  find "${root}" -type f | rg -N "${regex}" | wc -l | xargs || true
}

dataset_ready() {
  local root="$1"
  [[ -f "${root}/participants.tsv" ]] || return 1
  local n_events n_eeg
  n_events="$(count_files "${root}" "*task-sternberg*_events.tsv")"
  n_eeg="$(count_matching_regex "${root}" 'task-sternberg.*_eeg\.(vhdr|edf|bdf|set|fif|cnt|gdf)(\.gz)?$')"
  [[ "${n_events}" -gt 0 ]] || return 1
  [[ "${n_eeg}" -gt 0 ]] || return 1
  return 0
}

discover_official_s3_prefix() {
  local ds="$1"
  local bucket="$2"
  if aws s3 ls "${bucket}/${ds}/" --no-sign-request >/dev/null 2>&1; then
    echo "${bucket}/${ds}/"
    return 0
  fi
  return 1
}

stage_method="existing"
fallback_used=false
datalad_error=""
s3_error=""
s3_prefix=""

if dataset_ready "${DATASET_ROOT}"; then
  stage_method="existing"
else
  if has_cmd datalad; then
    stage_method="datalad"
    if [[ -d "${DATASET_ROOT}/.git" ]]; then
      (
        cd "${DATASET_ROOT}"
        datalad get participants.tsv participants.json dataset_description.json README CHANGES || true
        datalad get -r sub-*/eeg/*task-sternberg* || true
        datalad get -r sub-*/eeg/*task-rest* || true
        datalad get -r sourcedata/sub-*/logfiles/*task-sternberg*events*.txt || true
      ) || true
    else
      if ! datalad clone "${REPO_URL}" "${DATASET_ROOT}" 2>"${OUT_MANIFEST}.datalad.stderr"; then
        datalad_error="$(cat "${OUT_MANIFEST}.datalad.stderr" 2>/dev/null || true)"
      else
        (
          cd "${DATASET_ROOT}"
          datalad get participants.tsv participants.json dataset_description.json README CHANGES || true
          datalad get -r sub-*/eeg/*task-sternberg* || true
          datalad get -r sub-*/eeg/*task-rest* || true
          datalad get -r sourcedata/sub-*/logfiles/*task-sternberg*events*.txt || true
        ) || true
      fi
    fi
  fi

  if ! dataset_ready "${DATASET_ROOT}"; then
    fallback_used=true
    stage_method="aws_s3_sync"
    if s3_prefix="$(discover_official_s3_prefix "${DATASET_ID}" "${OPENNEURO_BUCKET}")"; then
      mkdir -p "${DATASET_ROOT}"
      aws s3 sync "${s3_prefix}" "${DATASET_ROOT}/" \
        --no-sign-request \
        --only-show-errors \
        --exclude "*" \
        --include "participants.tsv" \
        --include "participants.json" \
        --include "dataset_description.json" \
        --include "README" \
        --include "CHANGES" \
        --include "sub-*/eeg/*task-sternberg*_events.tsv" \
        --include "sub-*/eeg/*task-sternberg*_events.tsv.gz" \
        --include "sub-*/eeg/*task-sternberg*_events.json" \
        --include "sub-*/eeg/*task-sternberg*_eeg.vhdr" \
        --include "sub-*/eeg/*task-sternberg*_eeg.vmrk" \
        --include "sub-*/eeg/*task-sternberg*_eeg.eeg" \
        --include "sub-*/eeg/*task-sternberg*_eeg.edf" \
        --include "sub-*/eeg/*task-sternberg*_eeg.bdf" \
        --include "sub-*/eeg/*task-sternberg*_eeg.set" \
        --include "sub-*/eeg/*task-sternberg*_eeg.fdt" \
        --include "sub-*/eeg/*task-rest*_events.tsv" \
        --include "sub-*/eeg/*task-rest*_events.tsv.gz" \
        --include "sub-*/eeg/*task-rest*_events.json" \
        --include "sub-*/eeg/*task-rest*_eeg.vhdr" \
        --include "sub-*/eeg/*task-rest*_eeg.vmrk" \
        --include "sub-*/eeg/*task-rest*_eeg.eeg" \
        --include "sub-*/eeg/*task-rest*_eeg.edf" \
        --include "sub-*/eeg/*task-rest*_eeg.bdf" \
        --include "sub-*/eeg/*task-rest*_eeg.set" \
        --include "sub-*/eeg/*task-rest*_eeg.fdt" \
        --include "sourcedata/sub-*/logfiles/*task-sternberg*events*.txt" \
        --include "sourcedata/sub-*/logfiles/*task-sternberg*events*.csv" \
        2>"${OUT_MANIFEST}.aws.stderr" || true
      s3_error="$(cat "${OUT_MANIFEST}.aws.stderr" 2>/dev/null || true)"
    else
      s3_error="official OpenNeuro S3 prefix not discoverable for ${DATASET_ID}"
    fi
  fi
fi

if ! dataset_ready "${DATASET_ROOT}"; then
  echo "ERROR: staging failed for ${DATASET_ID}; dataset not ready at ${DATASET_ROOT}" >&2
  python - "${OUT_MANIFEST}" "${DATASET_ROOT}" "${REPO_URL}" "${stage_method}" "${fallback_used}" "${datalad_error}" "${s3_error}" "${s3_prefix}" <<'PY'
import json, os, sys
from pathlib import Path

out = Path(sys.argv[1])
payload = {
    "dataset_id": "ds004796",
    "dataset_root": sys.argv[2],
    "repo_url": sys.argv[3],
    "status": "FAIL",
    "staging_method": sys.argv[4],
    "fallback_used": sys.argv[5].lower() == "true",
    "datalad_error": sys.argv[6],
    "s3_error": sys.argv[7],
    "s3_prefix": sys.argv[8],
}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
  exit 1
fi

local_head="$(git -C "${DATASET_ROOT}" rev-parse HEAD 2>/dev/null || true)"
remote_head="$(git ls-remote "${REPO_URL}" HEAD 2>/dev/null | awk '{print $1}' | head -n 1 || true)"
if [[ -z "${local_head}" ]]; then
  local_head="<unavailable>"
fi
if [[ -z "${remote_head}" ]]; then
  remote_head="<unavailable>"
fi

n_events_sternberg="$(count_files "${DATASET_ROOT}" "*task-sternberg*_events.tsv")"
n_events_rest="$(count_files "${DATASET_ROOT}" "*task-rest*_events.tsv")"
n_eeg_sternberg="$(count_matching_regex "${DATASET_ROOT}" 'task-sternberg.*_eeg\.(vhdr|edf|bdf|set|fif|cnt|gdf)(\.gz)?$')"
n_eeg_rest="$(count_matching_regex "${DATASET_ROOT}" 'task-rest.*_eeg\.(vhdr|edf|bdf|set|fif|cnt|gdf)(\.gz)?$')"
n_logs_sternberg="$(count_matching_regex "${DATASET_ROOT}" 'task-sternberg.*events.*\.(txt|csv)$')"
disk_usage="$(du -sh "${DATASET_ROOT}" | awk '{print $1}')"

python - "${OUT_MANIFEST}" "${DATASET_ID}" "${DATASET_ROOT}" "${REPO_URL}" "${stage_method}" "${fallback_used}" "${local_head}" "${remote_head}" "${disk_usage}" "${n_events_sternberg}" "${n_events_rest}" "${n_eeg_sternberg}" "${n_eeg_rest}" "${n_logs_sternberg}" "${datalad_error}" "${s3_error}" "${s3_prefix}" <<'PY'
import json
import sys
from datetime import datetime, timezone

out = sys.argv[1]
payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_id": sys.argv[2],
    "dataset_root": sys.argv[3],
    "repo_url": sys.argv[4],
    "staging_method": sys.argv[5],
    "fallback_used": sys.argv[6].lower() == "true",
    "checked_out_commit": sys.argv[7],
    "remote_head_commit": sys.argv[8],
    "disk_usage": sys.argv[9],
    "counts": {
        "sternberg_events_tsv": int(sys.argv[10]),
        "rest_events_tsv": int(sys.argv[11]),
        "sternberg_eeg_headers": int(sys.argv[12]),
        "rest_eeg_headers": int(sys.argv[13]),
        "sourcedata_sternberg_logs": int(sys.argv[14]),
    },
    "datalad_error": sys.argv[15],
    "s3_error": sys.argv[16],
    "s3_prefix": sys.argv[17],
    "status": "PASS",
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY

echo "[pearl_stage_ds004796] wrote manifest: ${OUT_MANIFEST}"
