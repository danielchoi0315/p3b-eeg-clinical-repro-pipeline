#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-/filesystemHcog/runs/$(date +%Y%m%d_%H%M%S)_NN_SOLID_1_2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

attempt=1
while [[ "$attempt" -le "$MAX_ATTEMPTS" ]]; do
  echo "[NN_SOLID_1_2] attempt ${attempt}/${MAX_ATTEMPTS}"

  RESUME_FLAG=""
  if [[ "$attempt" -gt 1 ]]; then
    RESUME_FLAG="--resume"
  fi

  set +e
  python scripts/nn_solid_1_2_runner.py \
    --data_root /filesystemHcog/openneuro \
    --features_root_healthy /filesystemHcog/features_cache_FIX2_20260222_061927 \
    --features_root_mechanism /filesystemHcog/features_cache_FIX1_20260222_060109 \
    --mechanism_dataset ds003838 \
    --sternberg_datasets ds005095,ds003655,ds004117 \
    --event_map configs/lawc_event_map.yaml \
    --mechanism_event_map configs/mechanism_event_map.yaml \
    --config configs/default.yaml \
    --clinical_bids_root /filesystemHcog/clinical_bids \
    --clinical_severity_csv /filesystemHcog/clinical_bids/clinical_severity.csv \
    --wall_hours 10 \
    --lawc_n_perm 50000 \
    --mechanism_n_perm 2000 \
    --mechanism_seeds 0-49 \
    --gpu_parallel_procs 6 \
    --cpu_workers 32 \
    --out_root "$OUT_ROOT" \
    ${RESUME_FLAG}
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    REPORT_PATH="$OUT_ROOT/AUDIT/NN_SOLID_1_2_REPORT.md"
    echo "[NN_SOLID_1_2] success"
    echo "OUT_ROOT=$OUT_ROOT"
    echo "REPORT=$REPORT_PATH"
    exit 0
  fi

  mkdir -p "$OUT_ROOT/AUDIT"
  {
    echo "\n==== attempt ${attempt} failed at $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc} ===="
    if [[ -f "$OUT_ROOT/AUDIT/run_status.json" ]]; then
      cat "$OUT_ROOT/AUDIT/run_status.json"
    fi
    for lf in "$OUT_ROOT"/AUDIT/*.log; do
      [[ -f "$lf" ]] || continue
      echo "\n---- tail $lf ----"
      tail -n 120 "$lf" || true
    done
  } >> "$OUT_ROOT/AUDIT/FAIL_CONTEXT.txt"

  if [[ "$attempt" -lt "$MAX_ATTEMPTS" ]]; then
    # Minimal non-science auto-fix hook: keep permissions/syspath stable across retries.
    chmod +x scripts/*.sh || true
    chmod +x scripts/*.py || true
  fi

  attempt=$((attempt + 1))
done

echo "[NN_SOLID_1_2] failed after ${MAX_ATTEMPTS} attempts"
exit 1
