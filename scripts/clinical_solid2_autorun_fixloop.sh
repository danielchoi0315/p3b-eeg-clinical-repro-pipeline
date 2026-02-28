#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-/filesystemHcog/runs/$(date +%Y%m%d_%H%M%S)_CLINICAL_SOLID2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

attempt=1
while [[ "$attempt" -le "$MAX_ATTEMPTS" ]]; do
  echo "[CLINICAL_SOLID2] attempt ${attempt}/${MAX_ATTEMPTS}"

  RESUME_FLAG=""
  if [[ "$attempt" -gt 1 ]]; then
    RESUME_FLAG="--resume"
  fi

  set +e
  python scripts/clinical_solid2_runner.py \
    --data_root /filesystemHcog/openneuro \
    --out_root "$OUT_ROOT" \
    --clinical_bids_root /filesystemHcog/clinical_bids \
    --clinical_severity_csv /filesystemHcog/clinical_bids/clinical_severity.csv \
    --features_root_healthy /filesystemHcog/features_cache_FIX2_20260222_061927 \
    --config configs/default.yaml \
    --lawc_event_map configs/lawc_event_map.yaml \
    --clinical_event_map configs/clinical_event_map.yaml \
    --wall_hours 10 \
    --rt_n_perm 20000 \
    --gpu_parallel_procs 6 \
    --cpu_workers 32 \
    ${RESUME_FLAG}
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    report="$OUT_ROOT/AUDIT/CLINICAL_SOLID2_REPORT.md"
    echo "[CLINICAL_SOLID2] success"
    echo "OUT_ROOT=$OUT_ROOT"
    echo "REPORT=$report"
    exit 0
  fi

  mkdir -p "$OUT_ROOT/AUDIT"
  {
    echo "\n==== attempt ${attempt} failed $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc} ===="
    if [[ -f "$OUT_ROOT/AUDIT/run_status.json" ]]; then
      cat "$OUT_ROOT/AUDIT/run_status.json"
    fi
    for lf in "$OUT_ROOT"/AUDIT/*.log; do
      [[ -f "$lf" ]] || continue
      echo "\n---- tail $lf ----"
      tail -n 120 "$lf" || true
    done
  } >> "$OUT_ROOT/AUDIT/FAIL_CONTEXT.txt"

  # Minimal auto-fix hook: ensure scripts are executable and recompile.
  chmod +x scripts/*.sh scripts/*.py || true
  find . -name '*.py' -print0 | xargs -0 python -m py_compile || true

  attempt=$((attempt + 1))
done

echo "[CLINICAL_SOLID2] failed after ${MAX_ATTEMPTS} attempts"
exit 1
