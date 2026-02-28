#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT=""
DATASETS="ds003523,ds005114"
WALL_HOURS="10"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
GPU_PARALLEL_PROCS="${GPU_PARALLEL_PROCS:-12}"
CPU_WORKERS="${CPU_WORKERS:-32}"
MODEL_SEEDS="${MODEL_SEEDS:-0-199}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --wall_hours) WALL_HOURS="$2"; shift 2 ;;
    --max_attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --gpu_parallel_procs) GPU_PARALLEL_PROCS="$2"; shift 2 ;;
    --cpu_workers) CPU_WORKERS="$2"; shift 2 ;;
    --model_seeds) MODEL_SEEDS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$OUT_ROOT" ]]; then
  OUT_ROOT="/filesystemHcog/runs/$(date +%Y%m%d_%H%M%S)_CLINICAL_OVERNIGHT10H"
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

attempt=1
while [[ "$attempt" -le "$MAX_ATTEMPTS" ]]; do
  echo "[CLINICAL_OVERNIGHT10H] attempt ${attempt}/${MAX_ATTEMPTS}"

  RESUME_FLAG=""
  if [[ "$attempt" -gt 1 ]]; then
    RESUME_FLAG="--resume"
  fi

  set +e
  python scripts/clinical_overnight10h_runner.py \
    --data_root /filesystemHcog/openneuro \
    --out_root "$OUT_ROOT" \
    --clinical_bids_root /filesystemHcog/clinical_bids \
    --clinical_severity_csv /filesystemHcog/clinical_bids/clinical_severity.csv \
    --features_root_healthy /filesystemHcog/features_cache_FIX2_20260222_061927 \
    --config configs/default.yaml \
    --lawc_event_map configs/lawc_event_map.yaml \
    --clinical_event_map configs/clinical_event_map_autogen.yaml \
    --datasets "$DATASETS" \
    --model_seeds "$MODEL_SEEDS" \
    --wall_hours "$WALL_HOURS" \
    --rt_n_perm 20000 \
    --gpu_parallel_procs "$GPU_PARALLEL_PROCS" \
    --cpu_workers "$CPU_WORKERS" \
    ${RESUME_FLAG}
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    report="$OUT_ROOT/AUDIT/CLINICAL_OVERNIGHT10H_REPORT.md"
    echo "[CLINICAL_OVERNIGHT10H] success"
    echo "OUT_ROOT=$OUT_ROOT"
    echo "REPORT=$report"
    exit 0
  fi

  mkdir -p "$OUT_ROOT/AUDIT"
  {
    echo ""
    echo "==== attempt ${attempt} failed $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc} ===="
    if [[ -f "$OUT_ROOT/AUDIT/run_status.json" ]]; then
      cat "$OUT_ROOT/AUDIT/run_status.json"
    fi
    for lf in "$OUT_ROOT"/AUDIT/*.log; do
      [[ -f "$lf" ]] || continue
      echo ""
      echo "---- tail $lf ----"
      tail -n 160 "$lf" || true
    done
  } >> "$OUT_ROOT/AUDIT/FAIL_CONTEXT.txt"

  # Targeted auto-fixes (non-scientific):
  log_blob="$(tail -n 500 "$OUT_ROOT"/AUDIT/*.log 2>/dev/null || true)"
  if echo "$log_blob" | rg -qi "VLEN strings|embedded NULL|HDF5|Unicode"; then
    export HDF5_USE_FILE_LOCKING=FALSE
  fi
  if echo "$log_blob" | rg -qi "ImportError|No module named|NameError"; then
    find . -name '*.py' -print0 | xargs -0 python -m py_compile || true
  fi
  if echo "$log_blob" | rg -qi "IndexError|out of bounds"; then
    GPU_PARALLEL_PROCS=$(( GPU_PARALLEL_PROCS > 4 ? GPU_PARALLEL_PROCS / 2 : GPU_PARALLEL_PROCS ))
    CPU_WORKERS=$(( CPU_WORKERS > 8 ? CPU_WORKERS / 2 : CPU_WORKERS ))
  fi

  attempt=$((attempt + 1))
done

echo "[CLINICAL_OVERNIGHT10H] failed after ${MAX_ATTEMPTS} attempts"
exit 1

