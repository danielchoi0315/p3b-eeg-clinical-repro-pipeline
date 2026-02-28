#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-/filesystemHcog/runs/$(date +%Y%m%d_%H%M%S)_OVERNIGHT_REVIEW_FIX2}"
FEATURES_ROOT="${FEATURES_ROOT:-/filesystemHcog/features_cache_FIX2_20260222_061927}"
DATA_ROOT="${DATA_ROOT:-/filesystemHcog/openneuro}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"
MAX_ATTEMPTS=6

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "$OUT_ROOT/AUDIT/logs"

attempt=0
runner_rc=0

while true; do
  attempt=$((attempt + 1))
  if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
    echo "[AUTORUN] Failed after $MAX_ATTEMPTS attempts."
    echo "Failure context saved: $OUT_ROOT/AUDIT/FAIL_CONTEXT.txt"
    exit 1
  fi

  echo "[AUTORUN] Attempt $attempt/$MAX_ATTEMPTS"
  RUN_LOG="$OUT_ROOT/AUDIT/logs/runner_attempt_${attempt}.log"
  mkdir -p "$(dirname "$RUN_LOG")"

  set +e
  if [ "$attempt" -eq 1 ]; then
    "$PYTHON" "$REPO_ROOT/scripts/overnight_reviewerproof_runner.py" \
      --features_root "$FEATURES_ROOT" \
      --data_root "$DATA_ROOT" \
      --datasets ds005095,ds003655,ds004117 \
      --event_map "$REPO_ROOT/configs/lawc_event_map.yaml" \
      --config "$REPO_ROOT/configs/default.yaml" \
      --wall_hours 8 \
      --lawc_n_perm 20000 \
      --rtlink_n_perm 20000 \
      --seeds 0-99 \
      --gpu_parallel_procs 6 \
      --cpu_workers 32 \
      --out_root "$OUT_ROOT" \
      --resume | tee "$RUN_LOG"
    runner_rc=${PIPESTATUS[0]}
  else
    "$PYTHON" "$REPO_ROOT/scripts/overnight_reviewerproof_runner.py" \
      --features_root "$FEATURES_ROOT" \
      --data_root "$DATA_ROOT" \
      --datasets ds005095,ds003655,ds004117 \
      --event_map "$REPO_ROOT/configs/lawc_event_map.yaml" \
      --config "$REPO_ROOT/configs/default.yaml" \
      --wall_hours 8 \
      --lawc_n_perm 20000 \
      --rtlink_n_perm 20000 \
      --seeds 0-99 \
      --gpu_parallel_procs 6 \
      --cpu_workers 32 \
      --out_root "$OUT_ROOT" \
      --resume | tee "$RUN_LOG"
    runner_rc=${PIPESTATUS[0]}
  fi
  set -e

  if [ "$runner_rc" -eq 0 ]; then
    echo "[AUTORUN] PASS"
    echo "[AUTORUN] Report: $OUT_ROOT/AUDIT/OVERNIGHT_REVIEW_REPORT.md"
    exit 0
  fi

  fail_log="$(find "$OUT_ROOT/AUDIT/logs" -name '*.log' -type f -printf '%T@ %p\n' | sort -k1,1nr | head -n 1 | awk '{print $2}')"
  {
    echo "=== Attempt ${attempt} FAIL (${runner_rc}) ==="
    echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Failing log: ${fail_log:-$RUN_LOG}"
    echo "----- tail -n 200 (${fail_log:-$RUN_LOG}) -----"
    if [ -n "${fail_log}" ] && [ -f "${fail_log}" ]; then
      tail -n 200 "$fail_log"
    else
      echo "(no log found)"
    fi
    echo "----- diagnostic: torch / cuda ----"
    $PYTHON - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available())
PY
    nvidia-smi -L
    echo "-----------------------------------"
  } >> "$OUT_ROOT/AUDIT/FAIL_CONTEXT.txt"

  echo "[AUTORUN] Context appended to $OUT_ROOT/AUDIT/FAIL_CONTEXT.txt"
  echo "[AUTORUN] Re-running with --resume after short backoff..."
  sleep 3

done
