#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

MAX_ATTEMPTS=6
WALL_HOURS=10
OUT_ROOT=""
GPU_PARALLEL_PROCS="6"
CPU_WORKERS="32"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out_root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --wall_hours)
      WALL_HOURS="$2"
      shift 2
      ;;
    --gpu_parallel_procs)
      GPU_PARALLEL_PROCS="$2"
      shift 2
      ;;
    --cpu_workers)
      CPU_WORKERS="$2"
      shift 2
      ;;
    --max_attempts)
      MAX_ATTEMPTS="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$OUT_ROOT" ]]; then
  OUT_ROOT="/filesystemHcog/runs/$(date +%Y%m%d_%H%M%S)_NN_FINAL_PUSH_V2"
fi

mkdir -p "$OUT_ROOT/AUDIT" "$OUT_ROOT/OUTZIP"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

attempt=1
resume_flag=""
rc=1

while [[ "$attempt" -le "$MAX_ATTEMPTS" ]]; do
  echo "[NN_FINAL_PUSH_V2] attempt ${attempt}/${MAX_ATTEMPTS} out_root=${OUT_ROOT}"

  set +e
  cmd=(python scripts/nn_final_push_v2_runner.py
    --out_root "$OUT_ROOT"
    --wall_hours "$WALL_HOURS"
    --gpu_parallel_procs "$GPU_PARALLEL_PROCS"
    --cpu_workers "$CPU_WORKERS"
  )
  if [[ -n "$resume_flag" ]]; then
    cmd+=("$resume_flag")
  fi
  cmd+=("${EXTRA_ARGS[@]}")
  "${cmd[@]}"
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    echo "[NN_FINAL_PUSH_V2] success"
    break
  fi

  fail_ctx="$OUT_ROOT/AUDIT/FAIL_CONTEXT.txt"
  {
    echo ""
    echo "==== attempt=${attempt} ts=$(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc} ===="
    if [[ -f "$OUT_ROOT/AUDIT/run_status.json" ]]; then
      cat "$OUT_ROOT/AUDIT/run_status.json"
    fi
    newest_log="$(ls -1t "$OUT_ROOT"/AUDIT/*.log 2>/dev/null | head -n 1 || true)"
    if [[ -n "$newest_log" && -f "$newest_log" ]]; then
      echo ""
      echo "---- tail ${newest_log} ----"
      tail -n 240 "$newest_log" || true
    fi
  } >> "$fail_ctx"

  log_blob="$(tail -n 800 "$OUT_ROOT"/AUDIT/*.log 2>/dev/null || true)"

  # Minimal runtime-only fixes; never alter science locks.
  if echo "$log_blob" | rg -qi "HDF5|file locking|embedded NULL"; then
    export HDF5_USE_FILE_LOCKING=FALSE
  fi

  if echo "$log_blob" | rg -qi "No module named|ImportError|NameError|SyntaxError"; then
    find . -name '*.py' -print0 | xargs -0 python -m py_compile || true
  fi

  if echo "$log_blob" | rg -qi "Killed|out of memory|CUDA out of memory|Resource temporarily unavailable"; then
    if [[ "$GPU_PARALLEL_PROCS" -gt 2 ]]; then
      GPU_PARALLEL_PROCS=$((GPU_PARALLEL_PROCS / 2))
    fi
    if [[ "$CPU_WORKERS" -gt 8 ]]; then
      CPU_WORKERS=$((CPU_WORKERS / 2))
    fi
  fi

  resume_flag="--resume"
  attempt=$((attempt + 1))
done

report="$OUT_ROOT/AUDIT/NN_FINAL_PUSH_V2_REPORT.md"
zip_path="$OUT_ROOT/OUTZIP/NN_SUBMISSION_PACKET_V2.zip"

if [[ "$rc" -ne 0 ]]; then
  echo "[NN_FINAL_PUSH_V2] failed after ${MAX_ATTEMPTS} attempts (rc=${rc})"
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "REPORT=${report}"
  echo "BUNDLE=${zip_path}"
  exit "$rc"
fi

echo "OUT_ROOT=${OUT_ROOT}"
echo "REPORT=${report}"
echo "BUNDLE=${zip_path}"
