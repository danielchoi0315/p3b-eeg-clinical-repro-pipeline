#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

MAX_ATTEMPTS=6
WALL_HOURS=10
OUT_ROOT=""
DATA_ROOT="/filesystemHcog/openneuro"
FEATURES_ROOT=""
CONFIG="configs/default.yaml"
MEGA_CONFIG="configs/nn_final_mega.yaml"
DATASETS_CONFIG="configs/datasets_nn_final_mega.yaml"
GPU_PARALLEL_PROCS=""
CPU_WORKERS=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out_root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --data_root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --features_root)
      FEATURES_ROOT="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --mega_config)
      MEGA_CONFIG="$2"
      shift 2
      ;;
    --datasets_config)
      DATASETS_CONFIG="$2"
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
  OUT_ROOT="/filesystemHcog/runs/$(date +%Y%m%d_%H%M%S)_NN_FINAL_MEGA"
fi
if [[ -z "$FEATURES_ROOT" ]]; then
  FEATURES_ROOT="/filesystemHcog/features_cache_NN_FINAL_MEGA_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUT_ROOT/AUDIT" "$OUT_ROOT/OUTZIP"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

attempt=1
resume="false"
rc=1
fixloop_start_ts=$(date +%s)

apply_minimal_patch() {
  local log_blob="$1"
  local patched=0

  # Runtime-only env adjustments (non-science).
  if echo "$log_blob" | rg -qi "HDF5|file locking"; then
    export HDF5_USE_FILE_LOCKING=FALSE
  fi

  # Minimal code patch: make JSON writes robust if serialization fails.
  if echo "$log_blob" | rg -qi "not JSON serializable|TypeError: Object of type"; then
    if rg -q "json.dumps\(clean, indent=2, allow_nan=False\)" scripts/nn_final_mega_runner.py; then
      :
    else
      perl -0pi -e 's/json\.dumps\(payload, indent=2\)/json.dumps(payload, indent=2, default=str)/g' scripts/nn_final_mega_runner.py
      patched=1
    fi
  fi

  # Minimal code patch: accept bool-style --resume values if parser errors appear.
  if echo "$log_blob" | rg -qi "unrecognized arguments: true|invalid choice: true|argument --resume"; then
    perl -0pi -e 's/parser\.add_argument\("--resume",\s*type=_parse_bool,\s*default=False\)/parser.add_argument("--resume", type=_parse_bool, default=False, help="true\/false")/g' scripts/nn_final_mega_runner.py || true
    patched=1
  fi

  # Syntax/import/name failures: recompile to surface deterministic line refs.
  if echo "$log_blob" | rg -qi "SyntaxError|ImportError|NameError|ModuleNotFoundError"; then
    python -m py_compile scripts/nn_final_mega_runner.py common/hardware.py || true
  fi

  # Resource pressure mitigation (non-science).
  if echo "$log_blob" | rg -qi "CUDA out of memory|out of memory|Resource temporarily unavailable|Killed"; then
    if [[ -n "$GPU_PARALLEL_PROCS" ]]; then
      if [[ "$GPU_PARALLEL_PROCS" -gt 8 ]]; then
        GPU_PARALLEL_PROCS=$((GPU_PARALLEL_PROCS - 2))
      fi
    else
      GPU_PARALLEL_PROCS="8"
    fi

    if [[ -n "$CPU_WORKERS" ]]; then
      if [[ "$CPU_WORKERS" -gt 24 ]]; then
        CPU_WORKERS=$((CPU_WORKERS - 4))
      fi
    else
      CPU_WORKERS="24"
    fi
  fi

  if [[ "$patched" -eq 1 ]]; then
    echo "[NN_FINAL_MEGA] applied minimal code patch" | tee -a "$OUT_ROOT/AUDIT/FAIL_CONTEXT.txt"
  fi
}

while [[ "$attempt" -le "$MAX_ATTEMPTS" ]]; do
  now_ts=$(date +%s)
  elapsed_s=$((now_ts - fixloop_start_ts))
  elapsed_h=$(ELAPSED_S="$elapsed_s" python - <<'PY'
import os
print(int(os.environ['ELAPSED_S']) / 3600.0)
PY
)
  echo "[NN_FINAL_MEGA] attempt ${attempt}/${MAX_ATTEMPTS} out_root=${OUT_ROOT} elapsed_h=${elapsed_h}"

  set +e
  cmd=(python scripts/nn_final_mega_runner.py
    --out_root "$OUT_ROOT"
    --data_root "$DATA_ROOT"
    --features_root "$FEATURES_ROOT"
    --config "$CONFIG"
    --mega_config "$MEGA_CONFIG"
    --datasets_config "$DATASETS_CONFIG"
    --wall_hours "$WALL_HOURS"
    --resume "$resume"
  )
  if [[ -n "$GPU_PARALLEL_PROCS" ]]; then
    cmd+=(--gpu_parallel_procs "$GPU_PARALLEL_PROCS")
  fi
  if [[ -n "$CPU_WORKERS" ]]; then
    cmd+=(--cpu_workers "$CPU_WORKERS")
  fi
  cmd+=("${EXTRA_ARGS[@]}")
  "${cmd[@]}"
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    echo "[NN_FINAL_MEGA] success"
    break
  fi

  fail_ctx="$OUT_ROOT/AUDIT/FAIL_CONTEXT.txt"
  newest_log="$(ls -1t "$OUT_ROOT"/AUDIT/*.log 2>/dev/null | head -n 1 || true)"
  {
    echo ""
    echo "==== attempt=${attempt} ts=$(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc} ===="
    if [[ -f "$OUT_ROOT/AUDIT/run_status.json" ]]; then
      cat "$OUT_ROOT/AUDIT/run_status.json"
    fi
    if [[ -n "$newest_log" && -f "$newest_log" ]]; then
      echo ""
      echo "---- tail ${newest_log} ----"
      tail -n 320 "$newest_log" || true
    fi
  } >> "$fail_ctx"

  log_blob="$(tail -n 1600 "$OUT_ROOT"/AUDIT/*.log 2>/dev/null || true)"
  apply_minimal_patch "$log_blob"

  resume="true"
  attempt=$((attempt + 1))
done

report="$OUT_ROOT/AUDIT/NN_FINAL_MEGA_REPORT.md"
zip_path="$OUT_ROOT/OUTZIP/NN_FINAL_MEGA_SUBMISSION_PACKET.zip"

if [[ "$rc" -ne 0 ]]; then
  echo "[NN_FINAL_MEGA] failed after ${MAX_ATTEMPTS} attempts (rc=${rc})"
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "REPORT=${report}"
  echo "BUNDLE=${zip_path}"
  exit "$rc"
fi

echo "OUT_ROOT=${OUT_ROOT}"
echo "REPORT=${report}"
echo "BUNDLE=${zip_path}"
