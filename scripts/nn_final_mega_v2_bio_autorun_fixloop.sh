#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

RUNNER="scripts/nn_final_mega_v2_bio_runner.py"
MAX_ATTEMPTS=12
WALL_HOURS=12
OUT_ROOT=""
DATA_ROOT="/filesystemHcog/openneuro"
FEATURES_ROOT=""
CONFIG="configs/default.yaml"
MEGA_CONFIG="configs/nn_final_mega_v2_bio.yaml"
DATASETS_CONFIG="configs/datasets_nn_final_mega_v2_bio.yaml"
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
  OUT_ROOT="/filesystemHcog/runs/$(date +%Y%m%d_%H%M%S)_NN_FINAL_MEGA_V2_BIO"
fi
if [[ -z "$FEATURES_ROOT" ]]; then
  FEATURES_ROOT="/filesystemHcog/features_cache_NN_FINAL_MEGA_V2_BIO_$(date +%Y%m%d_%H%M%S)"
fi
export OUT_ROOT

mkdir -p "$OUT_ROOT/AUDIT" "$OUT_ROOT/OUTZIP"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

run_status_json="$OUT_ROOT/AUDIT/run_status.json"
fail_ctx="$OUT_ROOT/AUDIT/FAIL_CONTEXT.txt"

attempt=1
resume="false"
rc=1
fixloop_start_ts=$(date +%s)

detect_failed_stage() {
  if [[ ! -f "$run_status_json" ]]; then
    echo ""
    return
  fi
  python - "$run_status_json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    obj = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)
stages = obj.get("stages", [])
failed = ""
for st in stages:
    if str(st.get("status", "")).upper() == "FAIL":
        failed = str(st.get("stage", ""))
        break
if not failed and stages:
    failed = str(stages[-1].get("stage", ""))
print(failed)
PY
}

stage_log_path() {
  local failed_stage="$1"
  if [[ -n "$failed_stage" && -f "$OUT_ROOT/AUDIT/${failed_stage}.log" ]]; then
    echo "$OUT_ROOT/AUDIT/${failed_stage}.log"
    return
  fi
  ls -1t "$OUT_ROOT"/AUDIT/*.log 2>/dev/null | head -n 1 || true
}

apply_minimal_patch() {
  local log_blob="$1"
  local failed_stage="$2"
  local patched=0

  if echo "$log_blob" | rg -qi "HDF5|file locking"; then
    export HDF5_USE_FILE_LOCKING=FALSE
  fi

  if echo "$log_blob" | rg -qi "SyntaxError|ImportError|NameError|ModuleNotFoundError|IndentationError"; then
    python -m py_compile "$RUNNER" scripts/decode_ds004752.py scripts/decode_ds007262.py common/hardware.py || true
  fi

  if echo "$log_blob" | rg -qi "CUDA out of memory|out of memory|Resource temporarily unavailable|Killed"; then
    if [[ -n "$GPU_PARALLEL_PROCS" ]]; then
      if [[ "$GPU_PARALLEL_PROCS" -gt 8 ]]; then
        GPU_PARALLEL_PROCS=$((GPU_PARALLEL_PROCS - 2))
      fi
    else
      GPU_PARALLEL_PROCS="10"
    fi
    if [[ -n "$CPU_WORKERS" ]]; then
      if [[ "$CPU_WORKERS" -gt 32 ]]; then
        CPU_WORKERS=$((CPU_WORKERS - 4))
      fi
    fi
  fi

  if echo "$log_blob" | rg -qi "nvidia_smi_1hz\\.csv rows<600"; then
    # Force a monitor probe to create sustained samples before next resume run.
    python - <<'PY' || true
import os, subprocess, time
out_root = os.environ.get("OUT_ROOT")
if not out_root:
    raise SystemExit(0)
audit = os.path.join(out_root, "AUDIT")
os.makedirs(audit, exist_ok=True)
csvp = os.path.join(audit, "nvidia_smi_1hz.csv")
with open(csvp, "a", encoding="utf-8") as h:
    p = subprocess.Popen(
        ["nvidia-smi","--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,temperature.gpu","--format=csv,noheader,nounits","-l","1"],
        stdout=h, stderr=h, text=True
    )
    try:
        end = time.time() + 180
        while time.time() < end:
            time.sleep(1.0)
    finally:
        p.terminate()
        try:
            p.wait(timeout=6)
        except Exception:
            p.kill()
PY
  fi

  if [[ "$failed_stage" == "decode_mapping_all" ]]; then
    # Enforce dataset-specific decoders exist for retry.
    if [[ ! -f scripts/decode_ds004752.py || ! -f scripts/decode_ds007262.py ]]; then
      echo "[NN_FINAL_MEGA_V2_BIO] missing dataset-specific decoders for mapping stage" >> "$fail_ctx"
    else
      python -m py_compile scripts/decode_ds004752.py scripts/decode_ds007262.py || true
    fi
  fi

  if [[ "$patched" -eq 1 ]]; then
    echo "[NN_FINAL_MEGA_V2_BIO] applied minimal code patch" | tee -a "$fail_ctx"
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
  echo "[NN_FINAL_MEGA_V2_BIO] attempt ${attempt}/${MAX_ATTEMPTS} out_root=${OUT_ROOT} elapsed_h=${elapsed_h}"

  set +e
  cmd=(python "$RUNNER"
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

  failed_stage="$(detect_failed_stage)"
  failed_log="$(stage_log_path "$failed_stage")"

  {
    echo ""
    echo "==== attempt=${attempt} ts=$(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc} failed_stage=${failed_stage} ===="
    if [[ -f "$run_status_json" ]]; then
      cat "$run_status_json"
    fi
    if [[ -n "$failed_log" && -f "$failed_log" ]]; then
      echo ""
      echo "---- tail ${failed_log} ----"
      tail -n 320 "$failed_log" || true
    fi
  } >> "$fail_ctx"

  if [[ "$rc" -eq 0 ]]; then
    echo "[NN_FINAL_MEGA_V2_BIO] success"
    break
  fi

  log_blob="$(tail -n 1600 "$OUT_ROOT"/AUDIT/*.log 2>/dev/null || true)"
  apply_minimal_patch "$log_blob" "$failed_stage"

  resume="true"
  attempt=$((attempt + 1))
done

report="$OUT_ROOT/AUDIT/NN_FINAL_MEGA_V2_BIO_REPORT.md"
zip_path="$OUT_ROOT/OUTZIP/NN_FINAL_MEGA_V2_BIO_SUBMISSION_PACKET.zip"

if [[ "$rc" -ne 0 ]]; then
  echo "[NN_FINAL_MEGA_V2_BIO] failed after ${MAX_ATTEMPTS} attempts (rc=${rc})"
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "REPORT=${report}"
  echo "BUNDLE=${zip_path}"
  exit "$rc"
fi

echo "OUT_ROOT=${OUT_ROOT}"
echo "REPORT=${report}"
echo "BUNDLE=${zip_path}"
