#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="/filesystemHcog/logs"
DATA_ROOT="/filesystemHcog/openneuro"
FEATURES_ROOT="/filesystemHcog/features_cache"
CLINICAL_BIDS_ROOT="/filesystemHcog/clinical_bids"
VENV_DIR="${VENV_DIR:-/filesystemHcog/venvs/research_pipeline}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  for cand in /filesystemHcog/venvs/*; do
    if [[ -x "${cand}/bin/python" ]]; then
      VENV_DIR="${cand}"
      break
    fi
  done
fi
CONFIG_PATH="configs/default.yaml"
DATASETS_CONFIG="configs/datasets.yaml"
SEEDS="0,1,2,3,4"
RESUME=0
OUT_ROOT_ARG="${OUT_ROOT:-}"
DATALAD_BIN="/usr/bin/datalad"
DATALAD_GET_SOURCE="${DATALAD_GET_SOURCE:-s3-PUBLIC}"
DATALAD_GET_BATCH_SIZE="${DATALAD_GET_BATCH_SIZE:-4}"
CPU_CORES="$(nproc || echo 8)"
# git-annex 8.x on GH200 can spin at -J>1 for large OpenNeuro trees; use stable single-job staging.
DEFAULT_DATALAD_JOBS=1
DATALAD_GET_JOBS="${DATALAD_GET_JOBS:-$DEFAULT_DATALAD_JOBS}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1; shift 1 ;;
    --out_root) OUT_ROOT_ARG="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: bash scripts/run_gh200_full.sh [--resume] [--out_root /filesystemHcog/runs/<ts>] [--seeds 0,1,2,3,4]"
      exit 1
      ;;
  esac
done

if [[ -n "$OUT_ROOT_ARG" ]]; then
  OUT_ROOT="$OUT_ROOT_ARG"
else
  if [[ "$RESUME" -eq 1 ]]; then
    echo "[ERROR] --resume requires --out_root or OUT_ROOT env var"
    exit 1
  fi
  OUT_ROOT="/filesystemHcog/runs/${RUN_TS}"
fi

if [[ ! -d /filesystemHcog ]]; then
  echo "[ERROR] Required fast local disk path missing: /filesystemHcog"
  exit 1
fi

if ! mkdir -p "$LOG_ROOT" /filesystemHcog/venvs "$DATA_ROOT" "$FEATURES_ROOT" /filesystemHcog/runs; then
  echo "[ERROR] Unable to create required directories under /filesystemHcog (permission or mount issue)"
  exit 1
fi

if [[ "$RESUME" -eq 1 ]]; then
  if [[ ! -d "$OUT_ROOT" ]]; then
    echo "[ERROR] --resume set but OUT_ROOT does not exist: $OUT_ROOT"
    exit 1
  fi
fi

mkdir -p "$OUT_ROOT"
AUDIT_DIR="${OUT_ROOT}/AUDIT"
mkdir -p "$AUDIT_DIR"
STAGING_MANIFEST="${AUDIT_DIR}/staging_manifest.json"

export RUN_TS LOG_ROOT DATA_ROOT FEATURES_ROOT CLINICAL_BIDS_ROOT VENV_DIR CONFIG_PATH DATASETS_CONFIG SEEDS OUT_ROOT AUDIT_DIR STAGING_MANIFEST REPO_ROOT DATALAD_BIN DATALAD_GET_SOURCE DATALAD_GET_BATCH_SIZE DATALAD_GET_JOBS CPU_CORES

# Ensure we are not forcing RTX5090 arch.
unset TORCH_CUDA_ARCH_LIST || true

STAGES=(
  "preflight"
  "env_deps"
  "audit_snapshot_pre"
  "stage_data"
  "verify_data"
  "pipeline"
  "audit_snapshot_post"
  "gpu_stats"
)

declare -A STAGE_STATUS
declare -A STAGE_LOG
for s in "${STAGES[@]}"; do
  STAGE_STATUS["$s"]="NOT_RUN"
  STAGE_LOG["$s"]="${LOG_ROOT}/${RUN_TS}_${s}.log"
done

print_status_table() {
  echo
  echo "Run status:"
  printf "%-20s %-8s %s\n" "STAGE" "STATUS" "LOG"
  for s in "${STAGES[@]}"; do
    printf "%-20s %-8s %s\n" "$s" "${STAGE_STATUS[$s]}" "${STAGE_LOG[$s]}"
  done
}

on_exit() {
  local rc=$?
  echo
  echo "=============================================="
  if [[ $rc -eq 0 ]]; then
    echo "run_gh200_full.sh completed successfully"
  else
    echo "run_gh200_full.sh failed with exit code $rc"
  fi
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "aggregate_results=${OUT_ROOT}/aggregate_results.json"
  echo "audit_summary=${OUT_ROOT}/AUDIT_SUMMARY.md"
  echo "audit_copy=${AUDIT_DIR}/AUDIT_SUMMARY.md"
  echo "gpu_util_csv=${OUT_ROOT}/gpu_util.csv"
  echo "external_gpu_monitor_csv=${AUDIT_DIR}/nvidia_smi_1hz.csv"
  print_status_table
  echo "=============================================="
}
trap on_exit EXIT

run_stage() {
  local stage="$1"
  shift
  local log_path="${STAGE_LOG[$stage]}"
  STAGE_STATUS["$stage"]="RUNNING"

  set +e
  (
    set -euo pipefail
    echo "[${stage}] START $(date -Is)"
    "$@"
    echo "[${stage}] END $(date -Is)"
  ) 2>&1 | tee "$log_path"
  local ec=${PIPESTATUS[0]}
  set -e
  if [[ $ec -eq 0 ]]; then
    STAGE_STATUS["$stage"]="PASS"
  else
    STAGE_STATUS["$stage"]="FAIL"
  fi
  return $ec
}

run_stage "preflight" bash -lc '
  set -euo pipefail

  if [[ ! -d /filesystemHcog ]]; then
    echo "[ERROR] Required fast local disk path missing: /filesystemHcog"
    exit 1
  fi

  free_bytes=$(df -PB1 /filesystemHcog | awk "NR==2{print \$4}")
  free_gb=$(python - <<PY
b = int("${free_bytes}")
print(f"{b/1024**3:.2f}")
PY
)
  free_tb=$(python - <<PY
b = int("${free_bytes}")
print(f"{b/1024**4:.2f}")
PY
)
  echo "Free space on /filesystemHcog: ${free_gb} GB (${free_tb} TB)"

  min_bytes=$((500 * 1024 * 1024 * 1024))
  target_bytes=$((2 * 1024 * 1024 * 1024 * 1024))
  if (( free_bytes < min_bytes )); then
    echo "[ERROR] Free space below hard minimum 500GB"
    exit 1
  fi
  if (( free_bytes < target_bytes )); then
    echo "[WARN] Free space is below preferred 2TB but above hard minimum 500GB; continuing"
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found"
    exit 1
  fi

  gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)
  gpu_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | xargs)
  echo "Detected GPU: ${gpu_name} (compute_cap=${gpu_cc})"

  shopt -s nocasematch
  if [[ ! "${gpu_name}" =~ h100|hopper|gh200 ]] && [[ ! "${gpu_cc}" =~ ^9\. ]]; then
    echo "[ERROR] GPU is not Hopper/H100-class"
    exit 1
  fi
  shopt -u nocasematch

  if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    echo "[WARN] TORCH_CUDA_ARCH_LIST was set to ${TORCH_CUDA_ARCH_LIST}; unsetting for GH200 runtime"
    unset TORCH_CUDA_ARCH_LIST
  fi
  echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST-<unset>}"
'

run_stage "env_deps" bash -lc '
  set -euo pipefail

  if command -v sudo >/dev/null 2>&1; then
    if sudo -n true 2>/dev/null; then
      SUDO="sudo"
    elif [[ "$(id -u)" -eq 0 ]]; then
      SUDO=""
    else
      echo "[ERROR] sudo is required for apt install but passwordless sudo is unavailable"
      exit 1
    fi
  else
    if [[ "$(id -u)" -eq 0 ]]; then
      SUDO=""
    else
      echo "[ERROR] apt install requires root/sudo"
      exit 1
    fi
  fi

  ${SUDO} apt-get update
  ${SUDO} apt-get install -y datalad git-annex

  mkdir -p /filesystemHcog/venvs
  python3 -m venv "${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"

  python -m pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt

  python -m py_compile $(git ls-files "*.py")

  python - <<PY
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda_version={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] torch reports CUDA unavailable")
name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
print(f"device_name={name}")
print(f"compute_capability={cc}")
name_l = name.lower()
if cc[0] != 9 and all(k not in name_l for k in ("h100", "hopper", "gh200")):
    raise SystemExit("[ERROR] torch sees non-Hopper GPU")
PY
'

run_stage "audit_snapshot_pre" bash -lc '
  set -euo pipefail
  source "${VENV_DIR}/bin/activate"

  mkdir -p "${AUDIT_DIR}"
  nvidia-smi -q > "${AUDIT_DIR}/nvidia_smi_q.txt"
  nvidia-smi --query-gpu=timestamp,name,compute_cap,driver_version,memory.total,memory.used,power.draw,utilization.gpu,utilization.memory --format=csv > "${AUDIT_DIR}/nvidia_smi_query_snapshot.csv"

  python - <<PY > "${AUDIT_DIR}/torch_info.json"
import json
import torch
payload = {
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
}
if torch.cuda.is_available():
    payload["device_name"] = torch.cuda.get_device_name(0)
    payload["compute_capability"] = torch.cuda.get_device_capability(0)
print(json.dumps(payload, indent=2))
PY

  pip freeze > "${AUDIT_DIR}/pip_freeze.txt"
  (git rev-parse HEAD || true) > "${AUDIT_DIR}/git_rev_parse_head.txt" 2>&1
  git status --short > "${AUDIT_DIR}/git_status_short.txt"
'

run_stage "stage_data" bash -lc '
  set -euo pipefail
  source "${VENV_DIR}/bin/activate"
  export PYTHONUNBUFFERED=1
  echo "Using DATALAD_BIN=${DATALAD_BIN}"
  echo "Using DATALAD_GET_SOURCE=${DATALAD_GET_SOURCE}"
  echo "Using DATALAD_GET_JOBS=${DATALAD_GET_JOBS}"
  echo "Using DATALAD_GET_BATCH_SIZE=${DATALAD_GET_BATCH_SIZE}"

  bash 00_stage_data.sh \
    --config "${DATASETS_CONFIG}" \
    --openneuro_root "${DATA_ROOT}" \
    --manifest_out "${STAGING_MANIFEST}"
'

run_stage "verify_data" bash -lc '
  set -euo pipefail

  for ds in ds003838 ds005095 ds003655 ds004117; do
    ds_dir="${DATA_ROOT}/${ds}"
    if [[ ! -d "${ds_dir}" ]]; then
      echo "[ERROR] Missing dataset directory: ${ds_dir}"
      exit 1
    fi
    if [[ ! -f "${ds_dir}/dataset_description.json" ]]; then
      echo "[ERROR] Missing dataset_description.json: ${ds_dir}"
      exit 1
    fi
    if [[ ! -f "${ds_dir}/participants.tsv" ]]; then
      echo "[ERROR] Missing participants.tsv: ${ds_dir}"
      exit 1
    fi

    eeg_probe=$(find "${ds_dir}" -type f \( -path "*/sub-*/ses-*/eeg/*" -o -path "*/sub-*/eeg/*" \) | head -n1 || true)
    if [[ -z "${eeg_probe}" ]]; then
      echo "[ERROR] No non-empty EEG tree found in ${ds_dir} (checked sub-*/ses-*/eeg and sub-*/eeg)"
      exit 1
    fi
    echo "Verified ${ds}: eeg sample=${eeg_probe}"
  done

  eyetrack_probe=$(find -L "${DATA_ROOT}/ds003838" -type f \( -name "*_eyetrack.tsv" -o -name "*_eyetrack.tsv.gz" -o -name "*_pupil.tsv" -o -name "*_pupil.tsv.gz" \) | head -n1 || true)
  if [[ -z "${eyetrack_probe}" ]]; then
    echo "[ERROR] ds003838 has no eyetrack/pupil files; module 03 mechanism requires them"
    exit 1
  fi
  echo "Verified ds003838 eyetrack/pupil sample=${eyetrack_probe}"
'

run_stage "pipeline" bash -lc '
  set -euo pipefail
  source "${VENV_DIR}/bin/activate"

  tuned_config="${AUDIT_DIR}/gh200_tuned_config.yaml"
  python - <<PY
from pathlib import Path
import os
import yaml

base = Path("${CONFIG_PATH}")
cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
cores = int(os.environ.get("CPU_CORES", "8"))
# GPU modules are memory-light; too many workers causes overhead/stalls.
workers = max(8, min(24, max(8, cores // 2)))
for sec in ("bayes_mediation", "normative"):
    block = cfg.setdefault(sec, {})
    block["num_workers"] = int(workers)
    block["prefetch_to_gpu"] = True
tuned = Path("${AUDIT_DIR}/gh200_tuned_config.yaml")
tuned.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"Wrote tuned config: {tuned}")
print(f"Using GPU-module num_workers={workers} based on CPU_CORES={cores}")
PY

  monitor_csv="${AUDIT_DIR}/nvidia_smi_1hz.csv"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 1 > "${monitor_csv}" &
  monitor_pid=$!
  echo "${monitor_pid}" > "${AUDIT_DIR}/nvidia_smi_1hz.pid"

  cleanup_monitor() {
    if kill -0 "${monitor_pid}" >/dev/null 2>&1; then
      kill "${monitor_pid}" >/dev/null 2>&1 || true
      wait "${monitor_pid}" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup_monitor EXIT

  bash scripts/run_all.sh \
    --out_root "${OUT_ROOT}" \
    --data_root "${DATA_ROOT}" \
    --features_root "${FEATURES_ROOT}" \
    --log_root "${LOG_ROOT}" \
    --clinical_bids_root "${CLINICAL_BIDS_ROOT}" \
    --config "${tuned_config}" \
    --datasets_config "${DATASETS_CONFIG}" \
    --seeds "${SEEDS}"

  cleanup_monitor
  trap - EXIT
'

run_stage "audit_snapshot_post" bash -lc '
  set -euo pipefail

  if [[ ! -f "${OUT_ROOT}/AUDIT_SUMMARY.md" ]]; then
    echo "[ERROR] Missing ${OUT_ROOT}/AUDIT_SUMMARY.md"
    exit 1
  fi
  if [[ ! -f "${OUT_ROOT}/aggregate_results.json" ]]; then
    echo "[ERROR] Missing ${OUT_ROOT}/aggregate_results.json"
    exit 1
  fi
  if [[ ! -f "${OUT_ROOT}/gpu_util.csv" ]]; then
    echo "[ERROR] Missing ${OUT_ROOT}/gpu_util.csv"
    exit 1
  fi

  cp "${OUT_ROOT}/AUDIT_SUMMARY.md" "${AUDIT_DIR}/AUDIT_SUMMARY.md"
  cp "${OUT_ROOT}/aggregate_results.json" "${AUDIT_DIR}/aggregate_results.json"

  nvidia-smi --query-gpu=timestamp,name,compute_cap,driver_version,memory.total,memory.used,power.draw,utilization.gpu,utilization.memory --format=csv > "${AUDIT_DIR}/nvidia_smi_query_snapshot_post.csv"
'

run_stage "gpu_stats" bash -lc '
  set -euo pipefail

  python - <<PY
import csv
import datetime as dt
import json
from pathlib import Path
import statistics

csv_path = Path("${AUDIT_DIR}/nvidia_smi_1hz.csv")
if not csv_path.exists():
    raise SystemExit(f"[ERROR] Missing monitor csv: {csv_path}")

start_epoch = None
end_epoch = None
start_file = Path("${OUT_ROOT}/AUDIT/module03_04_start_epoch.txt")
end_file = Path("${OUT_ROOT}/AUDIT/module03_04_end_epoch.txt")
if start_file.exists() and end_file.exists():
    try:
        start_epoch = float(start_file.read_text().strip())
        end_epoch = float(end_file.read_text().strip())
    except Exception:
        start_epoch = None
        end_epoch = None

fmt_candidates = ["%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"]

def parse_ts(s: str):
    s = s.strip()
    for fmt in fmt_candidates:
        try:
            return dt.datetime.strptime(s, fmt).timestamp()
        except Exception:
            pass
    return None

vals = []
all_vals = []
with csv_path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ts = parse_ts(row.get("timestamp", ""))
        if ts is None:
            continue
        raw = row.get("utilization.gpu [%]", "").strip()
        if not raw or raw.upper().startswith("N/"):
            continue
        try:
            val = float(raw.split()[0])
        except Exception:
            continue
        all_vals.append(val)
        if start_epoch is not None and end_epoch is not None:
            if ts < start_epoch or ts > end_epoch:
                continue
        vals.append(val)

if not vals and all_vals:
    # Resume mode may skip module03/04, so current monitor window can miss active GPU phases.
    print("[WARN] No module03/04-window samples in current monitor file; falling back to full monitor span.")
    vals = list(all_vals)

out = {
    "n_samples": 0,
    "mean_util_gpu_pct": float("nan"),
    "median_util_gpu_pct": float("nan"),
    "window_start_epoch": start_epoch,
    "window_end_epoch": end_epoch,
}
if vals:
    out["n_samples"] = len(vals)
    out["mean_util_gpu_pct"] = float(sum(vals) / len(vals))
    out["median_util_gpu_pct"] = float(statistics.median(vals))
else:
    print("[WARN] No usable GPU utilization samples found; writing NaN stats.")
out_path = Path("${AUDIT_DIR}/gpu_util_stats.json")
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
PY

  mean_util=$(python - <<PY
import json
from pathlib import Path
p = Path("${AUDIT_DIR}/gpu_util_stats.json")
print(json.loads(p.read_text())["mean_util_gpu_pct"])
PY
)
  median_util=$(python - <<PY
import json
from pathlib import Path
p = Path("${AUDIT_DIR}/gpu_util_stats.json")
print(json.loads(p.read_text())["median_util_gpu_pct"])
PY
)

  echo "Mean GPU util during 03/04: ${mean_util}%"
  echo "Median GPU util during 03/04: ${median_util}%"

  python - <<PY
import math
mean_util = float("${mean_util}")
if math.isnan(mean_util):
    print("[WARN] Mean GPU utilization is NaN (no samples in this resume pass).")
elif mean_util < 70.0:
    print("[WARN] Mean GPU utilization < 70% during module 03/04.")
    print("Top suspected bottlenecks + exact knobs:")
    print("- dataloader stalls: set bayes_mediation.prefetch_to_gpu=true and normative.prefetch_to_gpu=true; raise num_workers carefully")
    print("- batch size too small: increase bayes_mediation.batch_size and normative.batch_size or raise batch_tuning.target_low toward 0.90")
    print("- compile disabled/fallback: keep compile=true; inspect module logs for torch.compile fallback reasons")
    print("- CPU oversubscription: cap OMP/MKL/OPENBLAS/NUMEXPR threads lower if workers are high")
PY
'

echo
echo "OUT_ROOT=${OUT_ROOT}"
echo "aggregate_results=${OUT_ROOT}/aggregate_results.json"
echo "audit_summary=${OUT_ROOT}/AUDIT_SUMMARY.md"
echo "audit_copy=${AUDIT_DIR}/AUDIT_SUMMARY.md"
echo "gpu_util_csv=${OUT_ROOT}/gpu_util.csv"
echo "external_gpu_monitor_csv=${AUDIT_DIR}/nvidia_smi_1hz.csv"
print_status_table
