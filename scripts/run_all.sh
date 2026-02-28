#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DATA_ROOT="/filesystemHcog/openneuro"
OUT_ROOT="/filesystemHcog/runs/${TIMESTAMP}"
FEATURES_ROOT="/filesystemHcog/features_cache"
LOG_ROOT="/filesystemHcog/logs"
CLINICAL_BIDS_ROOT="/filesystemHcog/clinical_bids"
CONFIG="configs/default.yaml"
DATASETS_CONFIG="configs/datasets.yaml"
LAWC_EVENT_MAP="configs/lawc_event_map.yaml"
SEEDS="0,1,2,3,4"
SEVERITY_CSV=""
DO_STAGE=1
RESUME=1
ALLOW_DETERMINISTIC=0

usage() {
  cat <<EOF
Usage: bash scripts/run_all.sh [options]

Options:
  --data_root <path>            (default: /filesystemHcog/openneuro)
  --out_root <path>             (default: /filesystemHcog/runs/<timestamp>)
  --features_root <path>        (default: /filesystemHcog/features_cache)
  --log_root <path>             (default: /filesystemHcog/logs)
  --clinical_bids_root <path>   (default: /filesystemHcog/clinical_bids)
  --config <path>               (default: configs/default.yaml)
  --datasets_config <path>      (default: configs/datasets.yaml)
  --lawc_event_map <path>       (default: configs/lawc_event_map.yaml)
  --seeds "0,1,2,3,4"           (default: 0,1,2,3,4)
  --severity_csv <path>
  --skip_stage
  --no-resume
  --allow_deterministic
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --features_root) FEATURES_ROOT="$2"; shift 2 ;;
    --log_root) LOG_ROOT="$2"; shift 2 ;;
    --clinical_bids_root) CLINICAL_BIDS_ROOT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --datasets_config) DATASETS_CONFIG="$2"; shift 2 ;;
    --lawc_event_map) LAWC_EVENT_MAP="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --severity_csv) SEVERITY_CSV="$2"; shift 2 ;;
    --skip_stage) DO_STAGE=0; shift 1 ;;
    --no-resume) RESUME=0; shift 1 ;;
    --allow_deterministic) ALLOW_DETERMINISTIC=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$OUT_ROOT" "$FEATURES_ROOT" "$LOG_ROOT"
mkdir -p "${OUT_ROOT}/AUDIT"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

RUN_LOG="${LOG_ROOT}/run_all_${TIMESTAMP}.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "[run_all] DATA_ROOT=${DATA_ROOT}"
echo "[run_all] OUT_ROOT=${OUT_ROOT}"
echo "[run_all] FEATURES_ROOT=${FEATURES_ROOT}"
echo "[run_all] LOG_ROOT=${LOG_ROOT}"
echo "[run_all] CLINICAL_BIDS_ROOT=${CLINICAL_BIDS_ROOT}"
echo "[run_all] CONFIG=${CONFIG}"
echo "[run_all] DATASETS_CONFIG=${DATASETS_CONFIG}"
echo "[run_all] LAWC_EVENT_MAP=${LAWC_EVENT_MAP}"
echo "[run_all] SEEDS=${SEEDS}"

STAGE_MANIFEST="${OUT_ROOT}/staging_manifest.json"
if [[ "$DO_STAGE" -eq 1 ]]; then
  echo "[run_all] Stage 00: DataLad staging + hash pinning + validation"
  bash 00_stage_data.sh \
    --config "$DATASETS_CONFIG" \
    --openneuro_root "$DATA_ROOT" \
    --manifest_out "$STAGE_MANIFEST"
fi

required_datasets=(ds003838 ds005095 ds003655 ds004117)
for ds in "${required_datasets[@]}"; do
  if [[ ! -d "${DATA_ROOT}/${ds}" ]]; then
    echo "[run_all][ERROR] Missing dataset directory: ${DATA_ROOT}/${ds}"
    exit 1
  fi
  if [[ ! -f "${DATA_ROOT}/${ds}/dataset_description.json" ]]; then
    echo "[run_all][ERROR] Missing BIDS dataset_description.json: ${DATA_ROOT}/${ds}"
    exit 1
  fi
done

CPU_CORES="$(nproc || echo 8)"
PREPROCESS_WORKERS_DEFAULT=$(( CPU_CORES / 2 ))
if (( PREPROCESS_WORKERS_DEFAULT < 2 )); then PREPROCESS_WORKERS_DEFAULT=2; fi
if (( PREPROCESS_WORKERS_DEFAULT > 32 )); then PREPROCESS_WORKERS_DEFAULT=32; fi
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-$PREPROCESS_WORKERS_DEFAULT}"
PREPROCESS_PER_RUN_THREADS_DEFAULT=$(( CPU_CORES / PREPROCESS_WORKERS ))
if (( PREPROCESS_PER_RUN_THREADS_DEFAULT < 1 )); then PREPROCESS_PER_RUN_THREADS_DEFAULT=1; fi
PREPROCESS_PER_RUN_THREADS="${PREPROCESS_PER_RUN_THREADS:-$PREPROCESS_PER_RUN_THREADS_DEFAULT}"
if (( PREPROCESS_WORKERS > 1 )); then
  # Avoid nested parallel pools (outer ProcessPool + inner MNE joblib), which can
  # hang on shutdown for large EEG batches.
  PREPROCESS_MNE_N_JOBS_DEFAULT=1
else
  PREPROCESS_MNE_N_JOBS_DEFAULT="$PREPROCESS_PER_RUN_THREADS"
  if (( PREPROCESS_MNE_N_JOBS_DEFAULT > 4 )); then PREPROCESS_MNE_N_JOBS_DEFAULT=4; fi
fi
PREPROCESS_MNE_N_JOBS="${PREPROCESS_MNE_N_JOBS:-$PREPROCESS_MNE_N_JOBS_DEFAULT}"
EXTRACT_WORKERS="${EXTRACT_WORKERS:-$PREPROCESS_WORKERS}"
EXTRACT_PER_RUN_THREADS="${EXTRACT_PER_RUN_THREADS:-$PREPROCESS_PER_RUN_THREADS}"

echo "[run_all] CPU_CORES=${CPU_CORES}"
echo "[run_all] PREPROCESS_WORKERS=${PREPROCESS_WORKERS}"
echo "[run_all] PREPROCESS_PER_RUN_THREADS=${PREPROCESS_PER_RUN_THREADS}"
echo "[run_all] PREPROCESS_MNE_N_JOBS=${PREPROCESS_MNE_N_JOBS}"
echo "[run_all] EXTRACT_WORKERS=${EXTRACT_WORKERS}"
echo "[run_all] EXTRACT_PER_RUN_THREADS=${EXTRACT_PER_RUN_THREADS}"

run_preprocess_extract() {
  local bids_root="$1"
  local deriv_root="$2"
  local cohort="$3"
  local dataset_id="$4"

  echo "[run_all] 01 preprocess dataset=${dataset_id} cohort=${cohort}"
  python 01_preprocess_CPU.py \
    --bids_root "$bids_root" \
    --deriv_root "$deriv_root" \
    --config "$CONFIG" \
    --workers "$PREPROCESS_WORKERS" \
    --per_run_threads "$PREPROCESS_PER_RUN_THREADS" \
    --mne_n_jobs "$PREPROCESS_MNE_N_JOBS"

  echo "[run_all] 02 extract dataset=${dataset_id} cohort=${cohort}"
  python 02_extract_features_CPU.py \
    --bids_root "$bids_root" \
    --deriv_root "$deriv_root" \
    --features_root "$FEATURES_ROOT" \
    --config "$CONFIG" \
    --lawc_event_map "$LAWC_EVENT_MAP" \
    --cohort "$cohort" \
    --dataset_id "$dataset_id" \
    --workers "$EXTRACT_WORKERS" \
    --per_run_threads "$EXTRACT_PER_RUN_THREADS"
}

# Mechanism dataset
run_preprocess_extract "${DATA_ROOT}/ds003838" "${OUT_ROOT}/derivatives/ds003838" "mechanism" "ds003838"

# Normative healthy Sternberg-family datasets
run_preprocess_extract "${DATA_ROOT}/ds005095" "${OUT_ROOT}/derivatives/ds005095" "healthy" "ds005095"
run_preprocess_extract "${DATA_ROOT}/ds003655" "${OUT_ROOT}/derivatives/ds003655" "healthy" "ds003655"
run_preprocess_extract "${DATA_ROOT}/ds004117" "${OUT_ROOT}/derivatives/ds004117" "healthy" "ds004117"

# Optional clinical BIDS application dataset
if [[ -d "$CLINICAL_BIDS_ROOT" && -f "$CLINICAL_BIDS_ROOT/dataset_description.json" ]]; then
  echo "[run_all] Found optional clinical BIDS at ${CLINICAL_BIDS_ROOT}; processing for cohort=clinical"
  run_preprocess_extract "$CLINICAL_BIDS_ROOT" "${OUT_ROOT}/derivatives/clinical_bids" "clinical" "clinical_bids"
else
  echo "[run_all] Optional clinical BIDS not found at ${CLINICAL_BIDS_ROOT}; module 04 will run normative-only"
fi

echo "[run_all] Inspecting events + running locked Law-C audit gate"
python scripts/inspect_events.py \
  --data_root "$DATA_ROOT" \
  --event_map "$LAWC_EVENT_MAP" \
  --datasets "ds005095,ds003655,ds004117" \
  --out_dir "${OUT_ROOT}/AUDIT"

python 05_audit_lawc.py \
  --features_root "$FEATURES_ROOT" \
  --out_root "$OUT_ROOT" \
  --event_map "$LAWC_EVENT_MAP" \
  --datasets "ds005095,ds003655,ds004117" \
  --n_perm 2000

module_resume_arg=()
if [[ "$RESUME" -eq 0 ]]; then
  module_resume_arg+=(--no-resume)
fi

# Seed sweep for module 03
date +%s > "${OUT_ROOT}/AUDIT/module03_04_start_epoch.txt"
bash scripts/run_module.sh \
  --module 03 \
  --features_root "$FEATURES_ROOT" \
  --out_root "$OUT_ROOT" \
  --config "$CONFIG" \
  --seeds "$SEEDS" \
  --only_dataset ds003838 \
  --only_cohort mechanism \
  "${module_resume_arg[@]}"

# Seed sweep for module 04
module04_cmd=(
  bash scripts/run_module.sh
  --module 04
  --features_root "$FEATURES_ROOT"
  --out_root "$OUT_ROOT"
  --config "$CONFIG"
  --seeds "$SEEDS"
  --healthy_cohort healthy
  --clinical_cohort clinical
  --healthy_dataset_ids "ds005095,ds003655,ds004117"
)
if [[ "$RESUME" -eq 0 ]]; then
  module04_cmd+=(--no-resume)
fi
if [[ -n "$SEVERITY_CSV" ]]; then
  module04_cmd+=(--severity_csv "$SEVERITY_CSV")
fi
"${module04_cmd[@]}"
date +%s > "${OUT_ROOT}/AUDIT/module03_04_end_epoch.txt"

aggregate_cmd=(
  python aggregate_results.py
  --out_root "$OUT_ROOT"
  --seeds "$SEEDS"
  --stage_manifest "$STAGE_MANIFEST"
)
if [[ "$ALLOW_DETERMINISTIC" -eq 1 ]]; then
  aggregate_cmd+=(--allow_deterministic)
fi
"${aggregate_cmd[@]}"

echo "[run_all] Complete. Outputs at ${OUT_ROOT}"
echo "[run_all] Logs at ${RUN_LOG}"
