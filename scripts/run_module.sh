#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

MODULE=""
FEATURES_ROOT=""
OUT_ROOT=""
CONFIG="configs/default.yaml"
SEEDS="0"
RESUME=1
ONLY_DATASET=""
ONLY_COHORT=""
HEALTHY_COHORT="healthy"
CLINICAL_COHORT="clinical"
HEALTHY_DATASET_IDS="ds005095,ds003655,ds004117"
SEVERITY_CSV=""
ENABLE_FP8=0
GPU_LOG_CSV=""
GPU_LOG_TAG=""

usage() {
  cat <<EOF
Usage:
  bash scripts/run_module.sh --module 03|04 --features_root <path> --out_root <path> [options]

Options:
  --config <path>
  --seeds "0,1,2,3,4"
  --no-resume
  --only_dataset <dataset_id>           (module 03)
  --only_cohort <cohort>                (module 03)
  --healthy_cohort <cohort>             (module 04)
  --clinical_cohort <cohort>            (module 04)
  --healthy_dataset_ids "a,b,c"         (module 04)
  --severity_csv <path>                 (module 04)
  --enable_fp8
  --gpu_log_csv <path>                  (module 03/04 optional)
  --gpu_log_tag <tag>                   (module 03/04 optional)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --module) MODULE="$2"; shift 2 ;;
    --features_root) FEATURES_ROOT="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --no-resume) RESUME=0; shift 1 ;;
    --only_dataset) ONLY_DATASET="$2"; shift 2 ;;
    --only_cohort) ONLY_COHORT="$2"; shift 2 ;;
    --healthy_cohort) HEALTHY_COHORT="$2"; shift 2 ;;
    --clinical_cohort) CLINICAL_COHORT="$2"; shift 2 ;;
    --healthy_dataset_ids) HEALTHY_DATASET_IDS="$2"; shift 2 ;;
    --severity_csv) SEVERITY_CSV="$2"; shift 2 ;;
    --enable_fp8) ENABLE_FP8=1; shift 1 ;;
    --gpu_log_csv) GPU_LOG_CSV="$2"; shift 2 ;;
    --gpu_log_tag) GPU_LOG_TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$MODULE" || -z "$FEATURES_ROOT" || -z "$OUT_ROOT" ]]; then
  usage
  exit 1
fi

if [[ "$MODULE" != "03" && "$MODULE" != "04" ]]; then
  echo "--module must be 03 or 04"
  exit 1
fi

mkdir -p "$OUT_ROOT"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

IFS=',' read -r -a seed_arr <<< "$SEEDS"
for raw_seed in "${seed_arr[@]}"; do
  seed="$(echo "$raw_seed" | xargs)"
  [[ -z "$seed" ]] && continue

  seed_out="${OUT_ROOT}/seed_${seed}"
  run_id="seed_${seed}"
  mkdir -p "$seed_out"

  if [[ "$MODULE" == "03" ]]; then
    done_file="${seed_out}/reports/mediation/${run_id}/mediation_summary.json"
    if [[ "$RESUME" -eq 1 && -f "$done_file" ]]; then
      echo "[run_module] module03 seed=${seed} already complete; skipping"
      continue
    fi

    cmd=(python 03_bayesian_mechanism_GPU.py
      --features_root "$FEATURES_ROOT"
      --out_root "$seed_out"
      --config "$CONFIG"
      --run_id "$run_id"
      --seed "$seed"
    )
    if [[ -n "$GPU_LOG_CSV" ]]; then
      cmd+=(--gpu_log_csv "$GPU_LOG_CSV")
      cmd+=(--gpu_log_tag "${GPU_LOG_TAG:-module03_seed${seed}}")
    fi
    if [[ "$RESUME" -eq 1 ]]; then
      cmd+=(--resume)
    fi
    if [[ -n "$ONLY_DATASET" ]]; then
      cmd+=(--only_dataset "$ONLY_DATASET")
    fi
    if [[ -n "$ONLY_COHORT" ]]; then
      cmd+=(--only_cohort "$ONLY_COHORT")
    fi
    if [[ "$ENABLE_FP8" -eq 1 ]]; then
      cmd+=(--enable_fp8)
    fi

    echo "[run_module] running module03 seed=${seed}"
    "${cmd[@]}"
  else
    done_file="${seed_out}/reports/normative/${run_id}/normative_metrics.json"
    if [[ "$RESUME" -eq 1 && -f "$done_file" ]]; then
      echo "[run_module] module04 seed=${seed} already complete; skipping"
      continue
    fi

    cmd=(python 04_normative_clinical_GPU.py
      --features_root "$FEATURES_ROOT"
      --out_root "$seed_out"
      --config "$CONFIG"
      --run_id "$run_id"
      --seed "$seed"
      --healthy_cohort "$HEALTHY_COHORT"
      --clinical_cohort "$CLINICAL_COHORT"
      --healthy_dataset_ids "$HEALTHY_DATASET_IDS"
    )
    if [[ -n "$GPU_LOG_CSV" ]]; then
      cmd+=(--gpu_log_csv "$GPU_LOG_CSV")
      cmd+=(--gpu_log_tag "${GPU_LOG_TAG:-module04_seed${seed}}")
    fi
    if [[ "$RESUME" -eq 1 ]]; then
      cmd+=(--resume)
    fi
    if [[ -n "$SEVERITY_CSV" ]]; then
      cmd+=(--severity_csv "$SEVERITY_CSV")
    fi
    if [[ "$ENABLE_FP8" -eq 1 ]]; then
      cmd+=(--enable_fp8)
    fi

    echo "[run_module] running module04 seed=${seed}"
    "${cmd[@]}"
  fi

done
