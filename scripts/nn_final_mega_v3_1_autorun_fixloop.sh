#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MAX_ATTEMPTS=10

OUT_ROOT=""
WALL_HOURS="8"
DATA_ROOT="/filesystemHcog/openneuro"
CANONICAL_ROOT="/filesystemHcog/runs/20260223_052006_NN_FINAL_MEGA_V2_BIO"
INCLUDE_DATA="${INCLUDE_DATA:-0}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out_root)
      OUT_ROOT="$2"; shift 2 ;;
    --wall_hours)
      WALL_HOURS="$2"; shift 2 ;;
    --data_root)
      DATA_ROOT="$2"; shift 2 ;;
    --canonical_root)
      CANONICAL_ROOT="$2"; shift 2 ;;
    --include_data)
      INCLUDE_DATA="$2"; shift 2 ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "${OUT_ROOT}" ]]; then
  echo "ERROR: --out_root is required" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/AUDIT"
FAIL_CONTEXT="${OUT_ROOT}/AUDIT/FAIL_CONTEXT.txt"

run_once() {
  local resume_flag="$1"
  python scripts/nn_final_mega_v3_1_runner.py \
    --out_root "${OUT_ROOT}" \
    --data_root "${DATA_ROOT}" \
    --canonical_root "${CANONICAL_ROOT}" \
    --wall_hours "${WALL_HOURS}" \
    --include_data "${INCLUDE_DATA}" \
    --resume "${resume_flag}" \
    "${EXTRA_ARGS[@]}"
}

status_from_json() {
  python - <<PY
import json, pathlib
p = pathlib.Path("${OUT_ROOT}") / "AUDIT" / "run_status.json"
if not p.exists():
    print("MISSING")
    raise SystemExit(0)
j = json.loads(p.read_text())
print(j.get("status", "MISSING"))
PY
}

error_from_json() {
  python - <<PY
import json, pathlib
p = pathlib.Path("${OUT_ROOT}") / "AUDIT" / "run_status.json"
if not p.exists():
    print("")
    raise SystemExit(0)
j = json.loads(p.read_text())
print(j.get("error", ""))
PY
}

failing_stage_from_json() {
  python - <<PY
import json, pathlib
p = pathlib.Path("${OUT_ROOT}") / "AUDIT" / "run_status.json"
if not p.exists():
    print("")
    raise SystemExit(0)
j = json.loads(p.read_text())
stage = ""
for rec in j.get("stages", []):
    if str(rec.get("status", "")) == "FAIL":
        stage = str(rec.get("stage", ""))
        break
print(stage)
PY
}

append_fail_context() {
  local attempt="$1"
  local status="$2"
  local stage="$3"
  {
    echo "==== attempt=${attempt} ts=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=${status} stage=${stage} ===="
    if [[ -n "${stage}" && -f "${OUT_ROOT}/AUDIT/${stage}.log" ]]; then
      echo "---- tail ${OUT_ROOT}/AUDIT/${stage}.log ----"
      tail -n 200 "${OUT_ROOT}/AUDIT/${stage}.log" || true
    elif [[ -f "${OUT_ROOT}/AUDIT/run_status.json" ]]; then
      cat "${OUT_ROOT}/AUDIT/run_status.json"
    fi
    echo
  } >> "${FAIL_CONTEXT}"
}

apply_minimal_runtime_fix() {
  local stage="$1"
  if [[ "${stage}" == "stage_verify_ds004584_full" || "${stage}" == "clinical_ds004584_fullN" ]]; then
    (
      cd "${DATA_ROOT}/ds004584" || exit 0
      datalad get -r -J 8 . >> "${FAIL_CONTEXT}" 2>&1 || true
      git annex get -J 8 -- sub-*/eeg/* >> "${FAIL_CONTEXT}" 2>&1 || true
      git annex enableremote s3-PUBLIC public=yes >> "${FAIL_CONTEXT}" 2>&1 || true
    )
  elif [[ "${stage}" == "stage_verify_ds007020_full" || "${stage}" == "clinical_ds007020_mortality_fixbeta" ]]; then
    (
      cd "${DATA_ROOT}/ds007020" || exit 0
      datalad get -r -J 8 . >> "${FAIL_CONTEXT}" 2>&1 || true
      git annex get -J 8 -- sub-*/**/eeg/* >> "${FAIL_CONTEXT}" 2>&1 || true
    )
  elif [[ "${stage}" == "bio_D_cross_modality_ds004752_repair" ]]; then
    (
      cd "${REPO_ROOT}" || exit 0
      python scripts/decode_ds004752.py \
        --dataset_root "${DATA_ROOT}/ds004752" \
        --out_yaml "${OUT_ROOT}/PACK_BIO_V3_1/BIO_D_cross_modality/event_map_ds004752.yaml" \
        --out_summary "${OUT_ROOT}/PACK_BIO_V3_1/BIO_D_cross_modality/decode_ds004752_summary.json" \
        --out_candidate "${OUT_ROOT}/PACK_BIO_V3_1/BIO_D_cross_modality/decode_ds004752_candidates.csv" \
        --stop_reason "${OUT_ROOT}/PACK_BIO_V3_1/BIO_D_cross_modality/STOP_REASON_ds004752.md" >> "${FAIL_CONTEXT}" 2>&1 || true
    )
  elif [[ "${stage}" == "workload_ds007262_repair" ]]; then
    (
      cd "${REPO_ROOT}" || exit 0
      python scripts/decode_ds007262.py \
        --dataset_root "${DATA_ROOT}/ds007262" \
        --out_yaml "${OUT_ROOT}/PACK_BIO_V3_1/workload_ds007262/event_map_ds007262.yaml" \
        --out_summary "${OUT_ROOT}/PACK_BIO_V3_1/workload_ds007262/decode_ds007262_summary.json" \
        --out_candidate "${OUT_ROOT}/PACK_BIO_V3_1/workload_ds007262/decode_ds007262_candidates.csv" \
        --stop_reason "${OUT_ROOT}/PACK_BIO_V3_1/workload_ds007262/STOP_REASON_ds007262.md" >> "${FAIL_CONTEXT}" 2>&1 || true
    )
  fi
}

attempt=1
while [[ ${attempt} -le ${MAX_ATTEMPTS} ]]; do
  if [[ ${attempt} -eq 1 ]]; then
    resume_flag="false"
  else
    resume_flag="true"
  fi

  set +e
  run_once "${resume_flag}"
  rc=$?
  set -e

  status="$(status_from_json)"
  err="$(error_from_json)"

  if [[ "${status}" == "PASS_STRICT" ]]; then
    echo "PASS_STRICT achieved on attempt ${attempt}" | tee -a "${FAIL_CONTEXT}"
    exit 0
  fi

  stage="$(failing_stage_from_json)"
  if [[ -z "${stage}" && "${status}" == "PARTIAL_PASS" ]]; then
    if [[ "${err}" == *"ds004584"* ]]; then
      stage="clinical_ds004584_fullN"
    elif [[ "${err}" == *"ds007020"* ]]; then
      stage="clinical_ds007020_mortality_fixbeta"
    elif [[ "${err}" == *"nvidia_smi_1hz"* ]]; then
      stage="rebundle_submission_packet"
    fi
  fi

  append_fail_context "${attempt}" "${status}" "${stage}"
  apply_minimal_runtime_fix "${stage}"

  if [[ "${status}" == "PARTIAL_PASS" && "${err}" == *"walltime exhausted"* ]]; then
    echo "PARTIAL_PASS due to walltime exhaustion on attempt ${attempt}" | tee -a "${FAIL_CONTEXT}"
    exit 0
  fi

  attempt=$((attempt + 1))
  sleep 2
done

final_status="$(status_from_json)"
if [[ "${final_status}" == "PARTIAL_PASS" ]]; then
  echo "PARTIAL_PASS after ${MAX_ATTEMPTS} attempts" | tee -a "${FAIL_CONTEXT}"
  exit 0
fi
echo "FAILED after ${MAX_ATTEMPTS} attempts" | tee -a "${FAIL_CONTEXT}"
exit 1
