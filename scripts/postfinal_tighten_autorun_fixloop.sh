#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MAX_ATTEMPTS=6

OUT_ROOT=""
DATA_ROOT="/filesystemHcog/openneuro"
CANONICAL_ROOT="/filesystemHcog/runs/20260223_185511_NN_FINAL_MASTER_V1"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out_root)
      OUT_ROOT="$2"; shift 2 ;;
    --data_root)
      DATA_ROOT="$2"; shift 2 ;;
    --canonical_root)
      CANONICAL_ROOT="$2"; shift 2 ;;
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
  python scripts/postfinal_tighten_runner.py \
    --out_root "${OUT_ROOT}" \
    --data_root "${DATA_ROOT}" \
    --canonical_root "${CANONICAL_ROOT}" \
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
  if [[ "${stage}" == "ds004584_retrieve_missing" || "${stage}" == "ds004584_rerun_pdrest_endpoints" ]]; then
    (
      cd "${DATA_ROOT}/ds004584" || exit 0
      shopt -s globstar nullglob
      datalad get -r -J 8 sub-*/**/eeg/* >> "${FAIL_CONTEXT}" 2>&1 || true
      git annex get -J 8 -- sub-*/**/eeg/* >> "${FAIL_CONTEXT}" 2>&1 || true
    )
  elif [[ "${stage}" == "ds004752_one_shot_repair" ]]; then
    (
      cd "${REPO_ROOT}" || exit 0
      python scripts/decode_ds004752.py \
        --dataset_root "${DATA_ROOT}/ds004752" \
        --out_yaml "${OUT_ROOT}/PACK_BIO_CROSSMODALITY_TIGHTEN/event_map_ds004752.yaml" \
        --out_summary "${OUT_ROOT}/PACK_BIO_CROSSMODALITY_TIGHTEN/decode_ds004752_summary.json" \
        --out_candidate "${OUT_ROOT}/PACK_BIO_CROSSMODALITY_TIGHTEN/decode_ds004752_candidates.csv" \
        --stop_reason "${OUT_ROOT}/PACK_BIO_CROSSMODALITY_TIGHTEN/STOP_REASON_ds004752.md" >> "${FAIL_CONTEXT}" 2>&1 || true
    )
  fi
  python -m py_compile scripts/postfinal_tighten_runner.py >> "${FAIL_CONTEXT}" 2>&1 || true
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
  stage="$(failing_stage_from_json)"

  if [[ "${status}" == "PASS" || "${status}" == "PARTIAL_PASS" ]]; then
    echo "completed with status=${status} on attempt ${attempt}" | tee -a "${FAIL_CONTEXT}"
    break
  fi

  append_fail_context "${attempt}" "${status}" "${stage}"
  apply_minimal_runtime_fix "${stage}"

  attempt=$((attempt + 1))
  sleep 2
done

final_status="$(status_from_json)"
tar_path="${OUT_ROOT}/TARBALLS/results_only.tar.gz"
echo "OUT_ROOT=${OUT_ROOT}"
echo "TARBALL=${tar_path}"
echo "SCP=scp <user>@<host>:\"${tar_path}\" ."

if [[ "${final_status}" == "PASS" || "${final_status}" == "PARTIAL_PASS" ]]; then
  exit 0
fi
exit 1
