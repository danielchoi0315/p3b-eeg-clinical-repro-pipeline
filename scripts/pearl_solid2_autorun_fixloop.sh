#!/usr/bin/env bash
set -euo pipefail

MAX_ATTEMPTS=6
WALL_HOURS=10
OUT_ROOT=""
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
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${OUT_ROOT}" ]]; then
  OUT_ROOT="/filesystemHcog/runs/$(date +%Y%m%d_%H%M%S)_PEARL_SOLID2"
fi

mkdir -p "${OUT_ROOT}/AUDIT" "${OUT_ROOT}/OUTZIP"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

attempt=1
resume_flag=""
rc=1

while [[ "${attempt}" -le "${MAX_ATTEMPTS}" ]]; do
  echo "[pearl_fixloop] attempt ${attempt}/${MAX_ATTEMPTS} out_root=${OUT_ROOT}"
  set +e
  cmd=(python scripts/pearl_solid2_runner.py --out_root "${OUT_ROOT}" --wall_hours "${WALL_HOURS}")
  if [[ -n "${resume_flag}" ]]; then
    cmd+=("${resume_flag}")
  fi
  cmd+=("${EXTRA_ARGS[@]}")
  "${cmd[@]}"
  rc=$?
  set -e

  if [[ "${rc}" -eq 0 ]]; then
    echo "[pearl_fixloop] success"
    break
  fi

  fail_ctx="${OUT_ROOT}/AUDIT/FAIL_CONTEXT.txt"
  {
    echo "## attempt=${attempt} ts=$(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc}"
    newest_log="$(ls -1t "${OUT_ROOT}/AUDIT/"*.log 2>/dev/null | head -n 1 || true)"
    if [[ -n "${newest_log}" && -f "${newest_log}" ]]; then
      echo "Newest log: ${newest_log}"
      tail -n 200 "${newest_log}" || true
    else
      echo "No stage logs found."
    fi
    echo
  } >> "${fail_ctx}"

  # Minimal non-science runtime fixes between retries.
  if [[ "${attempt}" -eq 1 ]]; then
    export HDF5_USE_FILE_LOCKING=FALSE
  elif [[ "${attempt}" -eq 2 ]]; then
    EXTRA_ARGS+=(--cpu_workers 16)
  elif [[ "${attempt}" -eq 3 ]]; then
    EXTRA_ARGS+=(--gpu_parallel_procs 6)
  fi

  resume_flag="--resume"
  attempt=$((attempt + 1))
done

if [[ "${rc}" -ne 0 ]]; then
  echo "[pearl_fixloop] failed after ${MAX_ATTEMPTS} attempts (rc=${rc})"
  exit "${rc}"
fi

report="${OUT_ROOT}/AUDIT/PEARL_SOLID2_REPORT.md"
zip_path="${OUT_ROOT}/OUTZIP/pearl_solid2_bundle.zip"
echo "OUT_ROOT=${OUT_ROOT}"
echo "REPORT=${report}"
echo "BUNDLE=${zip_path}"
