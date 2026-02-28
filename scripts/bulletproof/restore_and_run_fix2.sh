#!/usr/bin/env bash
set -euo pipefail

RUNS_ROOT="${RUNS_ROOT:-/lambda/nfs/HCog/filesystemHcog/runs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "${RUNS_ROOT}/_INCOMING"

cd "${REPO_ROOT}"

python3 scripts/bulletproof/restore_canonical_runs.py --runs_root "${RUNS_ROOT}"

python3 - "${RUNS_ROOT}" <<'PY'
import fnmatch
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1]).resolve()
patterns = {
    "canonical_v2": "*_NN_FINAL_MEGA_V2_BIO*",
    "master_v1": "*_NN_FINAL_MASTER_V1*",
    "postfinal_tighten": "*_POSTFINAL_TIGHTEN*",
}

out = {}
missing = []
for key, pat in patterns.items():
    cands = [p.resolve() for p in runs_root.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, pat)]
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        missing.append(key)
    else:
        out[key] = str(cands[0])

if missing:
    stop = runs_root / "STOP_REASON_restore_missing_archives.md"
    if not stop.exists():
        stop.write_text(
            "# STOP_REASON restore_missing_archives\n\n"
            "## Why\n"
            "Required canonical roots are still missing after restore.\n\n"
            "## Diagnostics\n"
            "```json\n"
            + json.dumps({"missing": missing, "runs_root": str(runs_root)}, indent=2)
            + "\n```\n",
            encoding="utf-8",
        )
    print(str(stop))
    raise SystemExit(1)

pointers = runs_root / "_CANONICAL_POINTERS.json"
pointers.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
print(str(pointers))
print(json.dumps(out, indent=2))
PY

POINTERS_JSON="${RUNS_ROOT}/_CANONICAL_POINTERS.json"
CANONICAL_V2="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["canonical_v2"])' "${POINTERS_JSON}")"
MASTER_V1="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["master_v1"])' "${POINTERS_JSON}")"
POSTFINAL_TIGHTEN="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["postfinal_tighten"])' "${POINTERS_JSON}")"

EXPECTED_KIT_ZIP="${RUNS_ROOT}/$(date +%Y%m%d_%H%M%S)_EXPECTEDKIT.zip"
python3 scripts/bulletproof/build_expected_kit.py \
  --out_zip "${EXPECTED_KIT_ZIP}" \
  --canonical_v2 "${CANONICAL_V2}" \
  --master_v1 "${MASTER_V1}" \
  --postfinal_tighten "${POSTFINAL_TIGHTEN}"

python3 - "${EXPECTED_KIT_ZIP}" "${RUNS_ROOT}" <<'PY'
import csv
import json
import sys
import tempfile
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1]).resolve()
runs_root = Path(sys.argv[2]).resolve()
stop = runs_root / "STOP_REASON_expectedkit_build_failed.md"

if not zip_path.exists():
    stop.write_text(
        "# STOP_REASON expectedkit_build_failed\n\n"
        "## Why\nExpectedKit zip was not created.\n\n"
        f"Missing file: {zip_path}\n",
        encoding="utf-8",
    )
    print(str(stop))
    raise SystemExit(1)

with tempfile.TemporaryDirectory(prefix="expectedkit_validate_") as td:
    tdp = Path(td)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tdp)

    metrics = tdp / "expected_confirmatory_metrics.csv"
    dsh = tdp / "dataset_hashes.json"

    errs = []
    n_rows = 0
    if not metrics.exists():
        errs.append("missing_expected_confirmatory_metrics.csv")
    else:
        with metrics.open("r", encoding="utf-8", newline="") as f:
            n_rows = max(0, sum(1 for _ in csv.reader(f)) - 1)
        if n_rows < 25:
            errs.append(f"rowcount_lt_25:{n_rows}")

    if not dsh.exists():
        errs.append("missing_dataset_hashes.json")
    else:
        try:
            payload = json.loads(dsh.read_text(encoding="utf-8"))
            rows = payload.get("datasets", []) if isinstance(payload.get("datasets"), list) else []
            if not rows:
                errs.append("dataset_hashes_empty")
        except Exception as exc:
            errs.append(f"dataset_hashes_parse_error:{exc}")

if errs:
    stop.write_text(
        "# STOP_REASON expectedkit_build_failed\n\n"
        "## Why\nExpectedKit validation failed.\n\n"
        "## Diagnostics\n"
        "```json\n"
        + json.dumps({"expected_kit_zip": str(zip_path), "n_rows": n_rows, "errors": errs}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(str(stop))
    raise SystemExit(1)

print(json.dumps({"expected_kit_zip": str(zip_path), "row_count": n_rows}, indent=2))
PY

NEW_OUT_ROOT="${RUNS_ROOT}/$(date +%Y%m%d_%H%M%S)_REPRO_BULLETPROOF_MASTER_RUN_FIX2"
bash scripts/bulletproof/run_master.sh \
  --out_root "${NEW_OUT_ROOT}" \
  --expected_kit "${EXPECTED_KIT_ZIP}" \
  --wall_hours 12 \
  --max_attempts 10

echo "NEW_OUT_ROOT=${NEW_OUT_ROOT}"
echo "AUDIT_REPORT=${NEW_OUT_ROOT}/AUDIT/BULLETPROOF_AUDIT_REPORT.md"
if [[ -f "${NEW_OUT_ROOT}/DIFF_REPORT.md" ]]; then
  echo "DIFF_REPORT=${NEW_OUT_ROOT}/DIFF_REPORT.md"
fi
echo "MANUSCRIPT_ZIP=${NEW_OUT_ROOT}/OUTZIP/MANUSCRIPT_KIT_UPDATED.zip"
echo "OVERLEAF_ZIP=${NEW_OUT_ROOT}/OUTZIP/OVERLEAF_UPDATED.zip"
echo "PRISM_ZIP=${NEW_OUT_ROOT}/OUTZIP/PRISM_ARTIST_PACK.zip"
echo "RESULTS_TARBALL=${NEW_OUT_ROOT}/TARBALLS/results_only.tar.gz"
echo "RESULTS_TARBALL_SHA=${NEW_OUT_ROOT}/TARBALLS/results_only.tar.gz.sha256"
