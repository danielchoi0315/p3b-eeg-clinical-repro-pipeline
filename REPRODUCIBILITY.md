# Reproducibility Guide

This document describes the minimum steps required to reproduce audited results from this repository.

## 1) Prepare environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install system tools (`datalad`, `git-annex`) if not available.

## 2) Provide an explicit expected kit

The expected kit zip is mandatory and must include:

- `dataset_hashes.json`
- non-empty confirmatory metrics table (`expected_confirmatory_metrics.csv` or compatible table)

## 3) Run full audited pipeline

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

bash scripts/bulletproof/run_master.sh \
  --out_root /filesystemHcog/runs/$(date +%Y%m%d_%H%M%S)_REPRO_BULLETPROOF_MASTER_RUN \
  --expected_kit /absolute/path/to/EXPECTED_KIT.zip \
  --wall_hours 12 \
  --max_attempts 10
```

## 4) Verify final artifacts

Expected paths under `<OUT_ROOT>`:

- `AUDIT/BULLETPROOF_AUDIT_REPORT.md`
- `AUDIT/CONFIRMATORY_MATCH_REPORT.md`
- `OUTZIP/MANUSCRIPT_KIT_UPDATED.zip`
- `OUTZIP/OVERLEAF_UPDATED.zip`
- `OUTZIP/PRISM_ARTIST_PACK.zip`
- `TARBALLS/results_only.tar.gz`

## 5) Integrity checks

```bash
sha256sum <OUT_ROOT>/TARBALLS/results_only.tar.gz
```

If any stage fails, inspect `STOP_REASON*.md` and stage logs in `<OUT_ROOT>/AUDIT/`.
