# P3b EEG-Clinical Reproducibility Pipeline

End-to-end, fail-closed pipeline for deterministic EEG confirmatory analyses, robustness checks, reliability diagnostics, and clinical calibration.

## What this repository provides

- Deterministic confirmatory reruns from pinned OpenNeuro commits.
- Strict expected-value matching against an explicit expected kit.
- Pre-specified robustness and reliability/attenuation analyses.
- Clinical stability and calibration outputs.
- Packaged run outputs for reporting, reuse, and audit trails.

## Repository layout

- `scripts/repro/`: public entrypoints for orchestration.
- `scripts/bulletproof/`: legacy implementation modules retained for compatibility.
- `src/p3b_pipeline/`: core processing modules.
- `configs/`: dataset and run configuration files.
- `scripts/`: helper runners and auto-resume wrappers.

## Environment

- Linux with Python 3.10+
- `datalad` + `git-annex` for dataset staging
- Slurm recommended for large runs

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline

From repository root:

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

bash scripts/repro/run_master.sh \
  --out_root /filesystemHcog/runs/$(date +%Y%m%d_%H%M%S)_REPRO_MASTER_RUN \
  --expected_kit /absolute/path/to/EXPECTED_KIT.zip \
  --wall_hours 12 \
  --max_attempts 10
```

## Expected kit contract

Stage0 requires an explicit expected kit and fails closed if incomplete.
The kit must provide:

- `dataset_hashes.json` with required confirmatory dataset commits
- non-empty confirmatory metrics table (`expected_confirmatory_metrics.csv` or compatible manuscript table)

## Main outputs

Under `<OUT_ROOT>`:

- `AUDIT/AUDIT_REPORT.md` (primary audit summary)
- `AUDIT/CONFIRMATORY_MATCH_REPORT.md`
- `OUTZIP/MANUSCRIPT_KIT_UPDATED.zip`
- `OUTZIP/OVERLEAF_UPDATED.zip`
- `OUTZIP/PRISM_ARTIST_PACK.zip`
- `TARBALLS/results_only.tar.gz`

Compatibility note:

- `AUDIT/BULLETPROOF_AUDIT_REPORT.md` is still written as a legacy alias.

## Reproducibility requirements

- Confirmatory stages are deterministic via seed registry.
- Stage1 enforces exact dataset commit checkout.
- Stage3 compares confirmatory endpoints with strict tolerances.
- Every skip/fail writes explicit `STOP_REASON*.md`.
