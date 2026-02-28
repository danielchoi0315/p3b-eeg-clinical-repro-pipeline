# EEG Cognitive-Clinical Reproducibility Pipeline

End-to-end, fail-closed pipeline for EEG confirmatory analyses, robustness checks, clinical calibration, and publication packaging.

## What this repository provides

- Deterministic confirmatory reruns from pinned OpenNeuro commits.
- Strict expected-value matching against an explicit expected kit.
- Pre-specified robustness grid and reliability/attenuation checks.
- Clinical stability and calibration outputs.
- Publication artifacts:
  - manuscript package zip
  - Overleaf package zip
  - Prism artist pack zip
  - results-only tarball

## Repository layout

- `scripts/bulletproof/`: staged orchestration (`stage0` ... `stage9`) and master runner.
- `src/p3b_pipeline/`: core processing modules.
- `configs/`: dataset and run configuration files.
- `scripts/`: helper runners and auto-resume wrappers.

## Environment

- Linux with Python 3.10+
- `datalad` + `git-annex` for dataset staging
- Slurm recommended for large runs

Install Python deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the bulletproof pipeline

From repository root:

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

bash scripts/bulletproof/run_master.sh \
  --out_root /filesystemHcog/runs/$(date +%Y%m%d_%H%M%S)_REPRO_BULLETPROOF_MASTER_RUN \
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

- `AUDIT/BULLETPROOF_AUDIT_REPORT.md`
- `OUTZIP/MANUSCRIPT_KIT_UPDATED.zip`
- `OUTZIP/OVERLEAF_UPDATED.zip`
- `OUTZIP/PRISM_ARTIST_PACK.zip`
- `TARBALLS/results_only.tar.gz`

## Reproducibility requirements

- Confirmatory stages are deterministic via seed registry.
- Stage1 enforces exact dataset commit checkout.
- Stage3 compares only confirmatory endpoints with strict tolerances.
- Every skip/fail writes explicit `STOP_REASON*.md`.

## Notes

Legacy script names are preserved for backward compatibility, but user-facing artifacts and defaults are now neutral and publication-ready.
