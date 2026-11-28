# AIPI 520 Project 2

This repository builds a leakage-audited ClinicalTrials.gov dataset and creates
modeling-ready training, validation, and test matrices for predicting whether a
trial succeeds.

The primary modeling target is `label_broad`:

- `1`: completed, or terminated early for clear benefit
- `0`: terminated for safety, efficacy/futility, or DSMB reasons

The repository uses the committed 2026-11-20 snapshot by default. It does not
require a new internet download for normal feature generation.

## Repository layout

- `src/fetch.py`: downloads the raw ClinicalTrials.gov snapshot
- `src/parse.py`: flattens raw JSONL pages into Parquet trial records
- `src/label.py`: creates labels, cohort membership, and temporal splits
- `src/split.py`: defines the train/validation/test date ranges
- `src/feature_policy.py`: defines blocked, caution, and allowed source fields
- `src/censoring_check.py`: checks censoring by start year
- `src/leakage_audit.py`: generates feature and text leakage evidence
- `src/label_report.py`: generates label and TOP benchmark reports
- `features/build_features.py`: creates modeling matrices
- `features/check_features.py`: validates modeling matrices
- `features/FEATURES.md`: documents every feature group and modeling experiment
- `data/processed/trials/`: committed parsed trial Parquet shards
- `data/processed/labels.parquet`: committed labels and temporal splits
- `reports/`: generated audit and report Markdown/CSV files

## Requirements

- Python 3.11 or newer
- Internet access for the initial TOP benchmark download or `--refresh`
- The dependencies in `requirements.txt`

## First-time setup

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd <repository-directory>
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository already includes parsed trials, labels, and reports. A normal
reproduction does not require downloading the raw snapshot again.

## Reproduce the processed data and reports

Run:

```bash
python run_pipeline.py
```

This performs the default steps:

1. Downloads missing TOP benchmark CSV files to `data/external/top/`
2. Rebuilds `data/processed/labels.parquet`
3. Regenerates `reports/labels.md`
4. Regenerates `reports/leakage.md`

The existing trial Parquet shards and `data/processed/snapshot.json` are used
as inputs.

Run a single pipeline stage when necessary:

```bash
python run_pipeline.py --only fetch_top
python run_pipeline.py --only label
python run_pipeline.py --only label_report
python run_pipeline.py --only leakage_audit
```

## Refresh from ClinicalTrials.gov (Optional)

Use this only when a new snapshot is needed:

```bash
python run_pipeline.py --refresh
```

This downloads raw trial pages, reparses them, recalculates censoring, rebuilds
labels/reports, and downloads TOP data if missing. A raw refresh is roughly
600 MB and can take around 10 minutes.

The fetch step resumes through `data/raw/manifest.json`. To intentionally start
over from an entirely new download, remove `data/raw/` before running refresh.
Do not delete the committed processed snapshot unless you intend to replace it.

The refresh pipeline runs:

1. `src/fetch.py`
2. `src/parse.py`
3. `src/censoring_check.py`
4. `src/fetch_top.py`
5. `src/label.py`
6. `src/label_report.py`
7. `src/leakage_audit.py`

## Build the modeling features

After labels are available:

```bash
python features/build_features.py
python features/check_features.py
```

Generated files:

- `features/train.parquet`
- `features/val.parquet`
- `features/test.parquet`
- `features/build_metadata.json`

For optional CSV copies:

```bash
python features/build_features.py --write-csv
```

This creates `train.csv`, `val.csv`, and `test.csv`.

All feature vocabularies, category lists, numeric medians, and text vocabularies
are fitted from the training split. Validation and test rows are transformed
using these training-fitted values.

## Temporal split

The cohort excludes withdrawn trials and trials outside the reproducible cohort
window:

- Training: start years 2005–2012
- Validation: start years 2013–2014
- Testing: start years 2015–2016

Rows are assigned in `src/split.py`. Modeling must preserve these splits; do not
shuffle or combine them for primary tuning.

## Tests

Run the existing regression tests:

```bash
python -m pytest tests/
```

Run feature validation separately:

```bash
python features/check_features.py
```

## Modeling workflow

1. Load `train.parquet`, `val.parquet`, and `test.parquet`.
2. Exclude `nct_id` and `split` from predictors.
3. Use `label_broad` as the primary modeling target.
4. Fit preprocessing and models on training data only.
5. Tune on validation data only.
6. Evaluate the test split once after model and feature choices are fixed.
7. Report at least ROC-AUC and average precision.
8. Add cautious text or caution-tier features only as documented ablations.

See [features/FEATURES.md](features/FEATURES.md) for feature definitions,
generation details, leakage decisions, and staged modeling recommendations.

---
Developers: Rahul Govil, Qingyi Hu, Arshaan Sayed