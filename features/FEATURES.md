# Modeling features

This directory contains leakage-safe modeling datasets built from the committed processed trial table.

## Run

```bash
python features/build_features.py
python features/check_features.py

# Optional: also write CSV files
python features/build_features.py --write-csv
```

Use `train.parquet`, `val.parquet`, and `test.parquet` for modeling. Each row is one in-cohort trial and includes:

- `nct_id`
- `split`
- `label_strict`
- `label_broad`
- `label_hybrid`
- `label_pvalue`

`label_broad` is the recommended primary target. It assigns completed trials and clear-benefit early stops to 1, and safety, efficacy/futility, and DSMB terminations to 0. Withdrawn and unrelated early-termination trials are excluded from the in-cohort target or remain missing where specified.

## Files

- `build_features.py`: merges trials and labels, derives train-fitted encodings, writes the three split files.
- `check_features.py`: verifies schemas, uniqueness, finite values, labels, and absence of blocked columns. Run with `--fix` to fill remaining missing feature values using training medians.
- `build_metadata.json`: records the training-fitted categories and text vocabularies.
- `train.parquet`, `val.parquet`, `test.parquet`: modeling matrices.

## Feature groups

| Feature group | How it is made | Leakage tier |
|---|---|---|
| `start_year` | Calendar year derived from `start_date` | Allowed |
| Numeric design fields | `n_collaborators`, `n_arms`, `n_interventions`, outcome counts | Allowed or caution for some outcome counts |
| Minimum/maximum age | Numeric values from `min_age_years` and `max_age_years` | Allowed |
| Age and eligibility engineering | `age_range_years`, `adult_only`, `pediatric_only`, eligibility word count | Allowed |
| Low-cardinality categories | One-hot indicators for phase, allocation, model, purpose, masking, sponsorship, design, and sponsor class | Allowed |
| List fields | Top 20-100 training categories converted to binary indicators | Allowed |
| Sponsor | One-hot indicators for the 100 most frequent training sponsor names and 50 most frequent sponsor classes; rare names become `__other__` | Allowed |
| Text | Scrubbed brief title, official title, summary, detailed description, and eligibility text vectorized with 1-2 gram TF-IDF | Allowed for titles/eligibility; cautioned for summary/description |
| Missing indicators | Binary indicators for missing numeric and categorical source values | Derivation of allowed source fields |

Numeric missing values are filled with the training-split median. Categorical missing values receive an explicit `__missing__` category. List fields use only categories learned from the training split. Text vocabularies are also learned from the training split only.

## Individual source-feature audit

| Source feature | Output transformation | Policy | Leakage reasoning |
|---|---|---|---|
| `start_date` | `start_year`, missing indicator | Allowed | Used only as the calendar year; original post-start/end dates are not used |
| `phase` | 50-category one-hot, plus other/missing | Allowed | Planned trial design |
| `allocation` | 50-category one-hot, plus missing | Allowed | Planned randomization |
| `intervention_model` | 50-category one-hot, plus missing | Allowed | Planned study design |
| `primary_purpose` | 50-category one-hot, plus missing | Allowed | Planned purpose |
| `masking` | 50-category one-hot, plus missing | Allowed | Planned masking |
| `healthy_volunteers` | 50-category one-hot, plus missing | Allowed | Eligibility design |
| `sex` | 50-category one-hot, plus missing | Allowed | Eligibility design |
| `lead_sponsor` | 100-category one-hot; rare values become `__other__` | Allowed | Identity known at trial start; no target/success-rate encoding is used |
| `lead_sponsor_class` | 50-category one-hot, plus missing | Allowed | Sponsor type known at trial start |
| `responsible_party_type` | 50-category one-hot, plus missing | Allowed | Study registration information |
| `has_dmc` | 50-category one-hot, plus missing | Allowed | Protocol oversight |
| `org_class` | 50-category one-hot, plus missing | Allowed | Trial organization type |
| `study_type` | 50-category one-hot, plus missing | Allowed | Registration design field |
| `n_collaborators` | Median-filled numeric plus missing indicator | Allowed | Known from protocol/registration |
| `n_arms` | Median-filled numeric plus missing indicator | Allowed | Planned arm count |
| `n_interventions` | Median-filled numeric plus missing indicator | Allowed | Planned intervention count |
| `n_primary_outcomes` | Median-filled numeric plus missing indicator | Allowed | Planned outcome structure |
| `n_secondary_outcomes` | Median-filled numeric plus missing indicator | Caution | Outcome lists may be revised after results are posted |
| `n_other_outcomes` | Median-filled numeric plus missing indicator | Caution | Outcome lists may be revised after results are posted |
| `min_age_years`, `max_age_years` | Numeric medians, range, adult/pediatric indicators, missing indicators | Allowed | Eligibility criteria |
| `arm_types`, `who_masked`, `std_ages` | Training-top category binary indicators | Allowed | Planned arm, masking, and age-group structure |
| `collaborator_classes` | Training-top category binary indicators | Allowed | Collaborator data known from registration |
| `intervention_types`, `intervention_names` | Training-top binary indicators | Allowed | Intervention identity known from the planned protocol |
| `conditions`, `keywords` | Training-top binary indicators plus count features | Allowed | Planned disease and keyword metadata |
| `primary_outcome_measures`, `primary_outcome_time_frames` | Training-top binary indicators | Allowed | Planned protocol outcomes |
| `condition_mesh_terms`, `condition_mesh_ancestors` | Training-top binary indicators | Allowed | Derived disease hierarchy; no post-trial result field |
| `intervention_mesh_terms`, `intervention_mesh_ancestors` | Training-top binary indicators | Allowed | Derived intervention hierarchy |
| `brief_title`, `official_title` | TF-IDF after `scrub_text` | Allowed | Post-hoc sentences are removed; title features still have lower-audit priority |
| `brief_summary` | TF-IDF after `scrub_text` | Caution | Latest record may be edited after outcome; post-hoc sentences are removed |
| `detailed_description` | TF-IDF after `scrub_text` | Caution | Latest record may contain termination/results text; post-hoc sentences are removed |
| `eligibility_criteria` | TF-IDF plus word count | Allowed | Eligibility is protocol information |

## Leakage exclusions

The builder calls `check_source_columns` before fitting features. It does not copy or derive from:

- `overall_status`, `why_stopped`
- `enrollment_count`, `enrollment_type`
- Trial completion/primary-completion dates and date precision fields
- `first_post_date`, `results_first_post_date`, `last_update_post_date`, `status_verified_date`
- `has_results`, p-value counts, p-value significance, and minimum p-value
- Trial duration or registration-lag derived fields

The generated outputs contain only identity/metadata, allowed or cautioned source-derived features, and labels. Caution features should be ablated rather than assumed harmless:

- `n_secondary_outcomes`
- `n_other_outcomes`
- Scrubbed `brief_summary`
- Scrubbed `detailed_description`

## Generated dataset audit

The current generated matrices pass [check_features.py](check_features.py):

| Split | Rows | Labelled `label_broad` rows | Success rate |
|---|---:|---:|---:|
| Train | 34,641 | 34,641 | 0.847 |
| Validation | 9,456 | 9,456 | 0.855 |
| Test | 9,465 | 9,465 | 0.842 |

Checks also enforce:

- identical feature schemas across train, validation, and test
- unique `nct_id`
- exactly one split value per table
- `label_broad` present and in `[0, 1]`
- no blocked source-column names
- no missing or infinite numeric features

## Recommended starting points for modeling

The 2,095 output columns are a superset intended for experiments. A good modeling
run should begin smaller and add feature groups deliberately rather than fitting
all columns at once.

### 1. Tabular baseline

Start with the numeric and low-cardinality categorical features:

- `start_year` and the engineered age fields
- Phase, allocation, intervention model, purpose, masking, sponsor class,
  organization class, responsible-party type, and DMC status
- Arm, intervention, and primary-outcome counts
- The short binary list features for arm types and masking
- Missing-value indicators

A regularized logistic regression or gradient-boosted tree is a suitable baseline.
Scale numeric columns before logistic regression. For tree models, scaling is
unnecessary.

Treat `nct_id` and `split` as metadata only. Do not include them as predictors.

### 2. Add disease and intervention representations

Next, add the binary indicators derived from:

- Conditions
- Keywords
- Intervention types and names
- MeSH condition terms and ancestors
- MeSH intervention terms and ancestors

Keep the 100-category limits for this baseline unless a separate sparse-feature
experiment is intentional. Fit logistic regression with L1 or L2 regularization
as the first sparse model. Compare it with a tree model if project requirements
include interpretable non-linear interactions.

### 3. Add sponsor identity separately

Test sponsor features in a separate experiment:

- `lead_sponsor` one-hot indicators
- `lead_sponsor_class`
- Collaborator classes

Sponsor identity can encode recurring organizations and programs. It may improve
accuracy, but it can also make temporal generalization weaker for unseen sponsors.
Compare the run with sponsor features against the tabular baseline on the
validation split.

### 4. Test text cautiously

Text features should be tested as another ablation:

- Start with eligibility text
- Next add scrubbed titles
- Add scrubbed summaries and detailed descriptions only in a later sensitivity
  run

The existing leakage audit found raw summary/description text to be stronger than
scrubbed text at validation time, so test the caution that text was transformed
but not assumed safe. Text may still reflect latest-record edits, even after
post-hoc sentence scrubbing.

The 1,000 text columns can be useful for linear sparse models. They are unlikely
to be the best first dense-tree representation.

### 5. Caution-feature ablation

Run the following features only in explicit experiments:

- `n_secondary_outcomes`
- `n_other_outcomes`
- Scrubbed `brief_summary`
- Scrubbed `detailed_description`

Before and after metrics, compare validation AUC and average precision. Keep a
caution feature only if it improves performance robustly and is justified as
information available at trial start.

## Labels and split use

Use `label_broad` as the first target:

- Train on `train.parquet`
- Tune preprocessing and hyperparameters on `val.parquet`
- Evaluate `test.parquet` only after the model and feature set are fixed

Recommended reporting includes:

- ROC-AUC
- Average precision
- Sensitivity and specificity
- Calibration and Brier score
- Metrics by phase and start year

Do not tune against the test split. `label_hybrid` is a useful sensitivity
target, but it is more dependent on p-value posting and should not replace
`label_broad` in the primary analysis.

## Leakage-safe modeling rules

- Fit all encoders, TF-IDF vocabularies, imputation values, and scaling parameters
  on the training split only.
- Do not use target encoding or outcome-derived sponsor statistics unless they
  are generated out of fold within the training split.
- Do not include blocked fields or post-trial dates.
- Do not recreate trial duration or registration-lag features.
- Run [check_features.py](check_features.py) again whenever feature generation
  or source data changes.

A practical experiment order is:

1. Tabular baseline
2. Tabular baseline plus conditions/interventions/MeSH
3. Previous run plus sponsor identity
4. Previous run plus eligibility text
5. Caution-feature and summary/description ablations
