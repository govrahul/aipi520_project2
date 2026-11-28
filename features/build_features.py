"""Build leakage-safe modeling tables for train, validation, and test splits.

Usage:
    python features/build_features.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from feature_policy import check_source_columns, scrub_text  # noqa: E402


CATEGORICAL = [
    "phase",
    "allocation",
    "intervention_model",
    "primary_purpose",
    "masking",
    "healthy_volunteers",
    "sex",
    "lead_sponsor_class",
    "responsible_party_type",
    "has_dmc",
    "org_class",
    "study_type",
    "lead_sponsor",
]

NUMERIC = [
    "n_collaborators",
    "n_arms",
    "n_interventions",
    "n_primary_outcomes",
    "n_secondary_outcomes",
    "n_other_outcomes",
    "min_age_years",
    "max_age_years",
]

LIST_FEATURES = {
    "arm_types": 30,
    "who_masked": 20,
    "std_ages": 20,
    "collaborator_classes": 20,
    "intervention_types": 30,
    "intervention_names": 100,
    "conditions": 100,
    "keywords": 100,
    "primary_outcome_measures": 100,
    "primary_outcome_time_frames": 50,
    "condition_mesh_terms": 100,
    "condition_mesh_ancestors": 100,
    "intervention_mesh_terms": 100,
    "intervention_mesh_ancestors": 100,
}

TEXT_FEATURES = {
    "brief_title": 100,
    "official_title": 150,
    "brief_summary": 250,
    "detailed_description": 300,
    "eligibility_criteria": 200,
}

SOURCE_COLUMNS = CATEGORICAL + NUMERIC + list(LIST_FEATURES) + list(TEXT_FEATURES) + [
    "start_date",
]

LABEL_COLUMNS = ["label_strict", "label_broad", "label_hybrid", "label_pvalue"]


def flattened_list(series: pd.Series) -> list[str]:
    values: list[str] = []
    for item in series:
        if item is None:
            continue
        if isinstance(item, (list, tuple, np.ndarray)):
            values.extend(str(v).strip() for v in item if v is not None and str(v).strip())
        else:
            text = str(item).strip()
            if text:
                values.append(text)
    return values


def top_categories(series: pd.Series, limit: int) -> list[str]:
    counts = Counter(flattened_list(series))
    return [value for value, _ in counts.most_common(limit)]


def add_missing_indicators(df: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC:
        df[f"{column}__missing"] = source[column].isna().astype("int8")
    for column in CATEGORICAL:
        df[f"{column}__missing"] = source[column].isna().astype("int8")
    return df


def build(base: pd.DataFrame, train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Create numeric/tabular features. `train` supplies every fitted vocabulary."""
    metadata: dict[str, object] = {
        "categorical": {},
        "list": {},
        "numeric_medians": {},
        "text": {},
    }
    out = pd.DataFrame(index=base.index)

    for column in NUMERIC:
        median = float(pd.to_numeric(train[column], errors="coerce").median())
        if not np.isfinite(median):
            median = 0.0
        metadata["numeric_medians"][column] = median
        values = pd.to_numeric(base[column], errors="coerce").fillna(median)
        out[column] = values.astype("float32")

    start_year = base["start_date"].dt.year
    train_start_year = train["start_date"].dt.year
    start_year_median = float(train_start_year.median())
    out["start_year"] = start_year.fillna(start_year_median).astype("float32")
    out["start_year__missing"] = start_year.isna().astype("int8")
    age_range = base["max_age_years"] - base["min_age_years"]
    train_age_range = train["max_age_years"] - train["min_age_years"]
    age_range_median = float(train_age_range.median())
    if not np.isfinite(age_range_median):
        age_range_median = 0.0
    metadata["numeric_medians"]["age_range_years"] = age_range_median
    out["age_range_years"] = age_range.fillna(age_range_median).astype("float32")
    out["age_range_years__missing"] = age_range.isna().astype("int8")
    out["adult_only"] = ((base["min_age_years"].fillna(0) >= 18) & base["std_ages"].apply(
        lambda v: "ADULT" in v if isinstance(v, (list, tuple, np.ndarray)) else True
    )).astype("int8")
    out["pediatric_only"] = ((base["max_age_years"].fillna(999) < 18)).astype("int8")
    out["n_conditions"] = base["conditions"].map(lambda v: len(v) if isinstance(v, (list, tuple, np.ndarray)) else 0).astype("int16")
    out["n_keywords"] = base["keywords"].map(lambda v: len(v) if isinstance(v, (list, tuple, np.ndarray)) else 0).astype("int16")
    out["eligibility_word_count"] = base["eligibility_criteria"].fillna("").str.split().str.len().astype("int16")
    out["brief_title_word_count"] = base["brief_title"].fillna("").str.split().str.len().astype("int16")

    for column in CATEGORICAL:
        values = base[column].fillna("__missing__").astype(str)
        train_values = train[column].fillna("__missing__").astype(str)
        limit = 100 if column == "lead_sponsor" else 50
        top = train_values.value_counts().head(limit).index.tolist()
        metadata["categorical"][column] = top
        values = values.where(values.isin(top), "__other__")
        categories = top + ["__other__", "__missing__"]
        for category in categories:
            safe = category.replace("/", "_").replace(" ", "_").replace("-", "_")
            out[f"{column}={safe}"] = (values == category).astype("int8")

    for column, limit in LIST_FEATURES.items():
        categories = top_categories(train[column], limit)
        metadata["list"][column] = categories
        for category in categories:
            safe = category.replace("/", "_").replace(" ", "_").replace("-", "_")
            out[f"{column}={safe}"] = base[column].map(
                lambda v, c=category: int(
                    isinstance(v, (list, tuple, np.ndarray)) and c in {str(x).strip() for x in v}
                )
            ).astype("int8")

    for column, limit in TEXT_FEATURES.items():
        text = base[column].fillna("").map(scrub_text)
        train_text = train[column].fillna("").map(scrub_text)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            min_df=5,
            max_features=limit,
            sublinear_tf=True,
        )
        vectorizer.fit(train_text)
        matrix = vectorizer.transform(text)
        for i, term in enumerate(vectorizer.get_feature_names_out()):
            safe = term.replace(" ", "_").replace("/", "_")
            out[f"text_{column}={safe}"] = matrix[:, i].toarray().ravel().astype("float32")
        metadata["text"][column] = {
            "max_features": limit,
            "n_features": len(vectorizer.get_feature_names_out()),
            "scrubbed": True,
        }

    out = add_missing_indicators(out, base)
    out["nct_id"] = base["nct_id"].astype(str)
    for column in LABEL_COLUMNS:
        out[column] = base[column].astype("float32")
    out["split"] = base["split"].astype(str)
    return out, metadata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", default=ROOT / "data/processed/trials", type=Path)
    ap.add_argument("--labels", default=ROOT / "data/processed/labels.parquet", type=Path)
    ap.add_argument("--out", default=ROOT / "features", type=Path)
    ap.add_argument("--write-csv", action="store_true", help="also write train/val/test CSV files")
    args = ap.parse_args()

    trials = pd.read_parquet(args.trials)
    labels = pd.read_parquet(args.labels)
    labels = labels.drop(columns=["phase"], errors="ignore")
    frame = trials.merge(labels, on="nct_id", validate="one_to_one")
    frame = frame[frame["in_cohort"]].copy()
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce")

    check_source_columns(SOURCE_COLUMNS)
    train = frame[frame["split"].eq("train")].copy()
    if train.empty:
        raise SystemExit("No training rows found")

    args.out.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        features, metadata = build(frame[frame["split"].eq(split)].copy(), train)
        datasets[split] = features
        features.to_parquet(args.out / f"{split}.parquet", index=False)
        if args.write_csv:
            features.to_csv(args.out / f"{split}.csv", index=False)
        print(f"{split}: {len(features)} rows, {len(features.columns) - 7} feature/metadata columns")

    (args.out / "build_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Wrote modeling datasets to {args.out}")


if __name__ == "__main__":
    main()
