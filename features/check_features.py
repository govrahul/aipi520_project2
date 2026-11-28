"""Validate modeling feature tables and optionally impute remaining missing values.

Usage:
    python features/check_features.py
    python features/check_features.py --fix
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BLOCKED_OUTPUT_NAMES = {
    "overall_status",
    "why_stopped",
    "enrollment_count",
    "enrollment_type",
    "start_date_type",
    "primary_completion_date",
    "completion_date",
    "results_first_post_date",
    "last_update_post_date",
    "status_verified_date",
    "has_results",
    "n_primary_pvalues",
    "n_primary_sig",
    "primary_min_pvalue",
}

LABELS = ["label_strict", "label_broad", "label_hybrid", "label_pvalue"]
IDENTITY = ["nct_id", "split"]


def impute_missing(df: pd.DataFrame, medians: pd.Series) -> tuple[pd.DataFrame, dict[str, float]]:
    counts = df.isna().sum()
    counts = counts[counts > 0]
    fixed = df.copy()
    values: dict[str, float] = {}
    for column in counts.index:
        if column in IDENTITY or column in LABELS:
            continue
        median = medians.get(column, 0.0)
        if not np.isfinite(median):
            median = 0.0
        fixed[column] = fixed[column].fillna(median)
        values[column] = float(median)
    return fixed, values


def validate(folder: Path, fix: bool) -> dict:
    frames: dict[str, pd.DataFrame] = {}
    schema: set[str] | None = None
    train_medians = None

    for split in ("train", "val", "test"):
        path = folder / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        if split == "train":
            train_medians = df.drop(columns=IDENTITY + LABELS).median(numeric_only=True)

        if fix:
            df, _ = impute_missing(df, train_medians)
            df.to_parquet(path, index=False)

        frames[split] = df
        if schema is None:
            schema = set(df.columns)
        elif set(df.columns) != schema:
            raise AssertionError(f"{split} schema differs from train schema")

        if df["nct_id"].duplicated().any():
            raise AssertionError(f"{split} has duplicate nct_id values")
        if df["split"].nunique() != 1 or df["split"].iloc[0] != split:
            raise AssertionError(f"{split} contains the wrong split value")
        if df["label_broad"].isna().any():
            raise AssertionError(f"{split} has missing label_broad values")
        if (df["label_broad"] < 0).any() or (df["label_broad"] > 1).any():
            raise AssertionError(f"{split} has label_broad values outside 0/1")

        collision = set(df.columns) & BLOCKED_OUTPUT_NAMES
        if collision:
            raise AssertionError(f"{split} contains blocked columns: {sorted(collision)}")

        numeric = df.drop(columns=IDENTITY + LABELS).select_dtypes(include=[np.number])
        if numeric.isna().any().any():
            raise AssertionError(f"{split} still has missing feature values")
        if not np.isfinite(numeric.to_numpy()).all():
            raise AssertionError(f"{split} contains NaN or infinite feature values")

    return {
        "rows": {name: len(df) for name, df in frames.items()},
        "feature_count": len(frames["train"].columns) - len(IDENTITY) - len(LABELS),
        "schema_columns": len(frames["train"].columns),
        "fixed": fix,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default=Path(__file__).resolve().parent, type=Path)
    ap.add_argument("--fix", action="store_true", help="fill remaining feature missingness using train medians")
    args = ap.parse_args()

    report = validate(args.folder.resolve(), args.fix)
    print("Feature validation passed")
    print(report)


if __name__ == "__main__":
    main()
