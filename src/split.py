"""Temporal train/validation/test split by trial start year.

The model predicts a trial's outcome when it starts, so it is trained on older
trials and tested on newer ones. A random split would let the model learn from
trials that ran at the same time as the test trials (same sponsors, same drug
programs, same standard of care) and overstate performance.

The cohort stops at 2016: for later start years, 8-15%+ of trials are still
running at the snapshot (reports/censoring_by_year.csv), so the finished ones
we can see are biased toward short, terminated trials.
"""

from __future__ import annotations

import pandas as pd

SPLITS: dict[str, tuple[int, int]] = {
    "train": (2005, 2012),
    "val": (2013, 2014),
    "test": (2015, 2016),
}
START_MIN = min(lo for lo, _ in SPLITS.values())
START_MAX = max(hi for _, hi in SPLITS.values())


def outcome_known_by(trials: pd.DataFrame, date: str | pd.Timestamp) -> pd.Series:
    """True where the trial had ended (primary completion, else completion) before ``date``.

    Training labels can resolve after the test period starts (14% of train trials
    end after 2015-01-01). Features such as drug or sponsor identity can then carry
    knowledge that was not available when a test trial started. Training only on
    rows where this is True simulates a model built on ``date``; use it as a
    sensitivity check. Trials with no end date count as not known.
    """
    end = trials["primary_completion_date"].fillna(trials["completion_date"])
    return end < pd.Timestamp(date)


def assign_split(start_year: pd.Series) -> pd.Series:
    """Map start years to 'train' / 'val' / 'test'; anything outside SPLITS -> None."""
    out = pd.Series(None, index=start_year.index, dtype=object)
    for name, (lo, hi) in SPLITS.items():
        out[start_year.between(lo, hi)] = name
    return out
