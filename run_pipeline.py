"""Run the data pipeline end to end.

    fetch          download trials from ClinicalTrials.gov      -> data/raw/          (~10 min, ~600 MB)
    parse          flatten to one row per trial                -> data/processed/trials.parquet
    fetch_top      download TOP benchmark labels               -> data/external/top/
    censoring      count still-running trials per start year   -> reports/censoring_by_year.csv
    label          labels, cohort, temporal split              -> data/processed/labels.parquet
    label_report   label counts, audit, agreement with TOP     -> reports/labels.md
    leakage_audit  evidence for the feature policy             -> reports/leakage.md

Usage:
    python run_pipeline.py                  # all steps (fetch resumes / skips if complete)
    python run_pipeline.py --from label     # start at a later step
    python run_pipeline.py --only parse     # one step
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["fetch", "parse", "fetch_top", "censoring", "label", "label_report", "leakage_audit"]
SCRIPTS = {"censoring": "censoring_check"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0])
    g.add_argument("--only", choices=STEPS)
    args = ap.parse_args()

    steps = [args.only] if args.only else STEPS[STEPS.index(args.start):]
    for step in steps:
        script = ROOT / "src" / f"{SCRIPTS.get(step, step)}.py"
        print(f"\n=== {step} ===", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
