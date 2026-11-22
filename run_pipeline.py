"""Run the data pipeline.

The repo already contains the processed data (data/processed/, snapshot of
ClinicalTrials.gov taken 2026-11-20; see data/processed/snapshot.json), so the
default run starts from it and needs no download besides the small TOP files:

    fetch_top      download TOP benchmark labels (~40 MB)      -> data/external/top/
    label          labels, cohort, temporal split              -> data/processed/labels.parquet
    label_report   label counts, audit, agreement with TOP     -> reports/labels.md
    leakage_audit  evidence for the feature policy             -> reports/leakage.md

--refresh first rebuilds the data from a NEW snapshot (numbers will change;
fetch resumes an existing data/raw/ download, so delete that folder to start over):

    fetch          download trials from ClinicalTrials.gov      -> data/raw/          (~10 min, ~600 MB)
    parse          flatten to one row per trial                -> data/processed/trials/ + snapshot.json
    censoring      count still-running trials per start year   -> reports/censoring_by_year.csv

Usage:
    python run_pipeline.py                  # from the committed data
    python run_pipeline.py --refresh        # new snapshot, then everything
    python run_pipeline.py --only label     # one step
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFRESH_STEPS = ["fetch", "parse", "censoring"]
DEFAULT_STEPS = ["fetch_top", "label", "label_report", "leakage_audit"]
SCRIPTS = {"censoring": "censoring_check"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--refresh", action="store_true", help="download a new snapshot first")
    g.add_argument("--only", choices=REFRESH_STEPS + DEFAULT_STEPS)
    args = ap.parse_args()

    steps = [args.only] if args.only else (REFRESH_STEPS if args.refresh else []) + DEFAULT_STEPS
    for step in steps:
        script = ROOT / "src" / f"{SCRIPTS.get(step, step)}.py"
        print(f"\n=== {step} ===", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
