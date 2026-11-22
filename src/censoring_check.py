"""How many trials of each start year have not reached a final status yet?

A trial still running at the snapshot is missing from our data. If many trials
of a start year are still running, the ones we do have for that year are biased
toward short (and terminated) trials. This counts, per start year, trials with
the same filters as src/fetch.py in each status group, via the API.

Output: reports/censoring_by_year.csv

Usage:
    python src/censoring_check.py --first 2005 --last 2022
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from fetch import API, DEFAULT_ADVANCED

GROUPS = {
    "final": "COMPLETED,TERMINATED",
    "withdrawn": "WITHDRAWN",
    "ongoing": "RECRUITING,NOT_YET_RECRUITING,ACTIVE_NOT_RECRUITING,ENROLLING_BY_INVITATION,SUSPENDED",
    "unknown": "UNKNOWN",
}


def count(session: requests.Session, year: int, statuses: str) -> int:
    params = {
        "filter.advanced": f"{DEFAULT_ADVANCED} AND AREA[StartDate]RANGE[{year}-01-01,{year}-12-31]",
        "filter.overallStatus": statuses,
        "countTotal": "true",
        "pageSize": 1,
        "fields": "NCTId",
    }
    r = session.get(f"{API}/studies", params=params, timeout=60)
    r.raise_for_status()
    return r.json()["totalCount"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", default=2005, type=int)
    ap.add_argument("--last", default=2022, type=int)
    ap.add_argument("--out", default="reports/censoring_by_year.csv", type=Path)
    args = ap.parse_args()

    s = requests.Session()
    rows = []
    for year in range(args.first, args.last + 1):
        row = {"start_year": year}
        for name, st in GROUPS.items():
            row[name] = count(s, year, st)
            time.sleep(0.5)
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    known = df.final + df.ongoing
    df["ongoing_share"] = (df.ongoing / known).round(4)          # still running, among trials with a known status
    df["unknown_share"] = (df.unknown / (known + df.unknown)).round(4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
