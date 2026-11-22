"""Download a filtered snapshot of ClinicalTrials.gov through the v2 API.

Each API page is saved as one gzipped JSON-lines file (one study per line) in
``data/raw/``. ``manifest.json`` records the query, the database snapshot
timestamp, and the page tokens, so an interrupted run resumes where it stopped.

Usage:
    python src/fetch.py                 # default filters (see DEFAULT_*)
    python src/fetch.py --out data/raw --page-size 1000
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import requests

API = "https://clinicaltrials.gov/api/v2"

# Interventional drug/biologic trials in phase 1-3 (phase 1/2 and 2/3 combos match too).
DEFAULT_ADVANCED = (
    "AREA[StudyType]INTERVENTIONAL"
    " AND AREA[Phase](PHASE1 OR PHASE2 OR PHASE3)"
    " AND AREA[InterventionType](DRUG OR BIOLOGICAL)"
)
# Only trials that have reached a final status can be labeled.
DEFAULT_STATUSES = "COMPLETED,TERMINATED,WITHDRAWN"


def get_with_retry(session: requests.Session, url: str, params: dict, tries: int = 6) -> requests.Response:
    for attempt in range(tries):
        try:
            r = session.get(url, params=params, timeout=180)
            if r.status_code == 200:
                return r
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on {url} after {tries} tries")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw", type=Path)
    ap.add_argument("--advanced", default=DEFAULT_ADVANCED, help="Essie expression for filter.advanced")
    ap.add_argument("--statuses", default=DEFAULT_STATUSES, help="comma-separated overall statuses")
    ap.add_argument("--page-size", default=1000, type=int, help="max 1000")
    ap.add_argument("--sleep", default=1.5, type=float, help="seconds between requests")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    query = {
        "filter.advanced": args.advanced,
        "filter.overallStatus": args.statuses,
        "pageSize": args.page_size,
    }

    session = requests.Session()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["query"] != query:
            raise SystemExit(f"{manifest_path} was made with a different query; use a new --out folder")
        if manifest.get("complete"):
            print(f"Already complete: {manifest['n_studies']} studies in {len(manifest['pages'])} pages")
            return
        print(f"Resuming after page {len(manifest['pages'])}")
    else:
        version = get_with_retry(session, f"{API}/version", {}).json()
        total = get_with_retry(session, f"{API}/studies", {**query, "pageSize": 1, "countTotal": "true", "fields": "NCTId"}).json()["totalCount"]
        manifest = {
            "query": query,
            "api_version": version.get("apiVersion"),
            "data_timestamp": version.get("dataTimestamp"),
            "expected_total": total,
            "pages": [],  # [{"file", "n", "next_token"}]
            "n_studies": 0,
            "complete": False,
        }
        print(f"Snapshot {manifest['data_timestamp']}: {total} studies to fetch")

    token = manifest["pages"][-1]["next_token"] if manifest["pages"] else None
    while True:
        params = dict(query)
        if token:
            params["pageToken"] = token
        page = get_with_retry(session, f"{API}/studies", params).json()
        studies = page.get("studies", [])
        idx = len(manifest["pages"])
        fname = f"page_{idx:04d}.jsonl.gz"
        with gzip.open(args.out / fname, "wt", encoding="utf-8") as f:
            for s in studies:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        token = page.get("nextPageToken")
        manifest["pages"].append({"file": fname, "n": len(studies), "next_token": token})
        manifest["n_studies"] += len(studies)
        manifest["complete"] = token is None
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"page {idx}: {len(studies)} studies (total {manifest['n_studies']}/{manifest['expected_total']})", flush=True)

        if token is None:
            break
        time.sleep(args.sleep)

    print(f"Done: {manifest['n_studies']} studies, snapshot {manifest['data_timestamp']}")


if __name__ == "__main__":
    main()
