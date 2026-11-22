"""Download the TOP benchmark labels (Fu et al., 2022, "HINT", Patterns).

Saves the nine phase_{I,II,III}_{train,valid,test}.csv files (~40 MB) to
data/external/top/. They are used only to check our labels against an
independent, published label set (src/label_report.py).

Usage:
    python src/fetch_top.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

BASE = "https://raw.githubusercontent.com/futianfan/clinical-trial-outcome-prediction/main/data"
FILES = [f"phase_{p}_{s}.csv" for p in ("I", "II", "III") for s in ("train", "valid", "test")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/external/top", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = args.out / name
        if dest.exists():
            print(f"have {dest}")
            continue
        r = requests.get(f"{BASE}/{name}", timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"saved {dest} ({len(r.content) // 1024} KB)")


if __name__ == "__main__":
    main()
