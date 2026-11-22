"""Flatten the raw API pages in ``data/raw/`` into one table, one row per trial.

Output: ``data/processed/trials/part-NN.parquet`` (read the folder with
``pd.read_parquet("data/processed/trials")``) and ``data/processed/snapshot.json``
(query and database timestamp). Text fields stay raw; list fields (conditions,
countries, MeSH terms, ...) are stored as lists.

Some columns describe what happened *after* the trial started. They are kept
for labeling and analysis; src/feature_policy.py says which columns may be
used as model features.

Usage:
    python src/parse.py
    python src/parse.py --raw data/raw --out data/processed/trials
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

AGE_UNITS = {"year": 1.0, "month": 1 / 12, "week": 7 / 365.25, "day": 1 / 365.25, "hour": 1 / 8766, "minute": 1 / 525960}
AGE_RE = re.compile(r"^\s*([\d.]+)\s*([a-zA-Z]+?)s?\s*$")
PV_RE = re.compile(r"^\s*(<=|>=|<|>|=|≤|≥)?\s*(\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")


def age_years(s: str | None) -> float | None:
    if not s:
        return None
    m = AGE_RE.match(s)
    if not m or m.group(2).lower() not in AGE_UNITS:
        return None
    return float(m.group(1)) * AGE_UNITS[m.group(2).lower()]


def parse_pvalue(s: str | None) -> tuple[str, float] | None:
    """'<0.001' -> ('<', 0.001); '0.38' -> ('=', 0.38); unparseable -> None."""
    if not s:
        return None
    m = PV_RE.match(s.replace(" ", ""))
    if not m:
        return None
    op = {"≤": "<=", "≥": ">=", None: "="}.get(m.group(1), m.group(1))
    return op, float(m.group(2))


def is_significant(op: str, val: float, alpha: float = 0.05) -> bool | None:
    if op == "<":
        return val <= alpha
    if op in ("=", "<="):
        return val < alpha
    if op in (">", ">="):
        return False if val >= alpha else None  # '>0.01' says nothing about 0.05
    return None


def to_date(s: str | None) -> pd.Timestamp | None:
    """API dates are 'YYYY', 'YYYY-MM' or 'YYYY-MM-DD'; month precision becomes the 1st."""
    if not s:
        return None
    parts = s.split("-") + ["01", "01"]
    try:
        return pd.Timestamp(f"{parts[0]}-{parts[1]}-{parts[2]}")
    except ValueError:
        return None


def flatten(study: dict) -> dict:
    p = study.get("protocolSection", {})
    d = study.get("derivedSection", {})
    r = study.get("resultsSection", {})

    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    spons = p.get("sponsorCollaboratorsModule", {})
    over = p.get("oversightModule", {})
    desc = p.get("descriptionModule", {})
    cond = p.get("conditionsModule", {})
    design = p.get("designModule", {})
    dinfo = design.get("designInfo", {})
    mask = dinfo.get("maskingInfo", {})
    arms = p.get("armsInterventionsModule", {})
    outc = p.get("outcomesModule", {})
    elig = p.get("eligibilityModule", {})
    locs = p.get("contactsLocationsModule", {})

    def date_of(key: str) -> tuple:
        st = status.get(key) or {}
        return to_date(st.get("date")), st.get("type")

    start, start_type = date_of("startDateStruct")
    pcomp, pcomp_type = date_of("primaryCompletionDateStruct")
    comp, comp_type = date_of("completionDateStruct")
    first_post, _ = date_of("studyFirstPostDateStruct")
    res_post, _ = date_of("resultsFirstPostDateStruct")
    last_post, _ = date_of("lastUpdatePostDateStruct")

    phases = design.get("phases") or []
    collaborators = spons.get("collaborators") or []
    arm_groups = arms.get("armGroups") or []
    interventions = arms.get("interventions") or []
    primary = outc.get("primaryOutcomes") or []
    locations = locs.get("locations") or []
    countries = sorted({loc["country"] for loc in locations if loc.get("country")})
    cmesh = d.get("conditionBrowseModule", {})
    imesh = d.get("interventionBrowseModule", {})

    # p-values of statistical analyses attached to PRIMARY outcomes (results section)
    pvals = []
    for om in (r.get("outcomeMeasuresModule") or {}).get("outcomeMeasures", []):
        if om.get("type") != "PRIMARY":
            continue
        for an in om.get("analyses") or []:
            pv = parse_pvalue(an.get("pValue"))
            if pv:
                pvals.append(pv)
    sig = [is_significant(op, v) for op, v in pvals]
    bounded = [v for op, v in pvals if op in ("=", "<", "<=")]

    return {
        # identity
        "nct_id": ident.get("nctId"),
        "brief_title": ident.get("briefTitle"),
        "official_title": ident.get("officialTitle"),
        "org_class": (ident.get("organization") or {}).get("class"),
        # status / dates
        "overall_status": status.get("overallStatus"),
        "why_stopped": status.get("whyStopped"),
        "start_date": start,
        "start_date_type": start_type,
        "primary_completion_date": pcomp,
        "primary_completion_date_type": pcomp_type,
        "completion_date": comp,
        "completion_date_type": comp_type,
        "first_post_date": first_post,
        "results_first_post_date": res_post,
        "last_update_post_date": last_post,
        "status_verified_date": to_date(status.get("statusVerifiedDate")),
        # sponsors / oversight
        "lead_sponsor": (spons.get("leadSponsor") or {}).get("name"),
        "lead_sponsor_class": (spons.get("leadSponsor") or {}).get("class"),
        "responsible_party_type": (spons.get("responsibleParty") or {}).get("type"),
        "n_collaborators": len(collaborators),
        "collaborator_classes": sorted({c.get("class") for c in collaborators if c.get("class")}),
        "has_dmc": over.get("oversightHasDmc"),
        "fda_regulated_drug": over.get("isFdaRegulatedDrug"),
        "fda_regulated_device": over.get("isFdaRegulatedDevice"),
        # text
        "brief_summary": desc.get("briefSummary"),
        "detailed_description": desc.get("detailedDescription"),
        "conditions": cond.get("conditions") or [],
        "keywords": cond.get("keywords") or [],
        # design
        "study_type": design.get("studyType"),
        "phase": "/".join(sorted(phases)),
        "allocation": dinfo.get("allocation"),
        "intervention_model": dinfo.get("interventionModel"),
        "primary_purpose": dinfo.get("primaryPurpose"),
        "masking": mask.get("masking"),
        "who_masked": mask.get("whoMasked") or [],
        "enrollment_count": (design.get("enrollmentInfo") or {}).get("count"),
        "enrollment_type": (design.get("enrollmentInfo") or {}).get("type"),
        # arms / interventions
        "n_arms": len(arm_groups),
        "arm_types": [a.get("type") for a in arm_groups if a.get("type")],
        "n_interventions": len(interventions),
        "intervention_types": [i.get("type") for i in interventions if i.get("type")],
        "intervention_names": [i.get("name") for i in interventions if i.get("name")],
        # outcomes (protocol, i.e. planned)
        "n_primary_outcomes": len(primary),
        "n_secondary_outcomes": len(outc.get("secondaryOutcomes") or []),
        "n_other_outcomes": len(outc.get("otherOutcomes") or []),
        "primary_outcome_measures": [o.get("measure") for o in primary if o.get("measure")],
        "primary_outcome_time_frames": [o.get("timeFrame") for o in primary if o.get("timeFrame")],
        # eligibility
        "eligibility_criteria": elig.get("eligibilityCriteria"),
        "healthy_volunteers": elig.get("healthyVolunteers"),
        "sex": elig.get("sex"),
        "minimum_age": elig.get("minimumAge"),
        "maximum_age": elig.get("maximumAge"),
        "min_age_years": age_years(elig.get("minimumAge")),
        "max_age_years": age_years(elig.get("maximumAge")),
        "std_ages": elig.get("stdAges") or [],
        # sites
        "n_locations": len(locations),
        "countries": countries,
        "n_countries": len(countries),
        "has_us_site": "United States" in countries,
        "n_overall_officials": len(locs.get("overallOfficials") or []),
        # MeSH (derived by the registry from conditions / interventions)
        "condition_mesh_terms": [m.get("term") for m in cmesh.get("meshes") or []],
        "condition_mesh_ancestors": [m.get("term") for m in cmesh.get("ancestors") or []],
        "intervention_mesh_terms": [m.get("term") for m in imesh.get("meshes") or []],
        "intervention_mesh_ancestors": [m.get("term") for m in imesh.get("ancestors") or []],
        # results (post-outcome)
        "has_results": bool(study.get("hasResults")),
        "n_primary_pvalues": len(pvals),
        "n_primary_sig": sum(1 for s in sig if s),
        "primary_min_pvalue": min(bounded) if bounded else None,
    }


def parse_file(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [flatten(json.loads(line)) for line in f]


def write_shards(df: pd.DataFrame, out_dir: Path, rows_per_shard: int) -> list[Path]:
    """Write df as out_dir/part-NN.parquet (zstd). Shards stay well under GitHub's 50 MB warning size,
    so the table can live in the repo without Git LFS. Read it back with pd.read_parquet(out_dir)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    paths = []
    for i, start in enumerate(range(0, len(df), rows_per_shard)):
        p = out_dir / f"part-{i:02d}.parquet"
        df.iloc[start:start + rows_per_shard].to_parquet(p, index=False, compression="zstd", compression_level=9)
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw", type=Path)
    ap.add_argument("--out", default="data/processed/trials", type=Path, help="output folder of parquet shards")
    ap.add_argument("--rows-per-shard", default=25000, type=int)
    args = ap.parse_args()

    manifest = json.loads((args.raw / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise SystemExit("raw download is incomplete; run src/fetch.py first")
    files = [args.raw / p["file"] for p in manifest["pages"]]

    with ProcessPoolExecutor() as ex:
        rows = [row for chunk in ex.map(parse_file, files) for row in chunk]
    df = pd.DataFrame(rows)

    n_before = len(df)
    df = df.drop_duplicates("nct_id").sort_values("nct_id").reset_index(drop=True)

    paths = write_shards(df, args.out, args.rows_per_shard)
    snapshot = {k: manifest[k] for k in ("data_timestamp", "api_version", "query", "expected_total")}
    snapshot["n_trials"] = len(df)
    (args.out.parent / "snapshot.json").write_text(json.dumps(snapshot, indent=1))

    mb = sum(p.stat().st_size for p in paths) / 2**20
    print(f"{len(df)} trials ({n_before - len(df)} duplicates dropped), {df.shape[1]} columns "
          f"-> {args.out}/ ({len(paths)} shards, {mb:.0f} MB)")
    print(f"snapshot: {manifest['data_timestamp']}  expected: {manifest['expected_total']}")


if __name__ == "__main__":
    main()
