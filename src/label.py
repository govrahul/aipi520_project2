"""Build outcome labels from overall_status + why_stopped.

Input:  data/processed/trials/          (src/parse.py)
Output: data/processed/labels.parquet   one row per trial, joinable on nct_id

Label definitions
-----------------
The free-text ``why_stopped`` of each terminated trial is sorted into one
category by keyword rules (``classify_reason``; checked on a held-out manual
sample of 200, see reports/labels.md).

* COMPLETED                                   -> 1
* TERMINATED, stopped early for clear benefit -> 1
* TERMINATED for safety, efficacy/futility, or
  on a data monitoring board's recommendation -> 0
* TERMINATED for any other reason (enrollment, business, funding, logistics,
  unclear, or no reason given):
      label_strict -> NaN (dropped: not a scientific failure)
      label_broad  -> 0   (every early stop counts as a failure)
* WITHDRAWN (never enrolled)                  -> NaN in both

``label_pvalue`` is a side label for completed trials with posted results:
1 if any primary-outcome analysis has p < 0.05, else 0. It covers only ~10% of
completed trials and is used to check the "completed = success" assumption.

``label_hybrid`` = ``label_pvalue`` where it exists, else ``label_broad``.
Caveat: only trials that posted results can be "completed but failed", so this
label is noisier for groups that rarely post results.

``in_cohort`` marks the trials used for modeling (see ``cohort_mask``) and
``split`` assigns them to train / val / test by start year (see src/split.py).

Usage:
    python src/label.py
    python src/label.py --start-min 2005 --start-max 2016
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from split import START_MAX, START_MIN, assign_split

# ---------------------------------------------------------------------------
# why_stopped classifier (keyword rules, applied in priority order)
# ---------------------------------------------------------------------------
CATEGORIES = ["early_benefit", "safety", "efficacy", "enrollment", "business", "funding", "logistics", "dsmb", "other", "missing"]
FAILURE_CATEGORIES = {"safety", "efficacy", "dsmb"}

_flags = re.IGNORECASE
PATTERNS: dict[str, re.Pattern] = {
    "early_benefit": re.compile(
        r"(stopp|terminat|halt|end)\w*\s+(early\s+)?(for|due to|because of)\s+(overwhelming\s+|positive\s+|clear\s+)?(efficacy|benefit)"
        r"|primary\s+(end\s*point|objective)\w*\s+(has\s+been\s+|was\s+|were\s+)?(successfully\s+)?(achieved|met|reached)"
        r"|positive\s+interim\s+(results|efficacy|analysis)"
        r"|(dramatic|clear|overwhelming|substantial|significant)\s+benefits?\s+(\w+\s+){0,2}(seen|observed|demonstrated|shown)",
        _flags,
    ),
    "safety": re.compile(
        r"safety|toxicit|\btoxic|adverse\s+(event|effect|reaction|drug)|side[\s-]effect|\bS?AEs?\b"
        r"|benefit[\s/:-]*(to[\s-])?risk|risk[\s/:-]*(to[\s-])?benefit|\bdeaths?\b|\bdied\b|\bfatal|mortality"
        r"|hepatotox|liver\s+(injury|enzyme)|cardiotox|tolerab|intolera|unacceptable|\bharm|class\s+effect"
        r"|thrombocytopenia|neutropenia|transaminase|bleeding|ha?emorrhag|occlusion|\bevents?\s+(were|was)\s+observed",
        _flags,
    ),
    "efficacy": re.compile(
        r"futil|lack\s+of\s+(\w+\s+){0,3}(efficacy|effect|benefit|response|activity|signal|improvement)"
        # "no additional efficacy", "insufficient response" -- but not "no safety signal" / "no efficacy concerns"
        r"|\b(no|insufficient|limited|lower\s+than\s+(expected|anticipated)|not\s+sufficient|minimal|poor)\s+"
        r"((?!safety)\w+\s+){0,3}(efficacy|benefit|effect|response|activity|signal|improvement)"
        r"(?!\s*(concern|issue|reason|problem|finding|consideration)s?)"
        r"|(did\s+not|fail\w*\s+to)\s+(meet|achieve|reach|show|demonstrate)(?!\s+(\w+\s+){0,3}(enrol|accru|recruit|sample|participant|patient|subject))"
        r"|(not|never)\s+(meet|met|achieve|reach)\w*\s+((?!enrol|accru|recruit)\w+\s+){0,3}(end\s*point|objective)"
        r"|unlikely\s+to\s+(meet|achieve|succeed|demonstrate|show|reach)"
        r"|(treatment|virologic\w*|therapeutic)\s+failure|(less|not)\s+favou?rable\s+(efficacy|results?|outcome)"
        r"|(did\s+not|failed\s+to|does\s+not)\s+improve|(missing|lack|absence|no)\s+(of\s+)?evidence\s+of\s+(the\s+)?(efficacy|effectiveness|activity|benefit)"
        r"|(negative|disappointing|dismal|unfavou?rable)\s+(\w+\s+){0,2}(results?|data|outcome|findings)"
        r"|ineffective|not\s+effective|inefficac|interim\s+analys\w+\s+(showed|indicated|demonstrated|revealed)"
        r"|relapse\s+rate|progression\s+rate|stopping\s+rule|(primary|study)\s+end\s*point\s+(was\s+)?not"
        r"|(not|no\s+longer)\s+(be\s+)?(favou?r(able|ed|s)?|promising|warrant\w*|superior|working|justif\w*)"
        r"|(does|did|do)\s+not\s+(support|warrant|justify)\s+(further|continu|the\s+continu|development|proceed)"
        r"|no\s+(statistically\s+)?(significant\s+)?(clinical\s+)?(advantage|difference|improvement)|low\s+efficacy|discouraging"
        r"|(did\s+not|failed\s+to)\s+result\s+in\s+(an?\s+)?(statistically\s+)?significant|not\s+statistically\s+significant"
        r"|(end\s*points?|objectives?|criteria|efficacy)\s+(\w+\s+){0,2}not\s+met"
        r"|(results?|data|outcome)\s+(of|from)\s+(the\s+)?(an?\s+)?interim"
        r"|(based\s+on|per|following|after)\s+((the|an?|preliminary|initial|available|emerging|unblinded)\s+){0,3}"
        r"(interim|efficacy\s+(results|data)|results)",
        _flags,
    ),
    "enrollment": re.compile(
        r"recruit|rectruit|accru|acrual|acru|enrol|screen\s*fail|sample\s+size|inclusion|participation"
        r"|(few|no|low|insufficient|lack\s+of|not\s+enough)\s+(\w+\s+){0,2}(patients|participants|subjects|volunteers|cases|eligible|population)"
        r"|participant\s+availability",
        _flags,
    ),
    "business": re.compile(
        r"business|strateg|sponsor|company|portfolio|prioriti|commercial|corporate|management\s+decision|pipeline|divest"
        r"|(development|program|compound|product)\w*\s+(\w+\s+){0,3}(discontinu|stop|halt|terminat|suspend|cancel)"
        r"|(discontinu|stop|halt|terminat|suspend)\w*\s+(\w+\s+){0,4}(program|development)"
        r"|acquisition|merger|licens|partner|marketing|market\s+|focus|(not|no\s+longer)\s+(to\s+)?pursu",
        _flags,
    ),
    "funding": re.compile(r"fund|financ|budget|grant\b|money|resource|cost|capitali|support\s+(is\s+|was\s+)?no\s+longer", _flags),
    "logistics": re.compile(
        r"supply|manufactur|(un|no\s+longer\s+)availab|shortage|expir|investigator|\bPIs?\b|staff|left\s+the|relocat|\bsites?\b"
        r"|facilit|logist|operational|organi[sz]ation|regulat|FDA|EMA|IRB|ethic|clinical\s+hold|health\s+authorit|protocol|amend"
        r"|covid|pandemic|epidemic|feasib|administrat|technical|contract|standard\s+of\s+care|practice|obsolete|new\s+(drugs?|technology|treatments?)"
        r"|replac|rolled|continuation\s+study|new\s+study|another\s+study",
        _flags,
    ),
    # data monitoring board stop with no other stated reason: data-driven, counted as a failure
    "dsmb": re.compile(r"\bdsmb|\bd?smc\b|\bi?dmc\b|data\s+(and\s+)?safety\s+monitoring|data\s+monitoring\s+(committee|board)", _flags),
}

# Phrases that rule a reason OUT ("not due to safety", "no safety or efficacy concerns",
# "not related to any efficacy, safety or clinical concerns") are removed before the safety
# and efficacy rules run. Only these specific shapes count, so "was not continued due to
# toxicity" still reads as safety and "showed no efficacy" still reads as efficacy.
_TERM = r"(?:safety|efficacy|toxicit\w*|tolerab\w*|adverse\s+\w+|risks?|medical|clinical|scientific|benefit[\s/:-]*risk|risk[\s/:-]*benefit)"
_TERM_LIST = rf"{_TERM}(?:(?:\s*(?:,|/|and/or|\bor\b|\band\b|\bnor\b)\s*)+{_TERM})*"
_NEG = r"\b(?:no|not|never|nor|neither|without)\b"
_FILL = (r"(?:to|on|of|by|with|from|in|any|the|a|an|for|new|major|significant|specific|relation|was|were|is|been"
         r"|terminated|stopped|discontinued|halted|ended|closed)")
_CONNECT = r"(?:due|relat\w*|based|because|driven|associated|regarding|caused|result\w*|attributable|linked|motivated|prompted)"
_NEGATED_REASON = re.compile(
    rf"{_NEG}(?:\s+{_FILL})*\s+{_CONNECT}(?:\s+{_FILL})*\s+{_TERM_LIST}"                      # not due to any safety ...
    rf"|(?:{_NEG}(?:\s+(?:any|new|major|significant|specific|known|other|unexpected|serious))*\s+|\bnon[\s-]?){_TERM_LIST}"
    rf"\s*(?:concern|issue|reason|finding|signal|problem|consideration|related|matter|event|data|factor|information)s?"  # no safety concerns
    rf"|\bunrelated\s+to\s+(?:any\s+)?{_TERM_LIST}"
    rf"|\bnon[\s-]?{_TERM_LIST}",                                                                  # non-safety, non clinical
    _flags,
)
_NEGATION = re.compile(r"\b(no|not|without|unrelated|nor|neither|never)\b|n't", _flags)
_CLAUSE_SPLIT = re.compile(r"[.;,()\n]|\bbut\b|\bhowever\b", _flags)


def _drop_negated_clauses(text: str) -> str:
    return _NEGATED_REASON.sub(" ", text)


def classify_reason(text: str | None) -> tuple[str, dict[str, bool]]:
    """Return (primary category, per-category match flags) for one why_stopped string."""
    if text is None or (isinstance(text, float) and np.isnan(text)) or not str(text).strip():
        return "missing", {c: False for c in PATTERNS}
    text = str(text)
    scrubbed = _drop_negated_clauses(text)
    flags = {c: bool(p.search(scrubbed if c in ("safety", "efficacy") else text)) for c, p in PATTERNS.items()}
    # early benefit only counts in a clause with no negation ("did not show positive results")
    flags["early_benefit"] = any(
        PATTERNS["early_benefit"].search(cl) and not _NEGATION.search(cl) for cl in _CLAUSE_SPLIT.split(text) if cl
    )
    for c in PATTERNS:  # dict order = priority
        if flags[c]:
            return c, flags
    return "other", flags


# ---------------------------------------------------------------------------
# labels and cohort
# ---------------------------------------------------------------------------
def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["nct_id", "overall_status", "why_stopped", "phase", "start_date", "completion_date",
              "primary_completion_date", "first_post_date", "n_primary_pvalues", "n_primary_sig"]].copy()

    res = [classify_reason(t) if s == "TERMINATED" else (None, {}) for s, t in zip(df.overall_status, df.why_stopped)]
    out["stop_category"] = [c for c, _ in res]
    for c in PATTERNS:
        out[f"stop_{c}"] = [f.get(c, False) for _, f in res]

    completed = out.overall_status.eq("COMPLETED")
    terminated = out.overall_status.eq("TERMINATED")
    failure = terminated & out.stop_category.isin(FAILURE_CATEGORIES)
    benefit = terminated & out.stop_category.eq("early_benefit")

    out["label_strict"] = np.select([completed | benefit, failure], [1.0, 0.0], default=np.nan)
    out["label_broad"] = np.select([completed | benefit, terminated], [1.0, 0.0], default=np.nan)
    out["label_pvalue"] = np.where(completed & (out.n_primary_pvalues > 0), (out.n_primary_sig > 0).astype(float), np.nan)
    # label_broad, except completed trials with posted primary p-values use the p-value result
    out["label_hybrid"] = out.label_pvalue.fillna(out.label_broad)
    return out


def cohort_mask(lab: pd.DataFrame, start_min: int, start_max: int, snapshot: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """Return (in_cohort, exclusion_reason). The first failing rule is recorded."""
    year = lab.start_date.dt.year
    end = lab.primary_completion_date.fillna(lab.completion_date)
    rules = [
        ("withdrawn", lab.overall_status.eq("WITHDRAWN")),
        ("no_start_date", lab.start_date.isna()),
        (f"start_before_{start_min}", year < start_min),
        (f"start_after_{start_max}", year > start_max),
        # a missing completion date is fine (not used); an impossible one signals a bad record
        ("bad_completion_date", (lab.completion_date < lab.start_date) | (lab.completion_date > snapshot)),
        # registered only after the trial ended: the outcome was already known at registration
        ("registered_after_end", lab.first_post_date > end),
    ]
    reason = pd.Series(None, index=lab.index, dtype=object)
    for name, m in reversed(rules):  # reversed so the first rule wins
        reason[m] = name
    return reason.isna(), reason


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", default="data/processed/trials", type=Path)
    ap.add_argument("--out", default="data/processed/labels.parquet", type=Path)
    ap.add_argument("--start-min", default=START_MIN, type=int, help="first start year kept (registration became mandatory in 2005)")
    ap.add_argument("--start-max", default=START_MAX, type=int, help="last start year kept (later cohorts are not finished yet)")
    args = ap.parse_args()

    cols = ["nct_id", "overall_status", "why_stopped", "phase", "start_date", "completion_date", "primary_completion_date",
            "first_post_date", "n_primary_pvalues", "n_primary_sig"]
    df = pd.read_parquet(args.trials, columns=cols)
    snapshot = pd.Timestamp(json.loads((args.trials.parent / "snapshot.json").read_text())["data_timestamp"]).normalize()

    lab = make_labels(df)
    lab["in_cohort"], lab["exclusion_reason"] = cohort_mask(lab, args.start_min, args.start_max, snapshot)
    lab["start_year"] = lab.start_date.dt.year
    lab["split"] = assign_split(lab.start_year).where(lab.in_cohort)
    lab = lab.drop(columns=["start_date", "completion_date", "primary_completion_date", "first_post_date", "n_primary_pvalues", "n_primary_sig"])
    lab.to_parquet(args.out, index=False)

    c = lab[lab.in_cohort]
    print(f"{len(lab)} trials, {len(c)} in cohort ({args.start_min}-{args.start_max}) -> {args.out}")
    print("exclusions:", lab.exclusion_reason.value_counts().to_dict())
    print("terminated reasons (cohort):", c.stop_category.value_counts().to_dict())
    for name in ("label_strict", "label_broad", "label_pvalue", "label_hybrid"):
        v = c[name].dropna()
        print(f"{name}: n={len(v)}  success rate={v.mean():.3f}")
    print("split sizes:", c.split.value_counts().to_dict())


if __name__ == "__main__":
    main()
