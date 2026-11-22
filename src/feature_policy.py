"""Which columns of trials.parquet may be used as model features.

The model predicts a trial's outcome when the trial starts. A feature is only
legitimate if its value would have been known then. ClinicalTrials.gov serves
the *latest* version of each record, so several fields were filled in or edited
after the trial ended. Every column gets one tier:

* BLOCKED  -- known only after the outcome (or defines the label). Never use.
* CAUTION  -- design information, but the latest record may have been edited
              after the trial ended. Usable; Phase 4 should ablate each one and
              text must go through ``scrub_text``.
* ALLOWED  -- fixed at design time.

Evidence for each decision is in reports/leakage.md (src/leakage_audit.py).
Feature code should call ``check_source_columns`` on the raw columns it reads.
"""

from __future__ import annotations

import re

ID_COLUMNS = {"nct_id"}

BLOCKED: dict[str, str] = {
    "overall_status": "label source",
    "why_stopped": "label source",
    "enrollment_count": "ACTUAL enrollment for finished trials (terminated trials enroll fewer); univariate AUC ~0.70",
    "enrollment_type": "ACTUAL vs ESTIMATED reflects whether the trial finished",
    "start_date_type": "ACTUAL vs ANTICIPATED is updated once the trial starts",
    "primary_completion_date": "end date: gives trial duration",
    "primary_completion_date_type": "ACTUAL vs ANTICIPATED reflects whether the trial finished",
    "completion_date": "end date: gives trial duration",
    "completion_date_type": "ACTUAL vs ANTICIPATED reflects whether the trial finished",
    "first_post_date": "registration lag: a trial registered late has already survived that long",
    "results_first_post_date": "results exist only after the trial",
    "last_update_post_date": "record history after the trial",
    "status_verified_date": "record history after the trial",
    "has_results": "results exist only after the trial",
    "n_primary_pvalues": "results section",
    "n_primary_sig": "results section",
    "primary_min_pvalue": "results section",
}

CAUTION: dict[str, str] = {
    "brief_summary": "sometimes edited after the trial ends; use scrub_text",
    "detailed_description": "sometimes edited after the trial ends (termination notes, results); use scrub_text",
    "n_secondary_outcomes": "outcome lists are often revised when results are posted",
    "n_other_outcomes": "outcome lists are often revised when results are posted",
    "n_locations": "final site list; a trial stopped early may never have opened later sites",
    "countries": "final site list; a trial stopped early may never have opened later sites",
    "n_countries": "final site list; a trial stopped early may never have opened later sites",
    "has_us_site": "final site list; a trial stopped early may never have opened later sites",
    "n_overall_officials": "contact list of the latest record version",
    "fda_regulated_drug": "field added in 2017: missing means the record was not updated since",
    "fda_regulated_device": "field added in 2017: missing means the record was not updated since",
}

ALLOWED: dict[str, str] = {
    "start_date": "prediction time; use the start year only",
    "brief_title": "",
    "official_title": "",
    "org_class": "",
    "lead_sponsor": "high cardinality: fit encodings on the train split only",
    "lead_sponsor_class": "",
    "responsible_party_type": "",
    "n_collaborators": "",
    "collaborator_classes": "",
    "has_dmc": "",
    "conditions": "high cardinality: fit encodings on the train split only",
    "keywords": "high cardinality: fit encodings on the train split only",
    "study_type": "constant in the cohort (INTERVENTIONAL)",
    "phase": "",
    "allocation": "",
    "intervention_model": "",
    "primary_purpose": "",
    "masking": "",
    "who_masked": "",
    "n_arms": "",
    "arm_types": "",
    "n_interventions": "",
    "intervention_types": "",
    "intervention_names": "high cardinality: fit encodings on the train split only",
    "n_primary_outcomes": "",
    "primary_outcome_measures": "",
    "primary_outcome_time_frames": "",
    "eligibility_criteria": "",
    "healthy_volunteers": "",
    "sex": "",
    "minimum_age": "",
    "maximum_age": "",
    "min_age_years": "",
    "max_age_years": "",
    "std_ages": "",
    "condition_mesh_terms": "high cardinality: fit encodings on the train split only",
    "condition_mesh_ancestors": "use to derive therapeutic area",
    "intervention_mesh_terms": "high cardinality: fit encodings on the train split only",
    "intervention_mesh_ancestors": "",
}

TIERS = {**{c: "id" for c in ID_COLUMNS}, **{c: "blocked" for c in BLOCKED}, **{c: "caution" for c in CAUTION},
         **{c: "allowed" for c in ALLOWED}}


def tier(column: str) -> str:
    return TIERS.get(column, "unknown")


def check_source_columns(columns, allow_caution: bool = True) -> None:
    """Raise if feature code reads a blocked, unknown (or, optionally, caution) column."""
    bad = {c: tier(c) for c in columns if tier(c) in ("blocked", "unknown", "id") or (tier(c) == "caution" and not allow_caution)}
    if bad:
        raise ValueError(f"columns not allowed as features: {bad}")


# ---------------------------------------------------------------------------
# text scrubbing
# ---------------------------------------------------------------------------
# Sentences that report what happened to the trial rather than what it planned.
POST_HOC_TEXT = re.compile(
    r"\b(terminat\w*|prematurely|halted|withdrawn|futility"
    r"|stopped\s+(early|prematurely)|discontinued\s+(early|prematurely)"
    r"|(study|trial)\s+(was|has\s+been)\s+(completed|closed|stopped|ended|suspended)"
    r"|results?\s+(showed|demonstrated|indicated|suggested|revealed)"
    r"|(met|did\s+not\s+meet|failed\s+to\s+meet)\s+(its|the)\s+primary"
    r"|a\s+total\s+of\s+\d+\s+(patients|subjects|participants|volunteers)\s+(were|was)\s+(enrolled|randomi[sz]ed|treated|included)"
    r"|\d+\s+(patients|subjects|participants|volunteers)\s+(were|have\s+been)\s+(enrolled|randomi[sz]ed|included)"
    r"|(was|were|has\s+been|had\s+been)\s+(not\s+)?(stopped|discontinued|closed|ended|suspended|initiated|conducted)\b"
    r"|decision\s+to\s+(stop|discontinue|close|end|suspend|halt|terminate)|\bthis\s+decision\b"
    r"|(stop|clos|discontinu|suspend|halt)\w*\s+(\w+\s+){0,2}(enrol\w*|recruit\w*|accrual|development|dosing)\b"
    r"|(stopped|closed|discontinued|suspended|halted)\s+(the\s+)?(study|trial)\b"
    r"|(dsmb|dsmc|i?dmc|monitoring\s+(board|committee))\W+(\w+\W+){0,2}recommend\w*"
    r"|(do|does|did)\s+not\s+outweigh)",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def scrub_text(text: str | None) -> str:
    """Drop every sentence that reports the trial's course or result."""
    if not text:
        return ""
    return " ".join(s for s in _SENTENCE_SPLIT.split(text) if s and not POST_HOC_TEXT.search(s))
