"""Regression cases for the why_stopped classifier (run: python -m pytest tests/)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from label import classify_reason  # noqa: E402

CASES = [
    # negated reasons must not count as safety/efficacy
    ("Business Decision; No Safety Or Efficacy Concerns.", "business"),
    ("Study terminated due to resources and slow enrollment; not due to safety", "enrollment"),
    ("The trial was terminated for strategic reasons. The decision was not based on any safety and/or efficacy concerns", "business"),
    ("The study was terminated by the Sponsor for administrative reasons and not due to any safety concerns.", "business"),
    ("The early termination of this study is a business decision, Akari have made the decision to close their global "
     "Phase III PNH program. The decision was not related to any efficacy, safety or clinical concerns regarding Coversin/rVA576.", "business"),
    ("Merck has decided to not pursue phase 3 development of ceralifimod (ONO-4641). The decision was not related to any safety and efficacy findings", "business"),
    ("no safety signal; sponsor decision", "business"),
    ("Terminated for non-safety reasons", "other"),
    ("Non-safety device related issues", "other"),
    ("Sponsor no longer supports the study", "business"),
    # real failures, including ones phrased with a negation
    ("Study was not continued due to toxicity", "safety"),
    ("The drug showed no efficacy", "efficacy"),
    ("No additional efficacy seen with REGN2176-3 over aflibercept alone", "efficacy"),
    ("Data Monitoring Committee (DMC) decision; terminated due to futility, with no safety concerns; FDA agreement to terminate the study.", "efficacy"),
    ("Study was discontinued based on futility analysis conducted on Phase 3 trials and not based on safety concerns.", "efficacy"),
    ("Sufficient evidence of efficacy not met. Discontinuation not based on any safety concerns.", "efficacy"),
    ("A pre-planned interim analysis indicated the trial was unlikely to meet the pre-specified efficacy criteria. The decision was not based on any safety concerns.", "efficacy"),
    ("No participants were enrolled in Phase 2 as Phase I was terminated early due to the lack of a clear signal of robust clinical activity.", "efficacy"),
    ("The overall profile does not support development for obesity", "efficacy"),
    ("Results of interim efficacy analysis showed results were not promising.", "efficacy"),
    ("Livoletide did not result in a statistically significant improvement in hyperphagia and food-related behaviors", "efficacy"),
    ("Based on the preliminary results seen during interim analysis, which were confirmed in the final analysis, the Sponsor terminated the study", "efficacy"),
    ("Modified dose schedule presented no advantage over previously studied schedule", "efficacy"),
    ("Study did not show positive results", "efficacy"),
    ("IDMC recommendation for safety concerns", "safety"),
    ("Terminated due to non clinical toxicology findings and reduced safety margins.", "safety"),
    ("The study was terminated during dose escalation after a determination was made that the risk:benefit ratio no longer favored continued evaluation.", "safety"),
    ("Retinal vascular occlusion in 2 patients", "safety"),
    ("Early closure based on audit by study investigators after 3 patients died on study.", "safety"),
    ("As recommended by the DSMB.", "dsmb"),
    # held-out audit misses (fixed after the 95.5% estimate was measured)
    ("SP arms were stopped due to high levels of treatment failure.CD not available.", "efficacy"),
    ("The study was terminated following less favorable efficacy by Iscalimab (CFZ533) in liver transplant patients compared to tacrolimus.", "efficacy"),
    ("Missing evidence of the effectiveness of the study medication", "efficacy"),
    ("5 patients on tivicay had virological failure", "efficacy"),
    ("Results of primary analysis showed addition of canakinumab to combination treatment did not improve tumor response or overall survival", "efficacy"),
    ("The study was terminated due to the review of the asset (VBIR-2) within the Sponsor's oncology portfolio; the study was not terminated because of safety concerns.", "business"),
    ("Enrollment challenges due to changes in standards of care resulted in premature termination. No safety or efficacy events factored into this action.", "enrollment"),
    ("Did not meet enrollment objectives.", "enrollment"),
    ("Failure to meet recruitment target during the recruitment period", "enrollment"),
    ("APPRISE closure prompted by SWOG S0205 not meeting primary endpoint-improving OS. APPRISE enrollment closure due to similar design;no unexpected safety data", "efficacy"),
    # stopped early because it worked
    ("The board decision has been made to conclude the trial ahead of schedule. The primary endpoint has been successfully achieved and published", "early_benefit"),
    ("The Sponsor is terminating the study because the primary objective was achieved after 5 of the 6 subjects were assessed in the final cohort.", "early_benefit"),
    ("Study was terminated because dramatic benefits seen during the first 24 hours of drug administration combined with COVID restrictions "
     "on research activities suggested it was both unethical and unnecessary to complete the planned enrollment", "early_benefit"),
    ("positive results from a larger study with same regimen was released.", "other"),
    # neutral (operational) reasons
    ("Recruitment Goal of 20 Not Met", "enrollment"),
    ("Recommended by DSMB due to lack of accrual", "enrollment"),
    ("Slow accrual and loss of funding", "enrollment"),
    ("Study terminated due to slow acrual", "enrollment"),
    ("\\<75% participation", "enrollment"),
    ("Pharmaceutical company withdrew support", "business"),
    ("The development program has been terminated", "business"),
    ("Funding unavailable", "funding"),
    ("Schering-Plough discontinued supplying study drug.", "logistics"),
    ("FDA Hold May 2007", "logistics"),
    ("Drug no longer available", "logistics"),
    ("Replaced by another study", "logistics"),
    ("Recent results from a Phase 3 study are expected to change the standard of care for patients with inoperable CCA, "
     "rendering the RELEASE study challenging to complete and potentially inadequate for NDA approval.", "logistics"),
    ("See termination reason in detailed description.", "other"),
    (None, "missing"),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_classify_reason(text, expected):
    assert classify_reason(text)[0] == expected
