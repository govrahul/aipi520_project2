"""Tests for the feature policy and text scrubbing (run: python -m pytest tests/)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from feature_policy import check_source_columns, scrub_text  # noqa: E402

POST_HOC = [
    "The study was terminated early due to slow enrollment.",
    "PF-3512676 dosing was stopped 20 June 2007 in response to DSMC recommendation to close the trial.",
    "There were no safety concerns involved in the decision to stop enrollment.",
    "The intervention was stopped on November 7, 2012 for safety concerns after an interim analysis.",
    "The sponsor discontinued Study AT1001-041 for logistical reasons.",
    "The phase 2 portion of this study was not initiated.",
    "On balance, the benefits of treating patients with fasiglifam do not outweigh the potential risks.",
    "A total of 212 patients were randomized.",
    "Results showed no difference between arms.",
    "It should be noted that safety concerns have not factored into this decision.",
]
PLANNED = [
    "This is a randomized, double-blind, placebo-controlled study of drug X in adults with asthma.",
    "Patients will receive 10 mg daily for 12 weeks.",
    "The primary endpoint is change from baseline in FEV1 at week 12.",
    "Treatment will continue until progression of disease or unacceptable toxicity.",
]


@pytest.mark.parametrize("sentence", POST_HOC)
def test_post_hoc_sentences_are_removed(sentence):
    assert scrub_text(sentence) == ""


@pytest.mark.parametrize("sentence", PLANNED)
def test_planned_sentences_are_kept(sentence):
    assert scrub_text(sentence) == sentence


def test_scrub_keeps_other_sentences():
    text = PLANNED[0] + " " + POST_HOC[0] + " " + PLANNED[1]
    assert scrub_text(text) == PLANNED[0] + " " + PLANNED[1]


def test_blocked_columns_are_rejected():
    with pytest.raises(ValueError):
        check_source_columns(["phase", "enrollment_count"])
    with pytest.raises(ValueError):
        check_source_columns(["n_locations"], allow_caution=False)
    check_source_columns(["phase", "n_locations", "eligibility_criteria"])
