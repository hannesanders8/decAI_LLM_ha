"""SPEC §9.8 — worked example (MUST pass)."""

import pytest

from locassess.config import ScoringCfg
from locassess.stage2_score import (
    Evidence, ScoringParams, age_months, build_profile, evidence_weight, overall_score, time_weight,
)

TOL = 0.005

PARAMS = ScoringParams.from_config(ScoringCfg(), run_date="2026-10", period_from="2021-10")

E1 = Evidence(1, "H20", article_id=101, direction=-1, intensity=2, evidence_type="fact",
              confidence=0.62, event_date="2025-03")   # 30 % of apprenticeships unfilled
E2 = Evidence(2, "H20", article_id=102, direction=-1, intensity=3, evidence_type="fact",
              confidence=0.85, event_date="2026-06")   # plant cuts production due to staff shortage
E3 = Evidence(3, "H20", article_id=103, direction=1, intensity=2, evidence_type="announcement",
              confidence=0.80, event_date="2022-11")   # state programme to recruit skilled workers
EVIDENCE = [E1, E2, E3]
ALL_MAIN = [f"H{i:02d}" for i in range(1, 27)]


@pytest.fixture(scope="module")
def h20():
    return build_profile(EVIDENCE, ALL_MAIN, PARAMS).criteria["H20"]


def test_time_weights():
    w = [time_weight(age_months(e.event_date, PARAMS), PARAMS.half_life_months) for e in EVIDENCE]
    assert w == pytest.approx([0.58, 0.89, 0.26], abs=TOL)


def test_evidence_weights():
    assert [evidence_weight(e, PARAMS) for e in EVIDENCE] == pytest.approx([0.36, 0.76, 0.12], abs=TOL)


def test_criterion_score_rating_status(h20):
    assert h20.score == pytest.approx(-2.212, abs=TOL)
    assert round(h20.score, 1) == -2.2
    assert h20.rating == "red"
    assert h20.status == "sufficient"


def test_trend(h20):
    assert h20.trend_older == pytest.approx(2.000, abs=TOL)
    assert h20.trend_newer == pytest.approx(-2.578, abs=TOL)
    assert h20.trend == "deteriorating"


def test_red_flags(h20):
    flags = {(f.rule, f.evidence_ids) for f in h20.red_flags}
    assert flags == {("R1", (2,)), ("R2", (1, 2))}


def test_overall_equal_weights(h20):
    scores = {k: None for k in ALL_MAIN} | {"H20": h20.score, "H10": 2.6, "H01": 0.4}
    assert overall_score(scores).score == pytest.approx(0.263, abs=TOL)


def test_overall_with_sliders(h20):
    scores = {k: None for k in ALL_MAIN} | {"H20": h20.score, "H10": 2.6, "H01": 0.4}
    result = overall_score(scores, sliders={"H20": 2, "H10": 1, "H01": 1})
    assert result.score == pytest.approx(-0.356, abs=TOL)
