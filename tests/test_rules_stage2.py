"""SPEC §9 — rules of Stage 2 and determinism."""

import random
from dataclasses import replace

import pytest

from locassess.config import ScoringCfg
from locassess.stage2_score import (
    Evidence, ScoringParams, build_profile, criterion_score, evidence_status, overall_score,
    rating, red_flags, score_criteria, trend,
)

ALL_MAIN = [f"H{i:02d}" for i in range(1, 27)]
FACTORS = {k: ("A" if k <= "H13" else "B") for k in ALL_MAIN}


def params(**kw):
    p = ScoringParams.from_config(ScoringCfg(), run_date="2026-10", period_from="2021-10")
    return replace(p, **kw)


def ev(i, main="H02", article=None, r=1, s=2, t="fact", c=0.9, d="2026-01"):
    return Evidence(i, main, article if article is not None else i, r, s, t, c, d)


# --- 9.1 / 9.2 -------------------------------------------------------------

def test_single_evidence_score_is_r_times_s():
    assert criterion_score([ev(1, r=-1, s=3)], params()) == pytest.approx(-3)


def test_score_range_and_neutral_evidence():
    s = criterion_score([ev(1, r=1, s=3), ev(2, r=0, s=3)], params())
    assert -3 <= s <= 3
    assert s == pytest.approx(1.5)  # neutral evidence weighs in the denominator


def test_type_weights_apply():
    p = params()
    s = criterion_score([ev(1, r=1, s=3, t="fact"), ev(2, r=-1, s=3, t="opinion")], p)
    assert s == pytest.approx((3 * 1.0 - 3 * 0.3) / 1.3)


def test_event_after_run_date_rejected():
    with pytest.raises(ValueError):
        criterion_score([ev(1, d="2026-11")], params())


@pytest.mark.parametrize("kw", [dict(r=2), dict(s=0), dict(t="rumour"), dict(c=1.5), dict(d="2026")])
def test_invalid_evidence_rejected(kw):
    with pytest.raises(ValueError):
        ev(1, **kw)


# --- 9.3 status ------------------------------------------------------------

def test_status_no_statement():
    assert evidence_status([], params()) == "no_statement"


def test_status_thin_by_count():
    assert evidence_status([ev(1), ev(2)], params()) == "thin"


def test_status_thin_single_article():
    assert evidence_status([ev(1, article=7), ev(2, article=7), ev(3, article=7)], params()) == "thin"


def test_status_sufficient():
    assert evidence_status([ev(1, article=7), ev(2, article=7), ev(3, article=8)], params()) == "sufficient"


def test_no_statement_is_not_neutral():
    c = score_criteria([ev(1, main="H02")], ALL_MAIN, params())["H03"]
    assert (c.score, c.rating, c.status, c.trend) == (None, None, "no_statement", "n/a")


# --- 9.4 rating ------------------------------------------------------------

@pytest.mark.parametrize("score,expected", [
    (1.0, "green"), (2.5, "green"), (0.99, "yellow"), (0.0, "yellow"), (-0.99, "yellow"),
    (-1.0, "red"), (-3.0, "red"), (None, None),
])
def test_rating_thresholds(score, expected):
    assert rating(score, params()) == expected


# --- 9.5 trend -------------------------------------------------------------

def test_trend_needs_both_halves():
    assert trend([ev(1, d="2026-01"), ev(2, d="2025-01")], params())[0] == "n/a"


def test_trend_midpoint_month_belongs_to_newer_half():
    # period 2021-10 .. 2026-10 → midpoint 2024-04
    label, older, newer = trend([ev(1, r=1, s=1, d="2024-03"), ev(2, r=1, s=3, d="2024-04")], params())
    assert (older, newer, label) == (pytest.approx(1), pytest.approx(3), "improving")


def test_trend_ignores_time_weight():
    # Same evidence in both halves → equal half scores despite different ages.
    label, older, newer = trend([ev(1, r=1, s=2, d="2022-01"), ev(2, r=1, s=2, d="2026-01")], params())
    assert older == pytest.approx(newer) and label == "stable"


def test_trend_threshold_boundaries():
    p = params()
    assert trend([ev(1, r=1, s=1, d="2022-01"), ev(2, r=1, s=2, d="2026-01")], p)[0] == "improving"
    assert trend([ev(1, r=1, s=2, d="2022-01"), ev(2, r=1, s=1, d="2026-01")], p)[0] == "deteriorating"
    assert trend([ev(1, r=0, s=1, d="2022-01"), ev(2, r=1, s=1, d="2026-01")], replace(p, trend_threshold=1.5))[0] == "stable"


# --- 9.6 red flags ---------------------------------------------------------

def _rules(flags):
    return sorted(f.rule for f in flags)


def test_r1_recent_severe_fact():
    assert _rules(red_flags("H02", [ev(1, r=-1, s=3, d="2025-10")], -3, params())) == ["R1"]  # 12 months


def test_r1_not_when_older_than_recent_months():
    assert _rules(red_flags("H02", [ev(1, r=-1, s=3, d="2025-09")], -3, params())) == []  # 13 months


@pytest.mark.parametrize("kw", [dict(t="announcement"), dict(t="opinion"), dict(s=2), dict(r=0)])
def test_r1_requires_negative_intensity3_fact(kw):
    base = dict(r=-1, s=3, d="2026-05") | kw
    assert "R1" not in _rules(red_flags("H02", [ev(1, **base)], -1, params()))


def test_r2_two_articles():
    flags = red_flags("H02", [ev(1, r=-1, s=2), ev(2, r=-1, s=2, t="opinion"), ev(3, r=1, s=3)], 0, params())
    assert [(f.rule, f.evidence_ids) for f in flags] == [("R2", (1, 2))]


def test_r2_not_from_same_article():
    assert _rules(red_flags("H02", [ev(1, article=5, r=-1, s=2), ev(2, article=5, r=-1, s=2)], -2, params())) == []


def test_r2_not_for_intensity_1():
    assert _rules(red_flags("H02", [ev(1, r=-1, s=1), ev(2, r=-1, s=2)], -1, params())) == []


def test_r3_knockout():
    p = params(knockout_criteria=("H02",))
    assert _rules(red_flags("H02", [ev(1, r=-1, s=1)], -1.0, p)) == ["R3"]
    assert _rules(red_flags("H02", [ev(1, r=-1, s=1)], -0.9, p)) == []
    assert _rules(red_flags("H03", [ev(1, main="H03", r=-1, s=1)], -1.0, p)) == []


# --- 9.7 overall -----------------------------------------------------------

def _scores(**kw):
    return {k: None for k in ALL_MAIN} | kw


def test_overall_equal_weights_is_mean_of_evidenced():
    o = overall_score(_scores(H01=1.0, H02=-2.0))
    assert o.score == pytest.approx(-0.5)
    assert o.weights == {"H01": 0.5, "H02": 0.5}


def test_overall_slider_zero_excludes():
    assert overall_score(_scores(H01=1.0, H02=-2.0), sliders={"H02": 0}).score == pytest.approx(1.0)


def test_overall_all_zero_sliders_gives_none():
    assert overall_score(_scores(H01=1.0), sliders={"H01": 0}).score is None


def test_overall_no_evidence_gives_none():
    assert overall_score(_scores()).score is None


def test_overall_sliders_on_unevidenced_criteria_ignored():
    assert overall_score(_scores(H01=1.0), sliders={"H05": 10}).score == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [{"H01": 11}, {"H01": -1}, {"H01": 1.5}, {"H01": True}, {"X99": 1}])
def test_overall_invalid_sliders(bad):
    with pytest.raises(ValueError):
        overall_score(_scores(H01=1.0), sliders=bad)


def test_factor_sub_scores():
    o = overall_score(_scores(H01=1.0, H02=3.0, H20=-2.0), factors=FACTORS)
    assert o.factor_scores == {"A": pytest.approx(2.0), "B": pytest.approx(-2.0)}
    assert overall_score(_scores(H01=1.0), factors=FACTORS).factor_scores["B"] is None


# --- determinism -----------------------------------------------------------

def _random_evidence(seed):
    rnd = random.Random(seed)
    return [
        Evidence(i, rnd.choice(ALL_MAIN), rnd.randint(1, 15), rnd.choice([-1, 0, 1]), rnd.randint(1, 3),
                 rnd.choice(["fact", "announcement", "opinion"]), round(rnd.uniform(0.3, 1), 3),
                 f"{rnd.randint(2021, 2026)}-{rnd.randint(1, 9):02d}")
        for i in range(1, 61)
    ]


def test_deterministic_and_order_independent():
    evs = _random_evidence(42)
    p = params(knockout_criteria=("H01", "H20"))
    a = build_profile(evs, ALL_MAIN, p, sliders={"H03": 4}, factors=FACTORS)
    b = build_profile(list(evs), ALL_MAIN, p, sliders={"H03": 4}, factors=FACTORS)
    shuffled = list(evs)
    random.Random(1).shuffle(shuffled)
    c = build_profile(shuffled, ALL_MAIN, p, sliders={"H03": 4}, factors=FACTORS)
    assert a == b == c


def test_unknown_criterion_rejected():
    with pytest.raises(ValueError):
        score_criteria([ev(1, main="H99")], ALL_MAIN, params())
