"""Stage 2 — rule-based scoring (SPEC §9).

Pure functions over plain data classes. No model calls, no database or network access:
the same evidence and the same parameters always give the same profile.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import date

from locassess.config import ScoringCfg

EVIDENCE_TYPES = ("fact", "announcement", "opinion")

NO_STATEMENT = "no_statement"
THIN = "thin"
SUFFICIENT = "sufficient"

GREEN, YELLOW, RED = "green", "yellow", "red"
IMPROVING, STABLE, DETERIORATING, TREND_NA = "improving", "stable", "deteriorating", "n/a"

SLIDER_MIN, SLIDER_MAX = 0, 10


# ---------------------------------------------------------------------------
# Months
# ---------------------------------------------------------------------------

def month_index(value: str | date) -> int:
    """Months since year 0 for a date or a 'YYYY-MM' / 'YYYY-MM-DD' string."""
    if isinstance(value, date):
        return value.year * 12 + value.month - 1
    parts = str(value).split("-")
    if len(parts) < 2:
        raise ValueError(f"expected YYYY-MM, got {value!r}")
    year, month = int(parts[0]), int(parts[1])
    if not 1 <= month <= 12:
        raise ValueError(f"invalid month in {value!r}")
    return year * 12 + month - 1


def month_str(index: int) -> str:
    year, m = divmod(index, 12)
    return f"{year:04d}-{m + 1:02d}"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Evidence:
    evidence_id: int
    main_id: str
    article_id: int
    direction: int          # -1 | 0 | 1
    intensity: int          # 1 | 2 | 3
    evidence_type: str      # fact | announcement | opinion
    confidence: float       # 0..1
    event_date: str         # YYYY-MM
    outlet: str | None = None

    def __post_init__(self) -> None:
        if self.direction not in (-1, 0, 1):
            raise ValueError(f"evidence {self.evidence_id}: direction must be -1, 0 or 1")
        if self.intensity not in (1, 2, 3):
            raise ValueError(f"evidence {self.evidence_id}: intensity must be 1, 2 or 3")
        if self.evidence_type not in EVIDENCE_TYPES:
            raise ValueError(f"evidence {self.evidence_id}: unknown evidence_type {self.evidence_type!r}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"evidence {self.evidence_id}: confidence must be in [0, 1]")
        month_index(self.event_date)


@dataclass(frozen=True)
class ScoringParams:
    run_month: str                      # YYYY-MM; ages are measured to this month
    period_from: str                    # YYYY-MM; start of the assessment period
    type_weights: Mapping[str, float]
    half_life_months: float
    green_threshold: float
    red_threshold: float
    trend_threshold: float
    sufficient_min_evidence: int
    sufficient_min_articles: int
    recent_months: int
    knockout_criteria: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, scoring: ScoringCfg, run_date: str | date, period_from: str | date) -> ScoringParams:
        return cls(
            run_month=month_str(month_index(run_date)),
            period_from=month_str(month_index(period_from)),
            type_weights=dict(scoring.type_weights),
            half_life_months=scoring.half_life_months,
            green_threshold=scoring.rating_thresholds.green,
            red_threshold=scoring.rating_thresholds.red,
            trend_threshold=scoring.trend_threshold,
            sufficient_min_evidence=scoring.status.sufficient_min_evidence,
            sufficient_min_articles=scoring.status.sufficient_min_articles,
            recent_months=scoring.red_flags.recent_months,
            knockout_criteria=tuple(scoring.red_flags.knockout_criteria),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type_weights"] = dict(sorted(self.type_weights.items()))
        d["knockout_criteria"] = list(self.knockout_criteria)
        return d


@dataclass(frozen=True)
class RedFlag:
    rule: str                           # R1 | R2 | R3
    main_id: str
    evidence_ids: tuple[int, ...]
    description: str


@dataclass(frozen=True)
class CriterionResult:
    main_id: str
    score: float | None
    rating: str | None
    status: str
    trend: str
    trend_older: float | None
    trend_newer: float | None
    evidence_ids: tuple[int, ...]
    red_flags: tuple[RedFlag, ...] = ()


@dataclass(frozen=True)
class OverallResult:
    score: float | None
    sliders: dict[str, int]
    weights: dict[str, float]           # only criteria with evidence
    factor_scores: dict[str, float | None] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 9.1 Evidence value
# ---------------------------------------------------------------------------

def age_months(event_date: str, params: ScoringParams) -> int:
    age = month_index(params.run_month) - month_index(event_date)
    if age < 0:
        raise ValueError(f"event date {event_date} lies after run month {params.run_month}")
    return age


def time_weight(age: int, half_life_months: float) -> float:
    return 0.5 ** (age / half_life_months)


def evidence_weight(ev: Evidence, params: ScoringParams, use_time: bool = True) -> float:
    """w_type · c · w_time (the denominator term of 9.2)."""
    w = params.type_weights[ev.evidence_type] * ev.confidence
    if use_time:
        w *= time_weight(age_months(ev.event_date, params), params.half_life_months)
    return w


def evidence_value(ev: Evidence, params: ScoringParams, use_time: bool = True) -> float:
    """e_i = r · s · w_type · c · w_time."""
    return ev.direction * ev.intensity * evidence_weight(ev, params, use_time)


# ---------------------------------------------------------------------------
# 9.2 – 9.5 Criterion score, status, rating, trend
# ---------------------------------------------------------------------------

def _sorted(evidence: Iterable[Evidence]) -> list[Evidence]:
    return sorted(evidence, key=lambda e: e.evidence_id)


def criterion_score(evidence: Iterable[Evidence], params: ScoringParams, use_time: bool = True) -> float | None:
    """S_k = Σ e_i / Σ (w_type · c_i · w_time); None if there is nothing to weigh."""
    evs = _sorted(evidence)
    den = math.fsum(evidence_weight(e, params, use_time) for e in evs)
    if den <= 0:
        return None
    return math.fsum(evidence_value(e, params, use_time) for e in evs) / den


def evidence_status(evidence: Iterable[Evidence], params: ScoringParams) -> str:
    evs = list(evidence)
    if not evs:
        return NO_STATEMENT
    n_articles = len({e.article_id for e in evs})
    if len(evs) >= params.sufficient_min_evidence and n_articles >= params.sufficient_min_articles:
        return SUFFICIENT
    return THIN


def rating(score: float | None, params: ScoringParams) -> str | None:
    if score is None:
        return None
    if score >= params.green_threshold:
        return GREEN
    if score <= params.red_threshold:
        return RED
    return YELLOW


def period_midpoint(params: ScoringParams) -> float:
    """Midpoint of the period as a (possibly fractional) month index."""
    return (month_index(params.period_from) + month_index(params.run_month)) / 2


def trend(evidence: Iterable[Evidence], params: ScoringParams) -> tuple[str, float | None, float | None]:
    """(trend, older-half score, newer-half score), both halves scored without w_time.

    Evidence dated before the midpoint month belongs to the older half; the midpoint month
    itself and later belong to the newer half.
    """
    mid = period_midpoint(params)
    evs = list(evidence)
    older = criterion_score([e for e in evs if month_index(e.event_date) < mid], params, use_time=False)
    newer = criterion_score([e for e in evs if month_index(e.event_date) >= mid], params, use_time=False)
    if older is None or newer is None:
        return TREND_NA, older, newer
    diff = newer - older
    if diff >= params.trend_threshold:
        return IMPROVING, older, newer
    if diff <= -params.trend_threshold:
        return DETERIORATING, older, newer
    return STABLE, older, newer


# ---------------------------------------------------------------------------
# 9.6 Red flags
# ---------------------------------------------------------------------------

def red_flags(main_id: str, evidence: Iterable[Evidence], score: float | None,
              params: ScoringParams) -> list[RedFlag]:
    evs = _sorted(evidence)
    flags: list[RedFlag] = []

    # R1: a recent fact with direction -1 and intensity 3.
    for e in evs:
        if (e.evidence_type == "fact" and e.direction == -1 and e.intensity == 3
                and age_months(e.event_date, params) <= params.recent_months):
            flags.append(RedFlag("R1", main_id, (e.evidence_id,),
                                 f"recent severe negative fact (within {params.recent_months} months)"))

    # R2: >= 2 negative evidence with intensity >= 2 from different articles.
    neg = [e for e in evs if e.direction == -1 and e.intensity >= 2]
    if len({e.article_id for e in neg}) >= 2:
        flags.append(RedFlag("R2", main_id, tuple(e.evidence_id for e in neg),
                             "repeated negative evidence from different articles"))

    # R3: knockout criterion with S <= red threshold.
    if main_id in params.knockout_criteria and score is not None and score <= params.red_threshold:
        flags.append(RedFlag("R3", main_id, tuple(e.evidence_id for e in evs),
                             "knockout criterion rated red"))
    return flags


# ---------------------------------------------------------------------------
# Criterion level
# ---------------------------------------------------------------------------

def score_criterion(main_id: str, evidence: Iterable[Evidence], params: ScoringParams) -> CriterionResult:
    evs = _sorted(e for e in evidence if e.main_id == main_id)
    score = criterion_score(evs, params) if evs else None
    tr, older, newer = trend(evs, params)
    return CriterionResult(
        main_id=main_id,
        score=score,
        rating=rating(score, params),
        status=evidence_status(evs, params),
        trend=tr,
        trend_older=older,
        trend_newer=newer,
        evidence_ids=tuple(e.evidence_id for e in evs),
        red_flags=tuple(red_flags(main_id, evs, score, params)),
    )


def score_criteria(evidence: Iterable[Evidence], main_ids: Iterable[str],
                   params: ScoringParams) -> dict[str, CriterionResult]:
    """Score every main criterion; criteria without evidence get status no_statement."""
    evs = _sorted(evidence)
    ids = sorted(set(main_ids))
    unknown = {e.main_id for e in evs} - set(ids)
    if unknown:
        raise ValueError(f"evidence for unknown criteria: {sorted(unknown)}")
    return {k: score_criterion(k, evs, params) for k in ids}


# ---------------------------------------------------------------------------
# 9.7 Overall score and weight sliders
# ---------------------------------------------------------------------------

def _weighted(scores: Mapping[str, float], sliders: Mapping[str, int]) -> tuple[float | None, dict[str, float]]:
    total = sum(sliders[k] for k in scores)
    if total == 0:
        return None, {}
    weights = {k: sliders[k] / total for k in sorted(scores)}
    return math.fsum(weights[k] * scores[k] for k in weights), weights


def overall_score(scores: Mapping[str, float | None], sliders: Mapping[str, int] | None = None,
                  default_slider: int = 1, factors: Mapping[str, str] | None = None) -> OverallResult:
    """G = Σ_{k∈E} w_k · S_k with w_k = g_k / Σ_{j∈E} g_j; E = criteria with a score.

    `scores` covers all main criteria (None = no evidence). `factors` maps main_id → factor
    and enables factor sub-scores.
    """
    full = {k: (sliders or {}).get(k, default_slider) for k in sorted(scores)}
    for k, g in full.items():
        if isinstance(g, bool) or not isinstance(g, int) or not SLIDER_MIN <= g <= SLIDER_MAX:
            raise ValueError(f"slider for {k} must be an integer in {SLIDER_MIN}..{SLIDER_MAX}, got {g!r}")
    unknown = set(sliders or {}) - set(scores)
    if unknown:
        raise ValueError(f"sliders for unknown criteria: {sorted(unknown)}")

    evidenced = {k: s for k, s in scores.items() if s is not None}
    g, weights = _weighted(evidenced, full)

    factor_scores: dict[str, float | None] = {}
    if factors:
        for factor in sorted(set(factors.values())):
            subset = {k: s for k, s in evidenced.items() if factors.get(k) == factor}
            factor_scores[factor] = _weighted(subset, full)[0] if subset else None
    return OverallResult(score=g, sliders=full, weights=weights, factor_scores=factor_scores)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Profile:
    params: ScoringParams
    criteria: dict[str, CriterionResult]
    overall: OverallResult

    @property
    def red_flags(self) -> list[RedFlag]:
        return [f for c in self.criteria.values() for f in c.red_flags]


def build_profile(evidence: Iterable[Evidence], main_ids: Iterable[str], params: ScoringParams,
                  sliders: Mapping[str, int] | None = None, default_slider: int = 1,
                  factors: Mapping[str, str] | None = None) -> Profile:
    criteria = score_criteria(evidence, main_ids, params)
    overall = overall_score({k: c.score for k, c in criteria.items()}, sliders, default_slider, factors)
    return Profile(params=params, criteria=criteria, overall=overall)
