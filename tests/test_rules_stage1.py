"""SPEC §8.4 extraction rules."""

import pytest

from locassess.config import ExtractionCfg
from locassess.providers.decision import DecisionResult, Option, Question
from locassess.stage1_extract import derive_evidence, merge_sections, most_probable

P = ExtractionCfg()   # affected 0.5, review 0.6, gap 0.2


def q(key, values):
    return Question(key, key, tuple(Option(v) for v in values))


QS = [q("direction", ["-1", "0", "1"]), q("intensity", ["1", "2", "3"]),
      q("evidence_type", ["fact", "announcement", "opinion"]), q("sub_criterion", ["H02.01", "H02.02", "overarching"]),
      q("evidence_sentence", ["S1", "S2", "S3"])]


def result(direction=None, intensity=None, etype=None, sub=None, sent=None, refusals=()):
    d = {"direction": direction or {"-1": 0.1, "0": 0.1, "1": 0.8},
         "intensity": intensity or {"1": 0.1, "2": 0.7, "3": 0.2},
         "evidence_type": etype or {"fact": 0.9, "announcement": 0.1, "opinion": 0.0},
         "sub_criterion": sub or {"H02.01": 0.6, "H02.02": 0.3, "overarching": 0.1},
         "evidence_sentence": sent or {"S1": 0.1, "S2": 0.7, "S3": 0.2}}
    for k in refusals:
        d.pop(k)
    return DecisionResult(distributions=d, refusals=list(refusals))


def derive(res, p_aff=0.9):
    return derive_evidence("H02", 1, p_aff, res, QS, ["S1", "S2", "S3"], P)


def test_most_probable_and_ties():
    assert most_probable({"a": 0.2, "b": 0.5, "c": 0.3}, ["a", "b", "c"]) == "b"
    assert most_probable({"a": 0.4, "b": 0.4, "c": 0.2}, ["b", "a", "c"]) == "b"  # tie → listed first


def test_fields_take_most_probable_option():
    e = derive(result())
    assert (e.direction, e.intensity, e.evidence_type, e.sub_id, e.sentence_keys) == (1, 2, "fact", "H02.01", ("S2",))


def test_confidence_is_minimum_of_core_fields():
    assert derive(result(), p_aff=0.9).confidence == pytest.approx(0.7)    # intensity weakest
    assert derive(result(), p_aff=0.55).confidence == pytest.approx(0.55)  # affected weakest
    e = derive(result(direction={"-1": 0.45, "0": 0.1, "1": 0.45}))
    assert e.direction == -1 and e.confidence == pytest.approx(0.45)        # tie → first option (-1)


def test_evidence_type_not_part_of_confidence():
    assert derive(result(etype={"fact": 0.34, "announcement": 0.33, "opinion": 0.33})).confidence == pytest.approx(0.7)


def test_needs_review_threshold():
    assert derive(result(), p_aff=0.59).needs_review
    assert not derive(result(), p_aff=0.6).needs_review


def test_second_sentence_within_gap():
    assert derive(result(sent={"S1": 0.45, "S2": 0.25, "S3": 0.3})).sentence_keys == ("S1", "S3")
    assert derive(result(sent={"S1": 0.5, "S2": 0.3, "S3": 0.2})).sentence_keys == ("S1", "S2")  # exactly the gap
    assert derive(result(sent={"S1": 0.6, "S2": 0.3, "S3": 0.1})).sentence_keys == ("S1",)


def test_overarching_gives_no_sub_id():
    assert derive(result(sub={"H02.01": 0.1, "H02.02": 0.1, "overarching": 0.8})).sub_id is None


def test_refused_core_field_gives_no_evidence():
    assert derive(result(refusals=("direction",))) is None
    e = derive(result(refusals=("sub_criterion", "evidence_sentence")))
    assert e.sub_id is None and e.sentence_keys == ("S1",)


def test_merge_sections_keeps_higher_confidence():
    a = derive_evidence("H02", 1, 0.62, result(), QS, ["S1", "S2", "S3"], P)   # confidence 0.62 < 0.7
    b = derive_evidence("H02", 2, 0.9, result(), QS, ["S1", "S2", "S3"], P)
    c = derive_evidence("H03", 2, 0.8, result(), QS, ["S1", "S2", "S3"], P)
    merged = merge_sections([a, b, c])
    assert merged["H02"].section == 2 and merged["H03"] == c
    tie = merge_sections([derive_evidence("H02", 2, 0.9, result(), QS, ["S1"], P),
                          derive_evidence("H02", 1, 0.9, result(), QS, ["S1"], P)])
    assert tie["H02"].section == 1
