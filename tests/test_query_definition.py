"""SPEC §8.2/8.3 query definition: identical questions for every decision model."""

import pytest

from locassess.codebook_draft import write_codebook_draft
from locassess.criteria import load_codebook, load_criteria_csv
from locassess.query_definition import query_a_questions, query_b_questions, render_state
from tests.conftest import CRITERIA_CSV
from tests.fakes import FakeLLM

CRITERIA = load_criteria_csv(CRITERIA_CSV)
BY_ID = {m.main_id: m for m in CRITERIA}


@pytest.fixture(scope="module")
def codebook(tmp_path_factory):
    path = tmp_path_factory.mktemp("cb") / "codebook.yaml"
    write_codebook_draft(path, FakeLLM(), CRITERIA, "gpt-test")
    return load_codebook(path, CRITERIA)


def test_query_a(codebook):
    qs = query_a_questions(codebook, CRITERIA, "Leipzig")
    assert [q.key for q in qs] == [f"H{i:02d}_affected" for i in range(1, 27)]
    assert all(q.values == ["yes", "no"] for q in qs)
    assert "Definition of H02." in qs[1].instructions and "Leipzig" in qs[1].instructions


def test_query_b_fields(codebook):
    qs = query_b_questions(codebook, BY_ID["H02"], ["S1", "S2", "S3"], "Leipzig")
    by = {q.key: q for q in qs}
    assert list(by) == ["direction", "intensity", "evidence_type", "sub_criterion", "evidence_sentence"]
    assert by["direction"].values == ["-1", "0", "1"]
    assert by["direction"].options[2].description.startswith("makes the location more attractive")
    assert by["intensity"].values == ["1", "2", "3"]
    assert by["evidence_type"].values == ["fact", "announcement", "opinion"]
    assert by["sub_criterion"].values == ["H02.01", "H02.02", "H02.03", "H02.04", "overarching"]
    assert by["evidence_sentence"].values == ["S1", "S2", "S3"]
    assert "Boundaries: Not H02." in by["direction"].instructions
    assert "H02 good news." in by["direction"].instructions


def test_query_b_without_sub_criteria_and_single_sentence(codebook):
    keys = [q.key for q in query_b_questions(codebook, BY_ID["H05"], ["S7"], "Leipzig")]
    assert keys == ["direction", "intensity", "evidence_type"]


def test_render_state():
    s = render_state({"name": "Leipzig", "region": "Saxony", "country": "Germany", "aliases": ["Region Leipzig"]},
                     "Title", "2025-03-01", [("S1", "One."), ("S2", "Two.")], section=2, n_sections=3)
    assert "Location under assessment: Leipzig, Saxony, Germany (also: Region Leipzig)" in s
    assert "Article section 2 of 3" in s and s.endswith("[S1] One.\n[S2] Two.")
