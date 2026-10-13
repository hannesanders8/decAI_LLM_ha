"""Stage 1 end to end on a frozen corpus with a fake decision model; then Stage 2 on the result."""

import json

import pytest

from locassess.codebook_draft import write_codebook_draft
from locassess.config import ExtractionCfg, ScoringCfg
from locassess.criteria import load_codebook, load_criteria_csv
from locassess.db import repo
from locassess.stage0_corpus import build_corpus, freeze
from locassess.stage1_extract import run_extraction, run_summary
from locassess.stage2_score import ScoringParams, build_profile
from tests.conftest import CRITERIA_CSV
from tests.fakes import FakeDecisionModel, FakeLLM
from tests.test_stage0_build import RUN, setup  # noqa: F401  (fixture)

ALL_CRITERIA = load_criteria_csv(CRITERIA_CSV)


def answer(state, q):
    """H01 affected in every section; H02 only in sections numbered 2; H03 refused; fixed B answers."""
    if q.key.endswith("_affected"):
        main = q.key[:3]
        if main == "H03":
            return None
        yes = {"H01": 0.9, "H02": 0.8 if "section 2 of" in state else 0.2}.get(main, 0.1)
        return {"yes": yes, "no": 1 - yes}
    return {"direction": {"-1": 0.7, "0": 0.1, "1": 0.2}, "intensity": {"1": 0.1, "2": 0.3, "3": 0.6},
            "evidence_type": {"fact": 0.8, "announcement": 0.1, "opinion": 0.1},
            "sub_criterion": {q.values[0]: 0.9},
            "evidence_sentence": {q.values[1]: 0.5, q.values[0]: 0.4}}[q.key]


def sections_of(conn, article_id):
    return sorted({s["section"] for s in repo.get_sentences(conn, article_id)})


@pytest.fixture
def frozen(setup, tmp_path):  # noqa: F811
    conn, cfg, criteria, llm, search, fetcher = setup
    cid = build_corpus(conn, cfg, criteria, llm, search, fetcher, run_date=RUN)
    freeze(conn, cid, max_options_per_question=2)           # 3 sentences → sections [S1, S2], [S3]
    cb_path = tmp_path / "codebook.yaml"
    write_codebook_draft(cb_path, FakeLLM(), ALL_CRITERIA, "gpt-test")
    return conn, cid, load_codebook(cb_path, ALL_CRITERIA), cb_path


def test_extraction_run_end_to_end(frozen):
    conn, cid, cb, cb_path = frozen
    run_id = run_extraction(conn, cid, FakeDecisionModel(answer), cb, cb_path, ALL_CRITERIA, ExtractionCfg())

    run = repo.get_extraction_run(conn, run_id)
    assert (run["model"], run["repetition"], run["query_definition_version"], run["model_version"]) == \
        ("fake", 1, "1.0", "fake-1")
    assert run["codebook_version"].startswith("0.1+") and run["confidence_method"] == "native"

    articles = repo.get_articles(conn, cid, selected_only=True)
    n_sections = sum(len(sections_of(conn, a["article_id"])) for a in articles)
    assert n_sections > len(articles)                       # sections really were used
    n_second = sum(1 for a in articles if 2 in sections_of(conn, a["article_id"]))
    s = run_summary(conn, run_id)
    assert s["responses_A"] == n_sections
    assert s["responses_B"] == n_sections + n_second      # H01 everywhere, H02 only in section 2

    ev = conn.execute("SELECT * FROM evidence WHERE extraction_run_id = ? ORDER BY article_id, main_id",
                      (run_id,)).fetchall()
    assert {e["main_id"] for e in ev} == {"H01", "H02"}     # H03 refused, others below the threshold
    assert sum(1 for e in ev if e["main_id"] == "H01") == len(articles)   # sections merged: one per article
    e = ev[0]
    art = {a["article_id"]: a for a in articles}[e["article_id"]]
    assert e["event_date"] == art["published_at"][:7]
    assert (e["direction"], e["intensity"], e["evidence_type"], e["sub_id"]) == (-1, 3, "fact", "H01.01")
    assert e["confidence"] == pytest.approx(0.6) and e["needs_review"] == 0
    keys = json.loads(e["sentence_keys"])
    assert len(keys) == 2                                     # second sentence within the gap
    texts = {x["sentence_key"]: x["text"] for x in repo.get_sentences(conn, e["article_id"])}
    assert e["sentence_text"] == " ".join(texts[k] for k in keys)

    # full distributions are stored, also for criteria that are not affected, plus refusals
    a_resp = json.loads(conn.execute(
        "SELECT distributions_json FROM model_response WHERE extraction_run_id=? AND query_type='A' LIMIT 1",
        (run_id,)).fetchone()[0])
    assert len(a_resp["distributions"]) == 25 and a_resp["refusals"] == ["H03_affected"]

    # Stage 2 runs on the extracted evidence
    params = ScoringParams.from_config(ScoringCfg(), run_date="2026-10", period_from="2021-10")
    profile = build_profile(repo.load_evidence(conn, run_id), [m.main_id for m in ALL_CRITERIA], params)
    assert profile.criteria["H01"].rating == "red" and profile.criteria["H05"].status == "no_statement"


def test_repetitions_are_numbered(frozen):
    conn, cid, cb, cb_path = frozen
    model = FakeDecisionModel(answer)
    r1 = run_extraction(conn, cid, model, cb, cb_path, ALL_CRITERIA, ExtractionCfg())
    r2 = run_extraction(conn, cid, model, cb, cb_path, ALL_CRITERIA, ExtractionCfg())
    assert [repo.get_extraction_run(conn, r)["repetition"] for r in (r1, r2)] == [1, 2]


def test_requires_frozen_corpus(setup, tmp_path):  # noqa: F811
    conn, cfg, criteria, llm, search, fetcher = setup
    cid = build_corpus(conn, cfg, criteria, llm, search, fetcher, run_date=RUN)
    with pytest.raises(ValueError, match="not frozen"):
        run_extraction(conn, cid, FakeDecisionModel(answer), None, tmp_path / "x", ALL_CRITERIA, ExtractionCfg())
