"""SPEC §10.1–10.2: report profile, validation rules, retries."""

import json

import pytest
from typer.testing import CliRunner

from locassess.cli import app
from locassess.db import repo
from locassess.prompts import load_prompt
from locassess.providers.llm import LLMResult
from locassess.stage3_report import (
    build_profile_json, generate_report, paragraphs, report_plan, split_sections, validate_report,
)
from tests.conftest import EXAMPLE_CONFIG
from tests.test_stage2_db import _seed

PROFILE = {
    "location": {"name": "Leipzig"}, "period": {"from": "2021-10-13", "to": "2026-10-13"}, "overall_score": 0.26,
    "red_flags": [{"rule": "R1", "criterion": "H20", "description": "x", "evidence_ids": ["E2"]},
                  {"rule": "R2", "criterion": "H20", "description": "x", "evidence_ids": ["E1", "E2"]}],
    "criteria": [
        {"id": "H01", "name": "Natural resources", "score": 0.4, "rating": "yellow", "status": "thin",
         "trend": "n/a", "evidence_ids": ["E5"]},
        {"id": "H05", "name": "Urban agglomeration", "score": None, "rating": None, "status": "no_statement",
         "trend": "n/a", "evidence_ids": []},
        {"id": "H10", "name": "Local and national government legislation/Policy", "score": 2.6, "rating": "green",
         "status": "sufficient", "trend": "stable", "evidence_ids": ["E4"]},
        {"id": "H20", "name": "Population potential", "score": -2.2, "rating": "red", "status": "sufficient",
         "trend": "deteriorating", "evidence_ids": ["E1", "E2", "E3"]},
    ],
    "evidence": {f"E{i}": {} for i in range(1, 6)},
}
PROFILE["report_plan"] = report_plan(PROFILE["criteria"])

GOOD = """## Summary
Leipzig shows a mixed picture with an overall score of +0.26. H20 Population potential (red) carries two red flags [E1, E2]. H10 Local and national government legislation/Policy (green) is the main strength [E4].

## Red flags
- R1 for H20 Population potential (red): a plant cut production because of a staff shortage [E2].
- R2 for H20 Population potential (red): repeated negative evidence from different articles [E1, E2].

## Strengths
- H10 Local and national government legislation/Policy (green), score +2.6, stable: new state funding [E4].

## Weaknesses
- H20 Population potential (red), score -2.2, deteriorating: unfilled apprenticeships and a production cut [E1, E2, E3].

## Thinly evidenced criteria
- H01 Natural resources (yellow), score +0.4, rests on a single article about land [E5].

## No evidence
H05 Urban agglomeration
"""


def failed(text):
    return {c.rule: c.violations for c in validate_report(text, PROFILE) if not c.passed}


def test_report_plan():
    assert PROFILE["report_plan"] == {"strengths": ["H10"], "weaknesses": ["H20"], "thin": ["H01"],
                                      "no_evidence": ["H05"]}
    assert report_plan([{"id": "H02", "status": "sufficient", "score": 0.0}])["weaknesses"] == ["H02"]


def test_good_report_passes():
    assert failed(GOOD) == {}


def test_split_sections_and_paragraphs():
    s = split_sections(GOOD)
    assert list(s) == ["Summary", "Red flags", "Strengths", "Weaknesses", "Thinly evidenced criteria", "No evidence"]
    assert len(paragraphs(s["Red flags"])) == 2


@pytest.mark.parametrize("old,new,rule", [
    ("## Strengths", "## Advantages", "structure"),
    ("[E4].\n\n## Weaknesses", "[E9].\n\n## Weaknesses", "cited_ids_exist"),
    ("[E1, E2, E3]", "[E1,E2,E3]", "cited_ids_exist"),                       # malformed format
    ("new state funding [E4].", "new state funding.", "paragraphs_cite"),
    ("H10 Local and national government legislation/Policy (green), score", "H10 Local and national government legislation/Policy (yellow), score", "ratings_match"),
    ("is the main strength [E4]", "is the main strength (green) [E4]", "ratings_match"),   # rating without ID
    ("repeated negative evidence from different articles [E1, E2]", "repeated negative evidence [E2]", "red_flags_complete"),
    ("H01 Natural resources (yellow), score", "H01 Natural resources (green, +0.4), score", "ratings_match"),
    ("H01 Natural resources (yellow), score", "H01 Natural resources (RED), score", "ratings_match"),
    ("H05 Urban agglomeration\n", "None.\n", "no_evidence_complete"),
    ("rests on a single article about land [E5].", "rests on a single article about land [E5]; H05 may also be weak.", "no_evidence_complete"),
    ("- H01 Natural resources (yellow), score +0.4, rests on a single article about land [E5].", "None.", "sections_complete"),
    ("is the main strength [E4].", "is the main strength [E4]. It is a good place. It is cheap.", "summary_length"),
])
def test_each_rule_detects_its_violation(old, new, rule):
    assert old in GOOD
    assert rule in failed(GOOD.replace(old, new))


def test_rating_with_score_in_parentheses_is_checked_and_accepted():
    assert failed(GOOD.replace("H01 Natural resources (yellow), score +0.4,", "H01 Natural resources (yellow, +0.4)")) == {}


def test_none_allowed_only_for_empty_sections():
    profile = json.loads(json.dumps(PROFILE))
    profile["red_flags"] = []
    text = GOOD.split("## Red flags")[0] + "## Red flags\nNone.\n\n## Strengths" + GOOD.split("## Strengths")[1]
    assert {c.rule for c in validate_report(text, profile) if not c.passed} == set()


class ScriptedLLM:
    def __init__(self, texts):
        self.texts, self.calls = list(texts), []

    def complete(self, prompt_id, variables, json_schema=None, temperature=0, n=1, logprobs=False):
        load_prompt(prompt_id).render(variables)
        self.calls.append(variables)
        return LLMResult(texts=[self.texts.pop(0)], prompt_version=load_prompt(prompt_id).version, call_ids=[1])


@pytest.fixture
def scored_db(tmp_path):
    db = tmp_path / "r.db"
    _seed(db)
    r = CliRunner().invoke(app, ["score", "--extraction-run", "4", "--config", str(EXAMPLE_CONFIG), "--db", str(db)])
    assert r.exit_code == 0, r.output
    return db


def test_build_profile_json_from_db(scored_db):
    conn = repo.connect(scored_db)
    p = build_profile_json(conn, 1)
    assert p["period"] == {"from": "2021-10-01", "to": "2026-10-13"}
    h20 = next(c for c in p["criteria"] if c["id"] == "H20")
    assert (h20["score"], h20["rating"], h20["status"], h20["trend"]) == (-2.2, "red", "sufficient", "deteriorating")
    assert h20["evidence_ids"] == ["E1", "E2", "E3"] and h20["name"] == "Population potential"
    assert {(f["rule"], tuple(f["evidence_ids"])) for f in p["red_flags"]} == {("R1", ("E2",)), ("R2", ("E1", "E2"))}
    assert p["evidence"]["E2"]["outlet"] == "B" and p["evidence"]["E2"]["date"] == "2026-06"
    assert p["report_plan"]["weaknesses"] == ["H20"] and len(p["report_plan"]["no_evidence"]) == 25
    assert "text" not in json.dumps(p["evidence"])   # no raw article text in the profile


def _good_for(profile):
    """A valid report for the seeded DB profile (H20 only)."""
    ne = ", ".join(f"{c['id']} {c['name']}" for c in profile["criteria"] if c["status"] == "no_statement")
    return (f"## Summary\nH20 Population potential (red) is the only evidenced criterion [E1, E2, E3].\n\n"
            f"## Red flags\n- R1: a recent production cut [E2].\n- R2: repeated negative evidence [E1, E2].\n\n"
            f"## Strengths\nNone.\n\n## Weaknesses\n- H20 Population potential (red), score -2.2 [E1, E2, E3].\n\n"
            f"## Thinly evidenced criteria\nNone.\n\n## No evidence\n{ne}\n")


def test_generate_retries_with_violations(scored_db):
    conn = repo.connect(scored_db)
    good = _good_for(build_profile_json(conn, 1))
    llm = ScriptedLLM([good.replace("[E1, E2, E3].\n\n## Thin", ".\n\n## Thin"), good])
    rid = generate_report(conn, llm, 1, max_attempts=3)
    row = conn.execute("SELECT * FROM report WHERE report_id = ?", (rid,)).fetchone()
    assert (row["passed"], row["attempts"], row["prompt_version"]) == (1, 2, "1.0")
    assert row["text"] == good.strip()
    assert llm.calls[0]["violations_block"] == ""
    assert "paragraph without evidence citation" in llm.calls[1]["violations_block"]
    checks = json.loads(row["checks_json"])
    assert [a["passed"] for a in checks] == [False, True]
    assert {c["rule"] for c in checks[0]["checks"] if not c["passed"]} == {"paragraphs_cite"}


def test_generate_stores_failure_after_max_attempts(scored_db):
    conn = repo.connect(scored_db)
    rid = generate_report(conn, ScriptedLLM(["no structure"] * 3), 1, max_attempts=3)
    row = conn.execute("SELECT * FROM report WHERE report_id = ?", (rid,)).fetchone()
    assert (row["passed"], row["attempts"]) == (0, 3)


def test_cli_report(scored_db, tmp_path, monkeypatch):
    conn = repo.connect(scored_db)
    good = _good_for(build_profile_json(conn, 1))
    monkeypatch.setattr("locassess.providers.llm.make_llm", lambda *a, **k: ScriptedLLM([good]))
    r = CliRunner().invoke(app, ["report", "--scoring-run", "1", "--config", str(EXAMPLE_CONFIG),
                                 "--db", str(scored_db), "--out", str(tmp_path / "res")])
    assert r.exit_code == 0 and "passed after 1 attempt" in r.output, r.output
    assert (tmp_path / "res" / "report_scoring1.md").read_text(encoding="utf-8").startswith("## Summary")
