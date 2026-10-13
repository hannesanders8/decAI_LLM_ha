"""UI data layer (SPEC §10.3) and a headless smoke test of the Streamlit app."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from locassess import ui_data
from locassess.cli import app
from locassess.db import repo
from tests.conftest import EXAMPLE_CONFIG
from tests.test_stage2_db import _seed

ROOT = Path(__file__).resolve().parents[1]
REPORT = "## Summary\nH20 Population potential (red) [E1].\n\n## Red flags\n- R1 [E2]\n"


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "ui.db"
    _seed(path)
    conn = repo.connect(path)
    conn.execute("UPDATE corpus SET location_json = ? WHERE corpus_id = 1",
                 (json.dumps({"name": "Leipzig", "country": "Germany", "aliases": []}),))
    conn.commit()
    runner = CliRunner()
    for label, params in (("base", None), ("strict", "scoring:\n  rating_thresholds: {red: -2.5}\n")):
        args = ["score", "--extraction-run", "4", "--config", str(EXAMPLE_CONFIG), "--db", str(path), "--label", label]
        if params:
            (tmp_path / "p.yaml").write_text(params, encoding="utf-8")
            args += ["--params", str(tmp_path / "p.yaml")]
        assert runner.invoke(app, args).exit_code == 0
    conn.execute("INSERT INTO report (scoring_run_id, text, attempts, passed, checks_json, prompt_version) "
                 "VALUES (1, ?, 1, 1, '[]', '1.0')", (REPORT,))
    conn.commit()
    return path


def test_scoring_runs_and_titles(db):
    runs = ui_data.scoring_runs(repo.connect(db))
    assert list(runs["scoring_run_id"]) == [2, 1]
    r1 = runs[runs["scoring_run_id"] == 1].iloc[0]
    assert (r1["location"], r1["report"], r1["model"]) == ("Leipzig", "passed", "mock")
    assert ui_data.run_title(r1.to_dict()) == "#1 Leipzig — mock rep None (corpus 1) · base"


def test_criteria_table_and_drill_down(db):
    conn = repo.connect(db)
    t = ui_data.criteria_table(conn, 1)
    assert len(t) == 26 and t["factor"].nunique() == 6
    h20 = t[t["main_id"] == "H20"].iloc[0]
    assert (round(h20["score"], 3), h20["rating"], h20["status"], h20["evidence"], h20["red_flags"]) == \
        (-2.212, "red", "sufficient", 3, "R1, R2")
    ev = ui_data.evidence_table(conn, 1, main_id="H20")
    assert list(ev["evidence"]) == ["E1", "E2", "E3"] and set(ev["outlet"]) == {"A", "B", "C"}
    assert {"sub_criterion", "sentence", "url", "title", "confidence"} <= set(ev.columns)
    assert ui_data.evidence_table(conn, 1, main_id="H01").empty
    flags = ui_data.red_flags(conn, 1)
    assert [(f["rule"], list(f["evidence"]["evidence"])) for f in flags] == [("R1", ["E2"]), ("R2", ["E1", "E2"])]


def test_sliders_recompute_overall_live(db):
    t = ui_data.criteria_table(repo.connect(db), 1)
    assert ui_data.overall_with_sliders(t, {}).score == pytest.approx(-2.212, abs=0.005)
    assert ui_data.overall_with_sliders(t, {"H20": 0}).score is None           # slider 0 excludes
    o = ui_data.overall_with_sliders(t, {"H20": 7, "H01": 10})                  # H01 has no evidence
    assert o.score == pytest.approx(-2.212, abs=0.005) and o.weights == {"H20": 1.0}
    assert o.factor_scores["Social factors"] == pytest.approx(-2.212, abs=0.005)


def test_ranking_and_comparison(db):
    conn = repo.connect(db)
    r = ui_data.ranking(conn, [1, 2], {})
    assert list(r["rank"]) == [1, 2] and len(r) == 2
    c = ui_data.comparison(conn, [1, 2])
    assert len(c) == 26 and len(c.columns) == 5
    h20 = c[c["main_id"] == "H20"].iloc[0]
    assert h20.iloc[3] == "-2.2 red" and h20.iloc[4] == "-2.2 yellow"        # stricter red threshold in run 2
    assert c[c["main_id"] == "H01"].iloc[0].iloc[3] == "—"


def test_exports(db):
    conn = repo.connect(db)
    md = ui_data.profile_markdown(conn, 1, {"H20": 2})
    assert md.startswith("# Location profile: Leipzig") and "| H20 |" in md and "H20=2" in md
    assert "### Summary" in md                                                  # validated report included
    assert "_No validated report available._" in ui_data.profile_markdown(conn, 2, {})
    pdf = ui_data.markdown_to_pdf(md)
    assert pdf[:4] == b"%PDF" and len(pdf) > 1000
    csv = ui_data.csv_bytes(ui_data.criteria_table(conn, 1)).decode("utf-8")
    assert csv.splitlines()[0].startswith("main_id,factor,criterion,score")


def test_streamlit_app_smoke(db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LOCASSESS_DB", str(db))
    monkeypatch.setenv("LOCASSESS_CONFIG", str(EXAMPLE_CONFIG))
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert any("Leipzig" in h.value for h in at.header)
    assert at.metric[0].value == "-2.21"
    assert len(at.sidebar.slider) == 26
    assert any("R1" in e.value for e in at.error)                               # red flags at the top

    at.slider(key="w_H20").set_value(0).run()                                   # live recalculation
    assert not at.exception and at.metric[0].value == "n/a"


def test_streamlit_app_empty_db(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    path = tmp_path / "empty.db"
    runner = CliRunner()
    assert runner.invoke(app, ["load-criteria", "--db", str(path)]).exit_code == 0
    monkeypatch.setenv("LOCASSESS_DB", str(path))
    monkeypatch.setenv("LOCASSESS_CONFIG", str(EXAMPLE_CONFIG))
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert any("No location profiles yet" in i.value for i in at.info)
