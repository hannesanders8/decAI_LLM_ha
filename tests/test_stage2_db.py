"""Stage 2 end to end: evidence in the DB → `locassess score` → stored scoring run."""

import json

from typer.testing import CliRunner

from locassess.cli import app
from locassess.criteria import load_criteria_csv
from locassess.db import repo
from tests.conftest import CRITERIA_CSV, EXAMPLE_CONFIG

runner = CliRunner()


def _seed(db):
    conn = repo.connect(db)
    repo.init_db(conn)
    repo.upsert_criteria(conn, load_criteria_csv(CRITERIA_CSV))
    conn.execute("INSERT INTO corpus (corpus_id, location_json, period_from, period_to, created_at, "
                 "config_hash) VALUES (1, '{}', '2021-10-01', '2026-10-13', 'now', 'h')")
    for aid, outlet in [(101, "A"), (102, "B"), (103, "C")]:
        conn.execute("INSERT INTO article (article_id, corpus_id, outlet) VALUES (?, 1, ?)", (aid, outlet))
    conn.execute("INSERT INTO extraction_run (extraction_run_id, corpus_id, model) VALUES (4, 1, 'mock')")
    conn.executemany(
        "INSERT INTO evidence (evidence_id, extraction_run_id, article_id, main_id, direction, intensity, "
        "evidence_type, event_date, confidence) VALUES (?, 4, ?, 'H20', ?, ?, ?, ?, ?)",
        [(1, 101, -1, 2, "fact", "2025-03", 0.62),
         (2, 102, -1, 3, "fact", "2026-06", 0.85),
         (3, 103, 1, 2, "announcement", "2022-11", 0.80)],
    )
    conn.commit()
    conn.close()


def test_score_command_stores_profile(tmp_path):
    db = tmp_path / "s.db"
    _seed(db)
    r = runner.invoke(app, ["score", "--extraction-run", "4", "--config", str(EXAMPLE_CONFIG),
                            "--db", str(db), "--label", "default"])
    assert r.exit_code == 0, r.output
    assert "scoring_run_id=1" in r.output and "H20  -2.2  red" in r.output

    conn = repo.connect(db)
    run = conn.execute("SELECT * FROM scoring_run").fetchone()
    assert json.loads(run["params_json"])["run_month"] == "2026-10"
    rows = {x["main_id"]: x for x in conn.execute("SELECT * FROM criterion_score WHERE scoring_run_id = 1")}
    assert len(rows) == 26
    h20 = rows["H20"]
    assert abs(h20["score"] - (-2.212)) < 0.005
    assert (h20["rating"], h20["status"], h20["trend"]) == ("red", "sufficient", "deteriorating")
    assert {f["rule"] for f in json.loads(h20["red_flags_json"])} == {"R1", "R2"}
    assert json.loads(h20["evidence_ids_json"]) == [1, 2, 3]
    assert rows["H01"]["status"] == "no_statement" and rows["H01"]["score"] is None
    overall = conn.execute("SELECT * FROM overall_score").fetchone()
    assert abs(overall["score"] - (-2.212)) < 0.005
    assert json.loads(overall["factor_scores_json"])["Social factors"] is not None


def test_params_override(tmp_path):
    db = tmp_path / "s.db"
    _seed(db)
    p = tmp_path / "params.yaml"
    p.write_text("scoring:\n  rating_thresholds: {red: -2.5}\n", encoding="utf-8")
    r = runner.invoke(app, ["score", "--extraction-run", "4", "--config", str(EXAMPLE_CONFIG),
                            "--params", str(p), "--db", str(db)])
    assert r.exit_code == 0, r.output
    conn = repo.connect(db)
    assert conn.execute("SELECT rating FROM criterion_score WHERE main_id='H20'").fetchone()[0] == "yellow"
    params = json.loads(conn.execute("SELECT params_json FROM scoring_run").fetchone()[0])
    assert params["red_threshold"] == -2.5 and params["green_threshold"] == 1.0
