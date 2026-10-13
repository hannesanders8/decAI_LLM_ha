"""SPEC §11 baseline and §13 evaluation: metric functions, annotation tool, end-to-end runner."""

import csv
import json
import re

import pytest
from typer.testing import CliRunner

from locassess.annotation import AnnotationError, import_annotations, write_template
from locassess.baseline import run_baseline
from locassess.cli import app
from locassess.config import ExtractionCfg, load_config
from locassess.db import repo
from locassess.evaluation import metrics as M
from locassess.evaluation.run_eval import Skipped, extraction_quality, run_metrics
from locassess.prompts import load_prompt
from locassess.providers.llm import LLMResult
from locassess.stage1_extract import run_extraction
from tests.conftest import EXAMPLE_CONFIG
from tests.test_stage0_build import setup  # noqa: F401  (fixture)
from tests.test_stage1_extract import ALL_CRITERIA, answer, frozen  # noqa: F401  (fixtures)
from tests.fakes import FakeDecisionModel

CFG = load_config(EXAMPLE_CONFIG)


@pytest.fixture
def frozen_db(frozen):  # noqa: F811
    """Frozen corpus with the criteria loaded (annotation.main_id references criterion)."""
    repo.upsert_criteria(frozen[0], ALL_CRITERIA)
    return frozen


# --- metric functions -------------------------------------------------------------

def test_basic_metrics():
    assert M.jaccard({1, 2, 3}, {2, 3, 4}) == 0.5
    p = M.binary_prf([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
    assert (p["tp"], p["fp"], p["fn"], p["tn"]) == (2, 1, 1, 1)
    assert p["precision"] == pytest.approx(2 / 3) and p["f1"] == pytest.approx(2 / 3)
    assert M.cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert M.brier([0.9, 0.2], [1, 0]) == pytest.approx((0.01 + 0.04) / 2)
    assert M.average_precision([0.9, 0.8, 0.1], [1, 0, 1]) == pytest.approx((1 + 2 / 3) / 2)
    assert M.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert M.share_identical([[1, 1], [1, 0], [0, 0]]) == pytest.approx(2 / 3)


def test_fleiss_kappa_reference_example():
    # Fleiss (1971) example as given on Wikipedia: 10 items, 14 raters, 5 categories → κ ≈ 0.210
    counts = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
              [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    ratings = [[c for c, n in enumerate(row) for _ in range(n)] for row in counts]
    assert M.fleiss_kappa(ratings) == pytest.approx(0.210, abs=0.001)


def test_reliability_bins_and_quotes():
    bins = M.reliability_bins([0.05, 0.15, 0.95, 1.0], [0, 0, 1, 1], n_bins=10)
    assert bins[0]["count"] == 1 and bins[9]["count"] == 2 and bins[9]["observed_frequency"] == 1.0
    assert M.quote_in_text("„Die  Stadt wächst.“", 'Text. "Die Stadt wächst." Mehr.')
    assert not M.quote_in_text("Leipzig is the best place.", "Leipzig is a place.")
    assert not M.quote_in_text("", "anything")


# --- annotation tool ----------------------------------------------------------------

def test_annotation_template_and_import(frozen_db, tmp_path):
    conn, cid, cb, cb_path = frozen_db
    path = tmp_path / "ann.csv"
    n = write_template(conn, cid, ALL_CRITERIA, "HA", path)
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    assert n == len(rows) == 2 * 26 and rows[0]["annotator"] == "HA"
    rows[0].update(affected="1", direction="-1", intensity="2", evidence_type="fact", sub_id="H01.01", sentence_keys="S2")
    rows[1].update(affected="0")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert import_annotations(conn, path, ALL_CRITERIA) == 2
    a = conn.execute("SELECT * FROM annotation WHERE main_id = 'H01'").fetchone()
    assert (a["affected"], a["direction"], a["sub_id"], a["sentence_keys"]) == (1, -1, "H01.01", "S2")
    assert import_annotations(conn, path, ALL_CRITERIA) == 2      # upsert, no duplicates
    assert conn.execute("SELECT COUNT(*) FROM annotation").fetchone()[0] == 2


@pytest.mark.parametrize("change,msg", [
    (dict(affected="2"), "affected must be 0 or 1"),
    (dict(affected="1", direction="5", intensity="1", evidence_type="fact"), "direction"),
    (dict(affected="1", direction="1", intensity="1", evidence_type="fact", sub_id="H02.01"), "not a sub-criterion"),
    (dict(affected="1", direction="1", intensity="1", evidence_type="fact", sentence_keys="S99"), "do not exist"),
])
def test_annotation_validation(frozen_db, tmp_path, change, msg):
    conn, cid, cb, cb_path = frozen_db
    path = tmp_path / "ann.csv"
    write_template(conn, cid, ALL_CRITERIA, "HA", path)
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    rows[0].update(change)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerow(rows[0])
    with pytest.raises(AnnotationError, match=msg):
        import_annotations(conn, path, ALL_CRITERIA)


# --- baseline -------------------------------------------------------------------------

class BaselineLLM:
    """Valid quote for H01 (taken from A1), invented quote for H02, everything else null."""

    def __init__(self):
        self.calls = []

    def complete(self, prompt_id, variables, json_schema=None, temperature=0, n=1, logprobs=False):
        load_prompt(prompt_id).render(variables)
        self.calls.append((variables, json_schema))
        quote = re.search(r"=== A1 ===.*?(The new logistics centre in Leipzig opened this week\.)",
                          variables["articles"], re.S).group(1)
        crit = {m: {"score": None, "rating": None, "justification": "", "evidence": []}
                for m in json_schema["properties"]["criteria"]["required"]}
        crit["H01"] = {"score": -2.0, "rating": "red", "justification": "x",
                       "evidence": [{"article": "A1", "quote": quote}]}
        crit["H02"] = {"score": 4.5, "rating": "green", "justification": "y",
                       "evidence": [{"article": "A2", "quote": "Leipzig builds a new spaceport."}]}
        parsed = {"criteria": crit, "overall": {"score": 0.5, "judgement": "Mixed."}}
        return LLMResult(texts=[json.dumps(parsed)], parsed=[parsed], prompt_version="1.0", call_ids=[])


def test_baseline_run_and_schema(frozen):  # noqa: F811
    conn, cid, cb, cb_path = frozen
    llm = BaselineLLM()
    bid = run_baseline(conn, llm, cid, ALL_CRITERIA, "gpt-test")
    variables, schema = llm.calls[0]
    assert "=== A1 ===" in variables["articles"] and "=== A2 ===" in variables["articles"]
    assert len(schema["properties"]["criteria"]["required"]) == 26
    row = conn.execute("SELECT * FROM baseline_assessment WHERE baseline_id = ?", (bid,)).fetchone()
    data = json.loads(row["per_criterion_json"])
    assert data["H02"]["score"] == 3.0 and data["H02"]["score_raw"] == 4.5          # clamped to the scale
    assert json.loads(row["overall_json"])["article_labels"]["A1"] > 0
    assert row["repetition"] == 1 and run_baseline(conn, llm, cid, ALL_CRITERIA, "gpt-test") == bid + 1


def test_baseline_needs_frozen_corpus(setup):  # noqa: F811
    from locassess.stage0_corpus import build_corpus
    from tests.test_stage0_build import RUN

    conn, cfg, criteria, llm, search, fetcher = setup
    cid = build_corpus(conn, cfg, criteria, llm, search, fetcher, run_date=RUN)
    with pytest.raises(ValueError, match="not frozen"):
        run_baseline(conn, BaselineLLM(), cid, ALL_CRITERIA, "m")


# --- evaluation end to end ---------------------------------------------------------------

@pytest.fixture
def evaluated(frozen_db, tmp_path):
    conn, cid, cb, cb_path = frozen_db
    model = FakeDecisionModel(answer)
    runs = [run_extraction(conn, cid, model, cb, cb_path, ALL_CRITERIA, ExtractionCfg()) for _ in range(2)]
    arts = [a["article_id"] for a in repo.get_articles(conn, cid, selected_only=True)]
    # gold: H01 affected in every article (agrees with the model), H02 never (model finds it), H04 in the first
    rows = []
    for a in arts:
        for m in [x.main_id for x in ALL_CRITERIA]:
            if m == "H01":
                rows.append((a, m, "HA", 1, -1, 3, "fact", "H01.01", "S1"))
            elif m == "H04" and a == arts[0]:
                rows.append((a, m, "HA", 1, 1, 1, "fact", None, "S1"))
            else:
                rows.append((a, m, "HA", 0, None, None, None, None, None))
    conn.executemany("INSERT INTO annotation VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    for _ in range(2):
        run_baseline(conn, BaselineLLM(), cid, ALL_CRITERIA, "gpt-test")
    from locassess.pipeline import score
    sid, _ = score(conn, CFG, runs[0])
    conn.execute("INSERT INTO report (scoring_run_id, text, attempts, passed, checks_json, prompt_version) VALUES "
                 f"({sid}, 'x', 2, 1, ?, '1.0')", (json.dumps([
                     {"attempt": 1, "checks": [{"rule": "structure", "passed": False}, {"rule": "ratings_match", "passed": True}]},
                     {"attempt": 2, "checks": [{"rule": "structure", "passed": True}, {"rule": "ratings_match", "passed": True}]}]),))
    conn.execute("INSERT INTO call_log (provider, model, purpose, input_tokens, output_tokens, latency_ms, created_at) "
                 "VALUES ('openai_decisions', 'gpt-6-luna', 'extraction_A', 2000000, 0, 500, 'now')")
    conn.commit()
    out = tmp_path / "results"
    return conn, cid, runs, arts, run_metrics(conn, CFG, out_dir=out), out


def read(out, name):
    return list(csv.DictReader(open(out / f"eval_{name}.csv", encoding="utf-8")))


def test_extraction_quality(evaluated):
    conn, cid, runs, arts, results, out = evaluated
    q = read(out, "extraction_quality")
    assert len(q) == 2                                           # two runs × one annotator
    r = q[0]
    n = len(arts)
    # H01: n TP; H02: model positive only where section 2 exists → FP; H04: 1 FN
    n_h02 = sum(1 for a in arts if 2 in {s["section"] for s in repo.get_sentences(conn, a)})
    assert (int(r["tp"]), int(r["fp"]), int(r["fn"])) == (n, n_h02, 1)
    assert int(r["cells"]) == 26 * n and float(r["direction_agreement"]) == 1.0
    assert float(r["evidence_type_agreement"]) == 1.0 and float(r["sub_criterion_agreement"]) == 1.0
    by = read(out, "extraction_quality_by_criterion")
    assert {(x["main_id"], x["gold_positive"]) for x in by if x["extraction_run_id"] == str(runs[0])} >= \
        {("H01", str(n)), ("H04", "1"), ("H05", "0")}


def test_hallucination_and_baseline(evaluated):
    conn, cid, runs, arts, results, out = evaluated
    h = {(x["source"], x["id"]): x for x in read(out, "hallucination")}
    assert all(float(h[("hybrid", str(r))]["rate"]) == 0.0 for r in runs)
    b = [x for x in h.values() if x["source"] == "baseline"]
    assert len(b) == 2 and all(float(x["rate"]) == 0.5 for x in b)   # 1 valid + 1 invented quote
    s = read(out, "baseline_comparison")[0]
    assert float(s["share_backed_hybrid"]) == 1.0 and float(s["share_backed_baseline"]) == 0.5
    assert float(s["mean_sd_baseline"]) == 0.0 and float(s["criteria_rated_baseline_mean"]) == 2.0
    detail = {x["main_id"]: x for x in read(out, "baseline_comparison_by_criterion")}
    assert detail["H01"]["baseline_rating"] == "red" and detail["H01"]["hybrid_rating"] == "red"


def test_stability_calibration_robustness_reports_cost(evaluated):
    conn, cid, runs, arts, results, out = evaluated
    st = read(out, "extraction_stability")[0]
    assert int(st["runs"]) == 2 and float(st["share_identical_affected"]) == 1.0      # deterministic fake
    cal = read(out, "calibration")[0]
    assert float(cal["brier"]) >= 0 and int(cal["positives"]) == len(arts) + 1
    assert len(read(out, "calibration_reliability_bins")) == 2 * 10
    var = read(out, "scoring_robustness_variants")
    assert len(var) == 2 * 36 and {x["variant"] for x in var} >= {"tw=default|hl=default|thr=default"}
    assert all(float(x["share_same_rating"]) == 1.0 for x in var if x["variant"] == "tw=default|hl=default|thr=default")
    rel = {x["rule"]: x for x in read(out, "report_reliability")}
    assert float(rel["structure"]["failure_rate_first_attempt"]) == 1.0
    assert float(rel["structure"]["failure_rate_all_attempts"]) == 0.5 and float(rel["(all reports)"]["mean_attempts"]) == 2
    cost = read(out, "cost_latency_by_purpose")[0]
    assert float(cost["cost_usd"]) == pytest.approx(0.2)                  # 2M tokens × $0.10 / 1M
    assert any("skipped" in o for n, o in results if n == "search_reproducibility")


def test_quality_skipped_without_annotations(frozen):  # noqa: F811
    conn, *_ = frozen
    with pytest.raises(Skipped):
        extraction_quality(conn, CFG)


def test_cli_evaluate(evaluated, tmp_path):
    conn, cid, runs, arts, results, out = evaluated
    db = conn.execute("PRAGMA database_list").fetchone()[2]
    r = CliRunner().invoke(app, ["evaluate", "--metric", "hallucination", "--config", str(EXAMPLE_CONFIG),
                                 "--db", db, "--out", str(tmp_path / "cli")])
    assert r.exit_code == 0 and "eval_hallucination.csv" in r.output, r.output
    r = CliRunner().invoke(app, ["evaluate", "--metric", "nonsense", "--config", str(EXAMPLE_CONFIG), "--db", db])
    assert r.exit_code != 0
