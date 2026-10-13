"""Stage 0 end to end with mocked LLM, search and fetching."""

import csv
import random
import sqlite3
from datetime import date

import pytest

from locassess.config import load_config
from locassess.criteria import load_criteria_csv
from locassess.db import repo
from locassess.stage0_corpus import build_corpus, export_selection_log, freeze, import_corpus, selection_log
from tests.conftest import CRITERIA_CSV, EXAMPLE_CONFIG
from tests.fakes import FakeFetcher, FakeLLM, FakeSearch, article_html

RUN = date(2026, 10, 13)


def words(seed, n=120):
    rnd = random.Random(seed)
    return " ".join(f"w{rnd.randint(0, 10**6)}" for _ in range(n))


def body(seed, n=120, place="Leipzig"):
    return f"The new logistics centre in {place} opened this week. {words(seed, n)}."


PAGES = {
    "https://www.lvz.de/a1": article_html("A1", "2025-05-01", body(1)),
    "https://www.lvz.de/a2": article_html("A2", "2025-06-01", body(2), lang=None),
    "https://www.mdr.de/a4": article_html("A4", "2019-01-01", body(4)),
    "https://www.mdr.de/a5": article_html("A5", "2025-01-01", body(5, place="Dresden")),
    "https://www.mdr.de/a6": article_html("A6", "2025-01-01", body(6, n=10)),
    "https://sport.mdr.de/a7": article_html("A7", "2025-05-02", body(1)),  # wire copy of A1
    "https://www.mdr.de/a8": 404,
    "https://www.mdr.de/a9": article_html("A9", "2023-01-01", body(9)),
    "https://www.lvz.de/a10": article_html("A10", "2024-03-01", body(10), lang="tr"),
}
RESULTS = {
    "H01": ["https://www.lvz.de/a1", "https://notlisted.com/a3", "https://www.mdr.de/a4"],
    "H02": ["https://www.lvz.de/a2", "https://www.mdr.de/a5", "https://www.mdr.de/a6",
            "https://www.lvz.de/a1/?utm_source=x"],
    "H03": ["https://sport.mdr.de/a7", "https://www.mdr.de/a8", "https://www.mdr.de/a9",
            "https://www.lvz.de/a10"],
}
RELEVANCE = {"A1": 2, "A2": 2, "A7": 2, "A9": 0, "A10": 1}


@pytest.fixture
def setup(tmp_path, conn):
    wl = tmp_path / "whitelist.csv"
    wl.write_text("domain,outlet_name,media_type,language,country\n"
                  "lvz.de,Leipziger Volkszeitung,regional,de,DE\nmdr.de,MDR,regional,de,DE\n", encoding="utf-8")
    cfg = load_config(EXAMPLE_CONFIG)
    cfg.whitelist_file = wl
    cfg.corpus.queries_per_criterion = 1
    cfg.corpus.target_articles = 3
    cfg.corpus.max_per_year = 1
    cfg.corpus.min_words = 50
    criteria = load_criteria_csv(CRITERIA_CSV)[:3]
    llm = FakeLLM(relevance=lambda v: RELEVANCE[v["title"]])
    search = FakeSearch(lambda q: RESULTS[q.split()[1]] if q.endswith("en 0") else [])
    fetcher = FakeFetcher(PAGES)
    return conn, cfg, criteria, llm, search, fetcher


def _by_url(conn, corpus_id):
    return {a["url"]: a for a in repo.get_articles(conn, corpus_id)}


def test_build_applies_all_rules(setup):
    conn, cfg, criteria, llm, search, fetcher = setup
    cid = build_corpus(conn, cfg, criteria, llm, search, fetcher, run_date=RUN)
    arts = _by_url(conn, cid)

    corpus = repo.get_corpus(conn, cid)
    assert (corpus["period_from"], corpus["period_to"], corpus["frozen"]) == ("2021-10-13", "2026-10-13", 0)
    assert len(conn.execute("SELECT * FROM search_query WHERE corpus_id=?", (cid,)).fetchall()) == 6  # 3 × en/de
    assert search.calls[0]["domains"] == ["lvz.de", "mdr.de"]

    reasons = {u.split("/")[-1]: a["exclusion_reason"] for u, a in arts.items()}
    assert reasons == {
        "a1": "selection:max_per_year", "a2": None, "a3": "filter:domain_not_whitelisted",
        "a4": "filter:outside_period", "a5": "filter:location_not_mentioned", "a6": "filter:too_short",
        "a7": f"filter:duplicate_of_{arts['https://www.lvz.de/a1']['article_id']}", "a8": "fetch:http_404",
        "a9": "relevance:below_min", "a10": None,
    }
    selected = {u.split("/")[-1] for u, a in arts.items() if a["selected"]}
    assert selected == {"a2", "a10"}
    # non-whitelisted URLs are never downloaded; the duplicate URL is not fetched twice
    assert "https://notlisted.com/a3" not in fetcher.calls and len(fetcher.calls) == 9

    a2 = arts["https://www.lvz.de/a2"]
    assert a2["outlet"] == "Leipziger Volkszeitung" and a2["language"] == "de"  # no page language → whitelist
    assert arts["https://www.lvz.de/a10"]["language"] == "tr"                   # page language wins
    assert a2["published_at"] == "2025-06-01" and a2["word_count"] > 50 and len(a2["text_sha256"]) == 64
    assert a2["relevance"] == 2 and a2["relevance_reason"]

    dup_hit = conn.execute("SELECT * FROM search_hit WHERE url LIKE '%utm_source%'").fetchone()
    assert dup_hit["status"] == "duplicate_url"


def test_selection_log(setup, tmp_path):
    conn, cfg, criteria, llm, search, fetcher = setup
    cid = build_corpus(conn, cfg, criteria, llm, search, fetcher, run_date=RUN)
    summary, per_article = selection_log(conn, cid)
    counts = {r["step"]: r["count"] for r in summary if r["step"] != "excluded"}
    assert counts == {"queries": 6, "search_hits": 11, "unique_urls": 10, "whitelisted": 9, "fetched": 8,
                      "passed_filter": 4, "relevant": 3, "selected": 2}
    assert len(per_article) == 10
    s, a = export_selection_log(conn, cid, tmp_path / "results")
    rows = list(csv.DictReader(open(s, encoding="utf-8")))
    assert {"step": "excluded", "count": "1", "detail": "fetch:http_404"} in rows
    assert len(list(csv.DictReader(open(a, encoding="utf-8")))) == 10


def test_freeze_makes_corpus_read_only(setup):
    conn, cfg, criteria, llm, search, fetcher = setup
    cid = build_corpus(conn, cfg, criteria, llm, search, fetcher, run_date=RUN)
    assert freeze(conn, cid, max_options_per_question=10) == 2
    a2 = _by_url(conn, cid)["https://www.lvz.de/a2"]
    sents = repo.get_sentences(conn, a2["article_id"])
    assert [s["sentence_key"] for s in sents[:2]] == ["S1", "S2"]
    assert sents[0]["text"] == "A2"  # trafilatura keeps the headline as the first line
    assert sents[1]["text"].startswith("The new logistics centre")
    assert max(s["section"] for s in sents) == (len(sents) - 1) // 10 + 1

    with pytest.raises(repo.CorpusFrozenError):
        repo.update_article(conn, a2["article_id"], title="changed")
    with pytest.raises(repo.CorpusFrozenError):
        repo.insert_article(conn, cid, url="https://lvz.de/new")
    with pytest.raises(repo.CorpusFrozenError):
        repo.replace_sentences(conn, a2["article_id"], [])
    with pytest.raises(repo.CorpusFrozenError):
        freeze(conn, cid, 10)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE article SET text = 'x' WHERE article_id = ?", (a2["article_id"],))


def test_freeze_needs_selected_articles(conn):
    cid = repo.create_corpus(conn, "{}", "2021-10-13", "2026-10-13", "h")
    with pytest.raises(ValueError):
        freeze(conn, cid, 255)


def test_manual_import(setup, tmp_path):
    conn, cfg, criteria, llm, search, fetcher = setup
    p = tmp_path / "urls.csv"
    p.write_text("url,note\nhttps://www.lvz.de/a2,hand-picked\nhttps://notlisted.com/a3,\n"
                 "https://www.lvz.de/a10,\n", encoding="utf-8")
    cid = import_corpus(conn, cfg, criteria, llm, p, fetcher, run_date=RUN)
    arts = _by_url(conn, cid)
    assert arts["https://notlisted.com/a3"]["exclusion_reason"] == "filter:domain_not_whitelisted"
    assert {u for u, a in arts.items() if a["selected"]} == {"https://www.lvz.de/a2", "https://www.lvz.de/a10"}
    q = conn.execute("SELECT * FROM search_query WHERE corpus_id = ?", (cid,)).fetchone()
    assert q["query_text"] == "manual_csv:urls.csv" and q["main_id"] is None and q["hit_count"] == 3


def test_cli_freeze_and_log(setup, tmp_path):
    from typer.testing import CliRunner

    from locassess.cli import app

    conn, cfg, criteria, llm, search, fetcher = setup
    db = tmp_path / "cli.db"
    c2 = repo.connect(db)
    repo.init_db(c2)
    cid = build_corpus(c2, cfg, criteria, llm, search, fetcher, run_date=RUN)
    c2.close()
    runner = CliRunner()
    r = runner.invoke(app, ["corpus", "freeze", "--corpus", str(cid), "--config", str(EXAMPLE_CONFIG), "--db", str(db)])
    assert r.exit_code == 0 and "frozen with 2 articles" in r.output, r.output
    r = runner.invoke(app, ["corpus", "log", "--corpus", str(cid), "--out", str(tmp_path / "res"), "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "res" / f"selection_log_corpus{cid}.csv").exists()


def test_selection_log_unfinished_run(conn):
    cid = repo.create_corpus(conn, "{}", "2021-10-13", "2026-10-13", "h")
    for i in range(3):
        repo.insert_article(conn, cid, url=f"https://lvz.de/{i}", domain="lvz.de")
    summary, _ = selection_log(conn, cid)
    counts = {r["step"]: r for r in summary}
    assert counts["fetched"]["count"] == 0 and counts["passed_filter"]["count"] == 0
    assert "3 not processed yet" in counts["whitelisted"]["detail"]


def test_process_resumes_interrupted_build(setup):
    """Searches done, nothing fetched (as after Ctrl+C) → process finishes the corpus like a full build."""
    from locassess.stage0_corpus import process_corpus

    conn, cfg, criteria, llm, search, fetcher = setup

    class Interrupt(Exception):
        pass

    def broken_fetch(url):
        raise Interrupt()

    import locassess.stage0_corpus as s0
    real = s0._process_articles
    s0._process_articles = lambda *a, **k: None  # stop right after the searches
    try:
        cid = build_corpus(conn, cfg, criteria, llm, search, fetcher, run_date=RUN)
    finally:
        s0._process_articles = real
    assert all(a["fetched_at"] is None for a in repo.get_articles(conn, cid))

    process_corpus(conn, cid, cfg, criteria, llm, fetcher)
    first = {a["url"]: (a["selected"], a["exclusion_reason"]) for a in repo.get_articles(conn, cid)}
    assert {u.split("/")[-1] for u, (sel, _) in first.items() if sel} == {"a2", "a10"}

    # a second run fetches nothing new, rates nothing new, and gives the same result
    n_fetch, n_rel = len(fetcher.calls), sum(1 for p, _ in llm.calls if p == "relevance")
    process_corpus(conn, cid, cfg, criteria, llm, fetcher)
    assert len(fetcher.calls) == n_fetch
    assert sum(1 for p, _ in llm.calls if p == "relevance") == n_rel
    assert {a["url"]: (a["selected"], a["exclusion_reason"]) for a in repo.get_articles(conn, cid)} == first


def test_process_refuses_frozen_corpus(setup):
    from locassess.stage0_corpus import process_corpus

    conn, cfg, criteria, llm, search, fetcher = setup
    cid = build_corpus(conn, cfg, criteria, llm, search, fetcher, run_date=RUN)
    freeze(conn, cid, 255)
    with pytest.raises(repo.CorpusFrozenError):
        process_corpus(conn, cid, cfg, criteria, llm, fetcher)


def test_cli_build_rejects_unknown_criteria(tmp_path):
    from typer.testing import CliRunner

    from locassess.cli import app

    r = CliRunner().invoke(app, ["corpus", "build", "--criteria", "H02,H99", "--config", str(EXAMPLE_CONFIG),
                                 "--db", str(tmp_path / "x.db")])
    assert r.exit_code != 0


def test_queries_run_round_robin_over_criteria(setup):
    conn, cfg, criteria, llm, search, fetcher = setup
    build_corpus(conn, cfg, criteria, llm, search, fetcher, run_date=RUN)
    order = [c["query"].split()[1:3] for c in search.calls]
    assert order == [["H01", "en"], ["H02", "en"], ["H03", "en"], ["H01", "de"], ["H02", "de"], ["H03", "de"]]


def test_max_found_articles_stops_search(setup):
    conn, cfg, criteria, llm, search, fetcher = setup
    cfg.corpus.max_found_articles = 4
    cid = build_corpus(conn, cfg, criteria, llm, search, fetcher, run_date=RUN)
    # H01 en finds a1, a3, a4 (3); H02 en adds a2 (4) → the remaining new URLs are over the limit
    assert [c["query"].split()[1] for c in search.calls] == ["H01", "H02"]
    assert [v["main_id"] for p, v in llm.calls if p == "query_generation"] == ["H01", "H02"]  # H03 never generated
    arts = repo.get_articles(conn, cid)
    assert len(arts) == 4 and len(fetcher.calls) == 3          # a3 not whitelisted → not downloaded
    over = conn.execute("SELECT url FROM search_hit WHERE status = 'over_limit'").fetchall()
    assert [r["url"] for r in over] == ["https://www.mdr.de/a5", "https://www.mdr.de/a6"]
    summary, _ = selection_log(conn, cid)
    hits = next(r for r in summary if r["step"] == "search_hits")
    assert hits["detail"] == "2 new URLs beyond max_found_articles (not used)"
    assert next(r for r in summary if r["step"] == "unique_urls")["count"] == 4
