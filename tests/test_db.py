import sqlite3

import pytest

from locassess.db import repo

SPEC_TABLES = {
    "criterion", "sub_criterion", "corpus", "search_query", "search_hit", "article", "sentence",
    "annotation", "extraction_run", "model_response", "evidence", "scoring_run", "criterion_score",
    "overall_score", "report", "baseline_assessment", "call_log",
}


def _tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_all_spec_tables_created(conn):
    assert SPEC_TABLES <= _tables(conn)


def test_init_db_idempotent(conn):
    repo.init_db(conn)
    repo.init_db(conn)
    assert SPEC_TABLES <= _tables(conn)


def _corpus_with_article(conn):
    conn.execute(
        "INSERT INTO corpus (corpus_id, location_json, created_at, config_hash) VALUES (1, '{}', 'now', 'h')"
    )
    conn.execute("INSERT INTO article (article_id, corpus_id, url, text) VALUES (1, 1, 'u', 't')")
    conn.execute("INSERT INTO sentence (article_id, sentence_key, section, text) VALUES (1, 'S1', 0, 'x')")
    conn.commit()


def test_unfrozen_corpus_is_writable(conn):
    _corpus_with_article(conn)
    conn.execute("UPDATE article SET selected = 1 WHERE article_id = 1")
    conn.execute("UPDATE sentence SET text = 'y' WHERE article_id = 1")
    conn.execute("DELETE FROM sentence WHERE article_id = 1")
    conn.commit()


@pytest.mark.parametrize("stmt", [
    "INSERT INTO article (corpus_id, url) VALUES (1, 'u2')",
    "UPDATE article SET title = 'changed' WHERE article_id = 1",
    "DELETE FROM article WHERE article_id = 1",
    "INSERT INTO sentence (article_id, sentence_key, section, text) VALUES (1, 'S2', 0, 'z')",
    "UPDATE sentence SET text = 'changed' WHERE article_id = 1",
    "DELETE FROM sentence WHERE article_id = 1",
])
def test_frozen_corpus_is_read_only(conn, stmt):
    _corpus_with_article(conn)
    conn.execute("UPDATE corpus SET frozen = 1 WHERE corpus_id = 1")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError, match="frozen"):
        conn.execute(stmt)
