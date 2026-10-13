"""Thin data-access layer over sqlite3."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

from locassess.criteria import MainCriterion
from locassess.stage2_score import Evidence, Profile

DEFAULT_DB_PATH = Path("data/locassess.db")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str | Path = DEFAULT_DB_PATH, check_same_thread: bool = True) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables and triggers (idempotent)."""
    schema = resources.files("locassess.db").joinpath("schema.sql").read_text(encoding="utf-8")
    conn.executescript(schema)
    conn.commit()


def upsert_criteria(conn: sqlite3.Connection, criteria: list[MainCriterion]) -> None:
    """Insert or update main and sub-criteria (idempotent)."""
    with conn:
        for m in criteria:
            conn.execute(
                "INSERT INTO criterion (main_id, factor, name, name_de) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(main_id) DO UPDATE SET factor=excluded.factor, name=excluded.name, "
                "name_de=excluded.name_de",
                (m.main_id, m.factor, m.name, m.name_de),
            )
            for s in m.sub_criteria:
                conn.execute(
                    "INSERT INTO sub_criterion (sub_id, main_id, name, name_de) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(sub_id) DO UPDATE SET main_id=excluded.main_id, name=excluded.name, "
                    "name_de=excluded.name_de",
                    (s.sub_id, s.main_id, s.name, s.name_de),
                )


def count_criteria(conn: sqlite3.Connection) -> tuple[int, int]:
    """(number of main criteria, number of sub-criteria) in the database."""
    n_main = conn.execute("SELECT COUNT(*) FROM criterion").fetchone()[0]
    n_sub = conn.execute("SELECT COUNT(*) FROM sub_criterion").fetchone()[0]
    return n_main, n_sub


@dataclass
class CallRecord:
    provider: str
    model: str | None
    purpose: str
    request_json: str
    response_json: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    error: str | None = None
    created_at: str | None = None


def insert_call_log(conn: sqlite3.Connection, rec: CallRecord) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO call_log (provider, model, model_version, purpose, prompt_version, "
            "request_json, response_json, input_tokens, output_tokens, cost_usd, latency_ms, "
            "created_at, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rec.provider, rec.model, rec.model_version, rec.purpose, rec.prompt_version,
                rec.request_json, rec.response_json, rec.input_tokens, rec.output_tokens,
                rec.cost_usd, rec.latency_ms, rec.created_at or utc_now(), rec.error,
            ),
        )
    return cur.lastrowid


def criteria_factors(conn: sqlite3.Connection) -> dict[str, str]:
    """main_id → factor for all loaded main criteria."""
    return {r["main_id"]: r["factor"] for r in conn.execute("SELECT main_id, factor FROM criterion")}


def extraction_run_period(conn: sqlite3.Connection, extraction_run_id: int) -> tuple[str | None, str | None]:
    """(period_from, period_to) of the corpus behind an extraction run."""
    row = conn.execute(
        "SELECT c.period_from, c.period_to FROM extraction_run r JOIN corpus c USING (corpus_id) "
        "WHERE r.extraction_run_id = ?", (extraction_run_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"extraction run {extraction_run_id} not found")
    return row["period_from"], row["period_to"]


def load_evidence(conn: sqlite3.Connection, extraction_run_id: int) -> list[Evidence]:
    rows = conn.execute(
        "SELECT e.evidence_id, e.main_id, e.article_id, e.direction, e.intensity, e.evidence_type, "
        "e.confidence, e.event_date, a.outlet FROM evidence e LEFT JOIN article a USING (article_id) "
        "WHERE e.extraction_run_id = ? ORDER BY e.evidence_id", (extraction_run_id,),
    ).fetchall()
    return [Evidence(**dict(r)) for r in rows]


def save_scoring_run(conn: sqlite3.Connection, extraction_run_id: int, profile: Profile,
                     label: str | None = None, variant: str = "default") -> int:
    """Store a profile as one scoring run with its criterion scores and overall score."""
    with conn:
        cur = conn.execute(
            "INSERT INTO scoring_run (extraction_run_id, params_json, label, created_at) VALUES (?, ?, ?, ?)",
            (extraction_run_id, json.dumps(profile.params.to_dict(), sort_keys=True), label, utc_now()),
        )
        scoring_run_id = cur.lastrowid
        for c in profile.criteria.values():
            conn.execute(
                "INSERT INTO criterion_score (scoring_run_id, main_id, score, rating, status, trend, "
                "red_flags_json, evidence_ids_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (scoring_run_id, c.main_id, c.score, c.rating, c.status, c.trend,
                 json.dumps([asdict(f) for f in c.red_flags]), json.dumps(list(c.evidence_ids))),
            )
        o = profile.overall
        conn.execute(
            "INSERT INTO overall_score (scoring_run_id, variant, sliders_json, weights_json, score, "
            "factor_scores_json) VALUES (?, ?, ?, ?, ?, ?)",
            (scoring_run_id, variant, json.dumps(o.sliders, sort_keys=True),
             json.dumps(o.weights, sort_keys=True), o.score, json.dumps(o.factor_scores, sort_keys=True)),
        )
    return scoring_run_id


# ---------------------------------------------------------------------------
# Stage 0: corpus, queries, hits, articles, sentences
# ---------------------------------------------------------------------------

class CorpusFrozenError(RuntimeError):
    """Raised on any attempt to change articles or sentences of a frozen corpus."""


ARTICLE_FIELDS = (
    "url", "domain", "outlet", "title", "published_at", "fetched_at", "language", "text",
    "text_sha256", "word_count", "relevance", "relevance_reason", "selected", "exclusion_reason",
)


def create_corpus(conn: sqlite3.Connection, location_json: str, period_from: str, period_to: str,
                  config_hash: str, note: str | None = None) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO corpus (location_json, period_from, period_to, created_at, config_hash, note) "
            "VALUES (?, ?, ?, ?, ?, ?)", (location_json, period_from, period_to, utc_now(), config_hash, note),
        )
    return cur.lastrowid


def get_corpus(conn: sqlite3.Connection, corpus_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM corpus WHERE corpus_id = ?", (corpus_id,)).fetchone()
    if row is None:
        raise LookupError(f"corpus {corpus_id} not found")
    return row


def is_frozen(conn: sqlite3.Connection, corpus_id: int) -> bool:
    return bool(get_corpus(conn, corpus_id)["frozen"])


def _require_unfrozen(conn: sqlite3.Connection, corpus_id: int) -> None:
    if is_frozen(conn, corpus_id):
        raise CorpusFrozenError(f"corpus {corpus_id} is frozen and read-only")


def insert_search_query(conn: sqlite3.Connection, corpus_id: int, main_id: str | None, query_text: str,
                        language: str | None, prompt_version: str | None) -> int:
    _require_unfrozen(conn, corpus_id)
    with conn:
        cur = conn.execute(
            "INSERT INTO search_query (corpus_id, main_id, query_text, language, prompt_version) "
            "VALUES (?, ?, ?, ?, ?)", (corpus_id, main_id, query_text, language, prompt_version),
        )
    return cur.lastrowid


def set_query_hit_count(conn: sqlite3.Connection, query_id: int, hit_count: int | None) -> None:
    with conn:
        conn.execute("UPDATE search_query SET hit_count = ? WHERE query_id = ?", (hit_count, query_id))


def insert_search_hit(conn: sqlite3.Connection, query_id: int, url: str, title: str | None, rank: int,
                      status: str, exclusion_reason: str | None = None) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO search_hit (query_id, url, title, rank, status, exclusion_reason) "
            "VALUES (?, ?, ?, ?, ?, ?)", (query_id, url, title, rank, status, exclusion_reason),
        )
    return cur.lastrowid


def insert_article(conn: sqlite3.Connection, corpus_id: int, **fields) -> int:
    _require_unfrozen(conn, corpus_id)
    unknown = set(fields) - set(ARTICLE_FIELDS)
    if unknown:
        raise ValueError(f"unknown article fields: {sorted(unknown)}")
    cols = ["corpus_id", *fields]
    with conn:
        cur = conn.execute(
            f"INSERT INTO article ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            (corpus_id, *fields.values()),
        )
    return cur.lastrowid


def article_corpus_id(conn: sqlite3.Connection, article_id: int) -> int:
    row = conn.execute("SELECT corpus_id FROM article WHERE article_id = ?", (article_id,)).fetchone()
    if row is None:
        raise LookupError(f"article {article_id} not found")
    return row[0]


def update_article(conn: sqlite3.Connection, article_id: int, **fields) -> None:
    _require_unfrozen(conn, article_corpus_id(conn, article_id))
    unknown = set(fields) - set(ARTICLE_FIELDS)
    if unknown:
        raise ValueError(f"unknown article fields: {sorted(unknown)}")
    if not fields:
        return
    with conn:
        conn.execute(
            f"UPDATE article SET {', '.join(f'{k} = ?' for k in fields)} WHERE article_id = ?",
            (*fields.values(), article_id),
        )


def get_articles(conn: sqlite3.Connection, corpus_id: int, selected_only: bool = False) -> list[sqlite3.Row]:
    sql = "SELECT * FROM article WHERE corpus_id = ?"
    if selected_only:
        sql += " AND selected = 1"
    return conn.execute(sql + " ORDER BY article_id", (corpus_id,)).fetchall()


def hit_criteria(conn: sqlite3.Connection, corpus_id: int) -> list[tuple[str, str | None]]:
    """(hit URL, main_id of the query that found it) for every search hit of a corpus."""
    return [(r["url"], r["main_id"]) for r in conn.execute(
        "SELECT h.url, q.main_id FROM search_hit h JOIN search_query q USING (query_id) "
        "WHERE q.corpus_id = ? ORDER BY h.hit_id", (corpus_id,),
    )]


def replace_sentences(conn: sqlite3.Connection, article_id: int,
                      rows: list[tuple[str, int, str]]) -> None:
    """Replace the sentences of an article. rows: (sentence_key, section, text)."""
    _require_unfrozen(conn, article_corpus_id(conn, article_id))
    with conn:
        conn.execute("DELETE FROM sentence WHERE article_id = ?", (article_id,))
        conn.executemany(
            "INSERT INTO sentence (article_id, sentence_key, section, text) VALUES (?, ?, ?, ?)",
            [(article_id, k, s, t) for k, s, t in rows],
        )


def get_sentences(conn: sqlite3.Connection, article_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM sentence WHERE article_id = ? ORDER BY CAST(SUBSTR(sentence_key, 2) AS INTEGER)",
        (article_id,),
    ).fetchall()


def freeze_corpus(conn: sqlite3.Connection, corpus_id: int) -> None:
    _require_unfrozen(conn, corpus_id)
    with conn:
        conn.execute("UPDATE corpus SET frozen = 1 WHERE corpus_id = ?", (corpus_id,))


# ---------------------------------------------------------------------------
# Stage 1: extraction runs, model responses, evidence
# ---------------------------------------------------------------------------

def next_repetition(conn: sqlite3.Connection, corpus_id: int, model: str) -> int:
    row = conn.execute("SELECT MAX(repetition) FROM extraction_run WHERE corpus_id = ? AND model = ?",
                       (corpus_id, model)).fetchone()
    return (row[0] or 0) + 1


def create_extraction_run(conn: sqlite3.Connection, corpus_id: int, model: str, query_definition_version: str,
                          codebook_version: str, repetition: int, confidence_method: str) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO extraction_run (corpus_id, model, query_definition_version, codebook_version, "
            "repetition, confidence_method, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (corpus_id, model, query_definition_version, codebook_version, repetition, confidence_method, utc_now()),
        )
    return cur.lastrowid


def set_extraction_run_model_version(conn: sqlite3.Connection, extraction_run_id: int, model_version: str | None) -> None:
    with conn:
        conn.execute("UPDATE extraction_run SET model_version = ? WHERE extraction_run_id = ?",
                     (model_version, extraction_run_id))


def get_extraction_run(conn: sqlite3.Connection, extraction_run_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM extraction_run WHERE extraction_run_id = ?", (extraction_run_id,)).fetchone()
    if row is None:
        raise LookupError(f"extraction run {extraction_run_id} not found")
    return row


def insert_model_response(conn: sqlite3.Connection, extraction_run_id: int, article_id: int, section: int,
                          query_type: str, main_id: str | None, distributions_json: str,
                          call_id: int | None) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO model_response (extraction_run_id, article_id, section, query_type, main_id, "
            "distributions_json, call_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (extraction_run_id, article_id, section, query_type, main_id, distributions_json, call_id),
        )
    return cur.lastrowid


def insert_evidence(conn: sqlite3.Connection, extraction_run_id: int, article_id: int, main_id: str,
                    sub_id: str | None, direction: int, intensity: int, evidence_type: str, event_date: str,
                    confidence: float, sentence_keys: list[str], sentence_text: str, needs_review: bool) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO evidence (extraction_run_id, article_id, main_id, sub_id, direction, intensity, "
            "evidence_type, event_date, confidence, sentence_keys, sentence_text, needs_review) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (extraction_run_id, article_id, main_id, sub_id, direction, intensity, evidence_type, event_date,
             confidence, json.dumps(sentence_keys), sentence_text, int(needs_review)),
        )
    return cur.lastrowid
