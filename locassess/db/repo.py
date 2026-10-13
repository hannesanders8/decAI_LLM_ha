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


def connect(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
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
