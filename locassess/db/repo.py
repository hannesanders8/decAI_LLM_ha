"""Thin data-access layer over sqlite3."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

from locassess.criteria import MainCriterion

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
