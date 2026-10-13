"""Baseline for SQ3 (SPEC §11): direct LLM assessment of the frozen corpus.

The LLM receives the same frozen corpus (full texts) and the criteria list and returns a
score (-3 … +3), rating and justification per main criterion plus an overall judgement —
no rules in between. To make traceability comparable with the hybrid approach, it also
cites verbatim quotes with article labels; the evaluation checks them against the texts.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable

from locassess.criteria import MainCriterion
from locassess.db import repo
from locassess.providers.llm import LLM

Progress = Callable[[str], None]


def article_labels(conn: sqlite3.Connection, corpus_id: int) -> dict[str, sqlite3.Row]:
    """A1, A2, … → selected article rows (stable order by article_id)."""
    return {f"A{i}": a for i, a in enumerate(repo.get_articles(conn, corpus_id, selected_only=True), start=1)}


def render_articles(labels: dict[str, sqlite3.Row]) -> str:
    blocks = []
    for label, a in labels.items():
        blocks.append(f"=== {label} ===\nTitle: {a['title'] or ''}\nOutlet: {a['outlet'] or ''}\n"
                      f"Published: {a['published_at'] or ''}\n\n{a['text']}")
    return "\n\n".join(blocks)


def baseline_schema(criteria: list[MainCriterion], labels: list[str]) -> dict:
    item = {
        "type": "object",
        "properties": {
            "score": {"type": ["number", "null"]},
            "rating": {"type": ["string", "null"], "enum": ["green", "yellow", "red", None]},
            "justification": {"type": "string"},
            "evidence": {"type": "array", "items": {
                "type": "object",
                "properties": {"article": {"type": "string", "enum": labels}, "quote": {"type": "string"}},
                "required": ["article", "quote"], "additionalProperties": False}},
        },
        "required": ["score", "rating", "justification", "evidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "criteria": {"type": "object", "properties": {m.main_id: item for m in criteria},
                         "required": [m.main_id for m in criteria], "additionalProperties": False},
            "overall": {"type": "object",
                        "properties": {"score": {"type": ["number", "null"]}, "judgement": {"type": "string"}},
                        "required": ["score", "judgement"], "additionalProperties": False},
        },
        "required": ["criteria", "overall"],
        "additionalProperties": False,
    }


def _clean(entry: dict) -> dict:
    score = entry.get("score")
    out = dict(entry)
    if score is not None:
        out["score_raw"] = score
        out["score"] = max(-3.0, min(3.0, float(score)))    # rule: clamp to the scale
    return out


def run_baseline(conn: sqlite3.Connection, llm: LLM, corpus_id: int, criteria: list[MainCriterion], model: str,
                 temperature: float = 0, repetition: int | None = None, progress: Progress = lambda m: None) -> int:
    corpus = repo.get_corpus(conn, corpus_id)
    if not corpus["frozen"]:
        raise ValueError(f"corpus {corpus_id} is not frozen; the baseline uses the same frozen corpus as Stage 1")
    labels = article_labels(conn, corpus_id)
    loc = json.loads(corpus["location_json"])
    if repetition is None:
        row = conn.execute("SELECT MAX(repetition) FROM baseline_assessment WHERE corpus_id = ? AND model = ?",
                           (corpus_id, model)).fetchone()
        repetition = (row[0] or 0) + 1
    progress(f"baseline corpus {corpus_id}, {len(labels)} articles, repetition {repetition}")
    criteria_list = "\n".join(
        f"- {m.main_id} {m.name} ({m.factor})" + (f": {', '.join(s.name for s in m.sub_criteria)}" if m.sub_criteria else "")
        for m in criteria)
    result = llm.complete("baseline", {
        "location": ", ".join(p for p in (loc.get("name"), loc.get("region"), loc.get("country")) if p),
        "period": f"{corpus['period_from']} to {corpus['period_to']}",
        "criteria_list": criteria_list, "articles": render_articles(labels),
    }, json_schema=baseline_schema(criteria, list(labels)), temperature=temperature)
    data = result.parsed[0]
    per_criterion = {k: _clean(v) for k, v in data["criteria"].items()}
    overall = _clean(data["overall"])
    overall["article_labels"] = {k: a["article_id"] for k, a in labels.items()}
    overall["prompt_version"] = result.prompt_version
    with conn:
        cur = conn.execute(
            "INSERT INTO baseline_assessment (corpus_id, model, repetition, per_criterion_json, overall_json, call_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (corpus_id, model, repetition, json.dumps(per_criterion, ensure_ascii=False),
             json.dumps(overall, ensure_ascii=False), result.call_ids[0] if result.call_ids else None),
        )
    return cur.lastrowid
