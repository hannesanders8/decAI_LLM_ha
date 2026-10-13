"""Stage 1 — evidence extraction (SPEC §8).

The decision model only returns probabilities; fixed rules turn them into evidence:
most probable option per field, confidence = min(P(affected), P(direction), P(intensity)),
review flag below `review_threshold`, a second sentence when it is within
`second_sentence_gap` of the best one, and one evidence per article × main criterion.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

from locassess.config import ExtractionCfg
from locassess.criteria import Codebook, MainCriterion
from locassess.db import repo
from locassess.providers.decision import DecisionModel, DecisionResult, Question
from locassess.query_definition import (
    OVERARCHING, QUERY_DEFINITION_VERSION, query_a_questions, query_b_questions, render_state,
)

CORE_FIELDS = ("direction", "intensity", "evidence_type")

Progress = Callable[[str], None]


def _quiet(_: str) -> None:
    pass


# ---------------------------------------------------------------------------
# 8.4 Rules (pure functions)
# ---------------------------------------------------------------------------

def most_probable(dist: dict[str, float], options: list[str]) -> str:
    """Most probable option; ties go to the option listed first (deterministic)."""
    best = max(dist.get(o, 0.0) for o in options)
    return next(o for o in options if dist.get(o, 0.0) == best)


@dataclass(frozen=True)
class EvidenceCandidate:
    main_id: str
    section: int
    sub_id: str | None
    direction: int
    intensity: int
    evidence_type: str
    confidence: float
    sentence_keys: tuple[str, ...]
    needs_review: bool


def derive_evidence(main_id: str, section: int, p_affected: float, b: DecisionResult,
                    questions: list[Question], sentence_keys: list[str],
                    params: ExtractionCfg) -> EvidenceCandidate | None:
    """Apply the rules to one Query B answer. None if a core field was refused."""
    options = {q.key: q.values for q in questions}
    if any(f in b.refusals or f not in b.distributions for f in CORE_FIELDS):
        return None
    chosen = {k: most_probable(b.distributions[k], options[k]) for k in options if k in b.distributions}

    p_dir = b.distributions["direction"][chosen["direction"]]
    p_int = b.distributions["intensity"][chosen["intensity"]]
    confidence = min(p_affected, p_dir, p_int)

    sub = chosen.get("sub_criterion")
    sub_id = None if sub in (None, OVERARCHING) else sub

    if "evidence_sentence" in b.distributions:
        dist = b.distributions["evidence_sentence"]
        order = options["evidence_sentence"]
        ranked = sorted(order, key=lambda k: (-dist.get(k, 0.0), order.index(k)))
        keys = [ranked[0]]
        if len(ranked) > 1 and dist.get(ranked[1], 0.0) >= dist.get(ranked[0], 0.0) - params.second_sentence_gap:
            keys.append(ranked[1])
    else:
        keys = list(sentence_keys[:1])   # single-sentence section, or the question was refused

    return EvidenceCandidate(
        main_id=main_id, section=section, sub_id=sub_id, direction=int(chosen["direction"]),
        intensity=int(chosen["intensity"]), evidence_type=chosen["evidence_type"], confidence=confidence,
        sentence_keys=tuple(keys), needs_review=confidence < params.review_threshold,
    )


def merge_sections(candidates: list[EvidenceCandidate]) -> dict[str, EvidenceCandidate]:
    """One evidence per main criterion: the higher-confidence section wins (ties: earlier section)."""
    out: dict[str, EvidenceCandidate] = {}
    for c in sorted(candidates, key=lambda c: (c.main_id, -c.confidence, c.section)):
        out.setdefault(c.main_id, c)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def codebook_version(codebook: Codebook, path: str | Path) -> str:
    """Codebook version plus a short content hash, so edits without a version bump stay traceable."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:8]
    return f"{codebook.version}+{digest}"


def run_extraction(conn: sqlite3.Connection, corpus_id: int, model: DecisionModel, codebook: Codebook,
                   codebook_path: str | Path, criteria: list[MainCriterion], params: ExtractionCfg,
                   repetition: int | None = None, progress: Progress = _quiet) -> int:
    """One extraction run = one decision model × one repetition on a frozen corpus."""
    corpus = repo.get_corpus(conn, corpus_id)
    if not corpus["frozen"]:
        raise ValueError(f"corpus {corpus_id} is not frozen; run `locassess corpus freeze` first")
    location = json.loads(corpus["location_json"])
    loc_name = location.get("name") or "the location"
    articles = repo.get_articles(conn, corpus_id, selected_only=True)
    repetition = repetition or repo.next_repetition(conn, corpus_id, model.name)
    run_id = repo.create_extraction_run(conn, corpus_id, model.name, QUERY_DEFINITION_VERSION,
                                        codebook_version(codebook, codebook_path), repetition,
                                        model.confidence_method)
    qa = query_a_questions(codebook, criteria, loc_name)
    by_id = {m.main_id: m for m in criteria}
    model_version = None

    for n, art in enumerate(articles, start=1):
        sentences = repo.get_sentences(conn, art["article_id"])
        sections = [(s, list(rows)) for s, rows in groupby(sentences, key=lambda r: r["section"])]
        texts = {r["sentence_key"]: r["text"] for r in sentences}
        candidates: list[EvidenceCandidate] = []
        for section, rows in sections:
            keys = [r["sentence_key"] for r in rows]
            state = render_state(location, art["title"], art["published_at"],
                                 [(r["sentence_key"], r["text"]) for r in rows], section, len(sections))
            progress(f"[{n}/{len(articles)}] article {art['article_id']} section {section}: query A")
            a = model.decide(state, qa, purpose="extraction_A", version=QUERY_DEFINITION_VERSION)
            model_version = a.model_version or model_version
            repo.insert_model_response(conn, run_id, art["article_id"], section, "A", None, a.to_json(),
                                       a.call_ids[0] if a.call_ids else None)
            for m in criteria:
                p_yes = a.distributions.get(f"{m.main_id}_affected", {}).get("yes")
                if p_yes is None or p_yes < params.affected_threshold:
                    continue
                qb = query_b_questions(codebook, by_id[m.main_id], keys, loc_name)
                progress(f"    {m.main_id} {m.name}: query B (P(affected)={p_yes:.2f})")
                b = model.decide(state, qb, purpose="extraction_B", version=QUERY_DEFINITION_VERSION)
                repo.insert_model_response(conn, run_id, art["article_id"], section, "B", m.main_id, b.to_json(),
                                           b.call_ids[0] if b.call_ids else None)
                cand = derive_evidence(m.main_id, section, p_yes, b, qb, keys, params)
                if cand is not None:
                    candidates.append(cand)

        event_date = (art["published_at"] or "")[:7]
        for c in merge_sections(candidates).values():
            repo.insert_evidence(
                conn, run_id, art["article_id"], c.main_id, c.sub_id, c.direction, c.intensity, c.evidence_type,
                event_date, c.confidence, list(c.sentence_keys), " ".join(texts[k] for k in c.sentence_keys),
                c.needs_review,
            )

    repo.set_extraction_run_model_version(conn, run_id, model_version)
    return run_id


def run_summary(conn: sqlite3.Connection, extraction_run_id: int) -> dict:
    run = repo.get_extraction_run(conn, extraction_run_id)
    q = lambda sql: conn.execute(sql, (extraction_run_id,)).fetchone()[0]  # noqa: E731
    return {
        "extraction_run_id": extraction_run_id, "model": run["model"], "model_version": run["model_version"],
        "repetition": run["repetition"], "confidence_method": run["confidence_method"],
        "responses_A": q("SELECT COUNT(*) FROM model_response WHERE extraction_run_id=? AND query_type='A'"),
        "responses_B": q("SELECT COUNT(*) FROM model_response WHERE extraction_run_id=? AND query_type='B'"),
        "evidence": q("SELECT COUNT(*) FROM evidence WHERE extraction_run_id=?"),
        "needs_review": q("SELECT COUNT(*) FROM evidence WHERE extraction_run_id=? AND needs_review=1"),
    }
