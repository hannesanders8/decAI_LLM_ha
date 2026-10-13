"""Manual gold standard (table `annotation`): CSV template export and validated import.

The codebook (data/codebook.yaml) is the annotation guideline. The template has one row per
selected article × main criterion, so "not affected" is coded explicitly (affected = 0).
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from locassess.criteria import MainCriterion
from locassess.db import repo

TEMPLATE_FIELDS = ["article_id", "main_id", "annotator", "affected", "direction", "intensity", "evidence_type",
                   "sub_id", "sentence_keys", "criterion_name", "article_title", "article_url"]


class AnnotationError(ValueError):
    pass


def write_template(conn: sqlite3.Connection, corpus_id: int, criteria: list[MainCriterion], annotator: str,
                   path: str | Path) -> int:
    """CSV with one row per selected article × main criterion; existing annotations are pre-filled."""
    if not repo.is_frozen(conn, corpus_id):
        raise AnnotationError(f"corpus {corpus_id} is not frozen; annotate the frozen corpus")
    existing = {(r["article_id"], r["main_id"]): r for r in conn.execute(
        "SELECT * FROM annotation WHERE annotator = ?", (annotator,))}
    n = 0
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TEMPLATE_FIELDS)
        w.writeheader()
        for a in repo.get_articles(conn, corpus_id, selected_only=True):
            for m in criteria:
                e = existing.get((a["article_id"], m.main_id))
                w.writerow({
                    "article_id": a["article_id"], "main_id": m.main_id, "annotator": annotator,
                    "affected": "" if e is None else e["affected"],
                    "direction": "" if e is None or e["direction"] is None else e["direction"],
                    "intensity": "" if e is None or e["intensity"] is None else e["intensity"],
                    "evidence_type": "" if e is None else (e["evidence_type"] or ""),
                    "sub_id": "" if e is None else (e["sub_id"] or ""),
                    "sentence_keys": "" if e is None else (e["sentence_keys"] or ""),
                    "criterion_name": m.name, "article_title": a["title"], "article_url": a["url"],
                })
                n += 1
    return n


def import_annotations(conn: sqlite3.Connection, path: str | Path, criteria: list[MainCriterion]) -> int:
    """Validate and upsert a filled template. Rows with an empty `affected` are skipped."""
    subs = {m.main_id: {s.sub_id for s in m.sub_criteria} for m in criteria}
    rows, errors = [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for line, r in enumerate(csv.DictReader(f), start=2):
            if not (r.get("affected") or "").strip():
                continue
            try:
                rows.append(_parse_row(conn, r, subs))
            except AnnotationError as exc:
                errors.append(f"line {line}: {exc}")
    if errors:
        raise AnnotationError("invalid annotations:\n" + "\n".join(errors[:50]))
    with conn:
        conn.executemany(
            "INSERT INTO annotation (article_id, main_id, annotator, affected, direction, intensity, evidence_type, "
            "sub_id, sentence_keys) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(article_id, main_id, annotator) DO UPDATE SET affected=excluded.affected, "
            "direction=excluded.direction, intensity=excluded.intensity, evidence_type=excluded.evidence_type, "
            "sub_id=excluded.sub_id, sentence_keys=excluded.sentence_keys", rows)
    return len(rows)


def _parse_row(conn: sqlite3.Connection, r: dict, subs: dict[str, set[str]]) -> tuple:
    def val(k):
        return (r.get(k) or "").strip()

    try:
        article_id = int(val("article_id"))
    except ValueError:
        raise AnnotationError("article_id must be an integer") from None
    main_id, annotator = val("main_id"), val("annotator")
    if main_id not in subs:
        raise AnnotationError(f"unknown main_id {main_id!r}")
    if not annotator:
        raise AnnotationError("annotator is empty")
    if val("affected") not in ("0", "1"):
        raise AnnotationError("affected must be 0 or 1")
    affected = int(val("affected"))
    if not affected:
        return (article_id, main_id, annotator, 0, None, None, None, None, None)

    if val("direction") not in ("-1", "0", "1"):
        raise AnnotationError("direction must be -1, 0 or 1 when affected = 1")
    if val("intensity") not in ("1", "2", "3"):
        raise AnnotationError("intensity must be 1, 2 or 3 when affected = 1")
    if val("evidence_type") not in ("fact", "announcement", "opinion"):
        raise AnnotationError("evidence_type must be fact, announcement or opinion when affected = 1")
    sub_id = val("sub_id") or None
    if sub_id and sub_id != "overarching" and sub_id not in subs[main_id]:
        raise AnnotationError(f"sub_id {sub_id!r} is not a sub-criterion of {main_id}")
    keys = [k.strip() for k in val("sentence_keys").replace(";", ",").split(",") if k.strip()]
    known = {s["sentence_key"] for s in repo.get_sentences(conn, article_id)}
    missing = [k for k in keys if k not in known]
    if missing:
        raise AnnotationError(f"sentence keys {missing} do not exist in article {article_id}")
    return (article_id, main_id, annotator, 1, int(val("direction")), int(val("intensity")), val("evidence_type"),
            None if sub_id in (None, "overarching") else sub_id, ",".join(keys) or None)
