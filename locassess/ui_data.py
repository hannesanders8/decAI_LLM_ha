"""Data functions behind the Streamlit UI (no Streamlit code here, so everything is testable).

Overall scores in the UI are recomputed live from the stored criterion scores with the
slider weights, using the Stage 2 formula (SPEC §9.7) — no model calls.
"""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path

import pandas as pd

from locassess.stage2_score import OverallResult, overall_score

RATING_COLOURS = {"green": "#2e7d32", "yellow": "#f9a825", "red": "#c62828"}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def scoring_runs(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per scoring run with location, corpus, model and report status."""
    rows = conn.execute("""
        SELECT s.scoring_run_id, s.label, s.created_at, e.extraction_run_id, e.model, e.repetition,
               c.corpus_id, c.location_json, c.period_from, c.period_to,
               (SELECT MAX(passed) FROM report r WHERE r.scoring_run_id = s.scoring_run_id) AS report_passed
        FROM scoring_run s JOIN extraction_run e USING (extraction_run_id) JOIN corpus c USING (corpus_id)
        ORDER BY s.scoring_run_id DESC""").fetchall()
    out = []
    for r in rows:
        loc = json.loads(r["location_json"])
        out.append({
            "scoring_run_id": r["scoring_run_id"], "location": loc.get("name"), "country": loc.get("country"),
            "corpus_id": r["corpus_id"], "extraction_run_id": r["extraction_run_id"], "model": r["model"],
            "repetition": r["repetition"], "label": r["label"], "period": f"{r['period_from']} – {r['period_to']}",
            "report": {1: "passed", 0: "failed"}.get(r["report_passed"], "none"), "created_at": r["created_at"],
        })
    return pd.DataFrame(out)


def run_title(row) -> str:
    label = f" · {row['label']}" if row.get("label") else ""
    return (f"#{row['scoring_run_id']} {row['location']} — {row['model']} rep {row['repetition']} "
            f"(corpus {row['corpus_id']}){label}")


def corpora(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = conn.execute("""
        SELECT c.corpus_id, c.location_json, c.period_from, c.period_to, c.frozen, c.note, c.created_at,
               (SELECT COUNT(*) FROM article a WHERE a.corpus_id = c.corpus_id AND a.selected = 1) AS selected
        FROM corpus c ORDER BY c.corpus_id DESC""").fetchall()
    return pd.DataFrame([{
        "corpus_id": r["corpus_id"], "location": json.loads(r["location_json"]).get("name"),
        "period": f"{r['period_from']} – {r['period_to']}", "selected": r["selected"],
        "frozen": bool(r["frozen"]), "note": r["note"], "created_at": r["created_at"]} for r in rows])


def extraction_runs(conn: sqlite3.Connection, corpus_id: int | None = None) -> pd.DataFrame:
    sql = """SELECT e.*, (SELECT COUNT(*) FROM evidence v WHERE v.extraction_run_id = e.extraction_run_id) AS evidence
             FROM extraction_run e"""
    args: tuple = ()
    if corpus_id is not None:
        sql += " WHERE e.corpus_id = ?"
        args = (corpus_id,)
    return pd.DataFrame([dict(r) for r in conn.execute(sql + " ORDER BY e.extraction_run_id DESC", args)])


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

def criteria_table(conn: sqlite3.Connection, scoring_run_id: int) -> pd.DataFrame:
    """All 26 main criteria of a scoring run, in framework order."""
    rows = conn.execute("""
        SELECT c.main_id, c.factor, c.name, s.score, s.rating, s.status, s.trend,
               s.evidence_ids_json, s.red_flags_json
        FROM criterion c LEFT JOIN criterion_score s ON s.main_id = c.main_id AND s.scoring_run_id = ?
        ORDER BY c.main_id""", (scoring_run_id,)).fetchall()
    return pd.DataFrame([{
        "main_id": r["main_id"], "factor": r["factor"], "criterion": r["name"], "score": r["score"],
        "rating": r["rating"], "status": r["status"], "trend": r["trend"],
        "evidence": len(json.loads(r["evidence_ids_json"] or "[]")),
        "red_flags": ", ".join(sorted({f["rule"] for f in json.loads(r["red_flags_json"] or "[]")})),
    } for r in rows])


def red_flags(conn: sqlite3.Connection, scoring_run_id: int) -> list[dict]:
    out = []
    for r in conn.execute("""SELECT s.main_id, c.name, s.red_flags_json FROM criterion_score s
                             JOIN criterion c USING (main_id) WHERE s.scoring_run_id = ? ORDER BY s.main_id""",
                          (scoring_run_id,)):
        for f in json.loads(r["red_flags_json"] or "[]"):
            ev = evidence_table(conn, scoring_run_id, ids=f["evidence_ids"])
            out.append({"rule": f["rule"], "main_id": r["main_id"], "criterion": r["name"],
                        "description": f["description"], "evidence": ev})
    return out


def evidence_table(conn: sqlite3.Connection, scoring_run_id: int, main_id: str | None = None,
                   ids: list[int] | None = None) -> pd.DataFrame:
    """Evidence of a scoring run with sub-criterion, sentence and article (drill-down)."""
    if ids is None:
        ids = []
        sql = "SELECT evidence_ids_json FROM criterion_score WHERE scoring_run_id = ?"
        args: tuple = (scoring_run_id,)
        if main_id:
            sql += " AND main_id = ?"
            args += (main_id,)
        for r in conn.execute(sql, args):
            ids += json.loads(r["evidence_ids_json"] or "[]")
    if not ids:
        return pd.DataFrame(columns=["evidence", "main_id", "sub_criterion", "direction", "intensity", "type",
                                     "confidence", "needs_review", "sentence", "outlet", "date", "title", "url"])
    marks = ",".join("?" * len(ids))
    rows = conn.execute(f"""
        SELECT e.*, sc.name AS sub_name, a.outlet, a.title, a.url, a.published_at
        FROM evidence e LEFT JOIN sub_criterion sc ON sc.sub_id = e.sub_id LEFT JOIN article a USING (article_id)
        WHERE e.evidence_id IN ({marks}) ORDER BY e.main_id, e.sub_id, e.evidence_id""", ids).fetchall()
    return pd.DataFrame([{
        "evidence": f"E{r['evidence_id']}", "main_id": r["main_id"],
        "sub_criterion": f"{r['sub_id']} {r['sub_name']}" if r["sub_id"] else "overarching",
        "direction": r["direction"], "intensity": r["intensity"], "type": r["evidence_type"],
        "confidence": round(r["confidence"], 2), "needs_review": bool(r["needs_review"]),
        "sentence": r["sentence_text"], "outlet": r["outlet"], "date": r["published_at"] or r["event_date"],
        "title": r["title"], "url": r["url"]} for r in rows])


# ---------------------------------------------------------------------------
# Weights, overall score, ranking, comparison (SPEC §9.7, §10.3)
# ---------------------------------------------------------------------------

def overall_with_sliders(table: pd.DataFrame, sliders: dict[str, int]) -> OverallResult:
    scores = {r.main_id: (None if pd.isna(r.score) else float(r.score)) for r in table.itertuples()}
    factors = dict(zip(table["main_id"], table["factor"]))
    return overall_score(scores, {k: int(v) for k, v in sliders.items() if k in scores}, factors=factors)


def ranking(conn: sqlite3.Connection, scoring_run_ids: list[int], sliders: dict[str, int]) -> pd.DataFrame:
    runs = scoring_runs(conn).set_index("scoring_run_id")
    rows = []
    for sid in scoring_run_ids:
        o = overall_with_sliders(criteria_table(conn, sid), sliders)
        rows.append({"scoring_run_id": sid, "run": run_title(runs.loc[sid].to_dict() | {"scoring_run_id": sid}),
                     "location": runs.loc[sid, "location"], "overall": o.score,
                     "criteria_with_evidence": len(o.weights), **{f: s for f, s in o.factor_scores.items()}})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("overall", ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def comparison(conn: sqlite3.Connection, scoring_run_ids: list[int]) -> pd.DataFrame:
    """Criteria × runs: 'score (rating, status)' side by side."""
    runs = scoring_runs(conn).set_index("scoring_run_id")
    base = None
    for sid in scoring_run_ids:
        t = criteria_table(conn, sid)
        col = run_title(runs.loc[sid].to_dict() | {"scoring_run_id": sid})
        t[col] = [("—" if r.status in (None, "no_statement") or pd.isna(r.score) else
                   f"{r.score:+.1f} {r.rating}{' (thin)' if r.status == 'thin' else ''}") for r in t.itertuples()]
        part = t[["main_id", "factor", "criterion", col]]
        base = part if base is None else base.merge(part, on=["main_id", "factor", "criterion"])
    return base if base is not None else pd.DataFrame()


# ---------------------------------------------------------------------------
# Stage 0 selection log, report, exports
# ---------------------------------------------------------------------------

def selection_log_tables(conn: sqlite3.Connection, corpus_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from locassess.stage0_corpus import selection_log

    summary, per_article = selection_log(conn, corpus_id)
    return pd.DataFrame(summary), pd.DataFrame(per_article)


def latest_report(conn: sqlite3.Connection, scoring_run_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM report WHERE scoring_run_id = ? ORDER BY report_id DESC LIMIT 1",
                        (scoring_run_id,)).fetchone()


def profile_markdown(conn: sqlite3.Connection, scoring_run_id: int, sliders: dict[str, int]) -> str:
    """Markdown export: profile table, red flags, overall score, and the report if it passed validation."""
    run = scoring_runs(conn).set_index("scoring_run_id").loc[scoring_run_id]
    table = criteria_table(conn, scoring_run_id)
    o = overall_with_sliders(table, sliders)
    lines = [f"# Location profile: {run['location']}", "",
             f"Scoring run {scoring_run_id} · extraction run {run['extraction_run_id']} ({run['model']}, "
             f"repetition {run['repetition']}) · corpus {run['corpus_id']} · period {run['period']}", "",
             f"**Overall score:** {'n/a' if o.score is None else f'{o.score:+.2f}'} "
             f"(sliders: {', '.join(f'{k}={v}' for k, v in sorted(o.sliders.items()) if v != 1) or 'all 1'})", "",
             "## Criteria", "", "| ID | Factor | Criterion | Score | Rating | Status | Trend | Evidence | Red flags |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        score = "" if pd.isna(r.score) else f"{r.score:+.1f}"
        lines.append(f"| {r.main_id} | {r.factor} | {r.criterion} | {score} | {r.rating or ''} | {r.status} | "
                     f"{r.trend} | {r.evidence} | {r.red_flags} |")
    rep = latest_report(conn, scoring_run_id)
    lines += ["", "## Report", ""]
    if rep is not None and rep["passed"]:
        lines.append(rep["text"].replace("\n## ", "\n### ").replace("## ", "### ", 1))
    else:
        lines.append("_No validated report available._")
    return "\n".join(lines) + "\n"


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


_FONT_CANDIDATES = [Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                    Path("/Library/Fonts/Arial Unicode.ttf")]


def markdown_to_pdf(markdown: str) -> bytes:
    """Simple PDF of a Markdown text (headings, tables and bullets rendered as plain lines)."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    font = next((p for p in _FONT_CANDIDATES if p.exists()), None)
    if font:
        pdf.add_font("body", "", str(font))
        family = "body"
    else:
        family = "helvetica"

    def clean(s: str) -> str:
        s = s.replace("**", "").replace("_", "")
        if family == "helvetica":   # core fonts are latin-1 only
            s = (s.replace("–", "-").replace("—", "-").replace("’", "'").replace("“", '"').replace("”", '"')
                 .replace("≥", ">=").replace("≤", "<=").replace("·", "-"))
            s = s.encode("latin-1", "replace").decode("latin-1")
        return s

    for line in markdown.splitlines():
        if line.startswith("|---"):
            continue
        size = 16 if line.startswith("# ") else 13 if line.startswith("## ") else 11 if line.startswith("### ") else 9
        pdf.set_font(family, size=size)
        text = clean(line.lstrip("# ").strip())
        if line.startswith("|"):
            text = "  ".join(c.strip() for c in line.strip("|").split("|"))
            text = clean(text)
        pdf.multi_cell(0, 5 if size == 9 else 7, text or " ", new_x="LMARGIN", new_y="NEXT")
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
