"""Stage 3 — report generation and rule-based validation (SPEC §10.1–10.2).

The LLM only receives the profile JSON (never raw articles) and writes text. Which criterion
belongs to which report section is decided here by fixed rules; the validator checks the text
against the profile and the report is regenerated with the list of violations if needed.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from locassess.db import repo
from locassess.providers.llm import LLM

SECTIONS = ("Summary", "Red flags", "Strengths", "Weaknesses", "Thinly evidenced criteria", "No evidence")
CITED_SECTIONS = ("Red flags", "Strengths", "Weaknesses")
SUMMARY_MAX_SENTENCES = 3

CITATION_RE = re.compile(r"\[E\d+(?:, E\d+)*\]")            # SPEC §10.2
BRACKET_E_RE = re.compile(r"\[[^\[\]]*\bE\d+[^\[\]]*\]")      # anything bracketed that cites an E-ID
# a rating in parentheses, also with extra content such as "(red, -2.0)" — every such rating is checked
RATING_RE = re.compile(r"\(\s*(red|green|yellow)\b[^()\n]*\)", re.I)
CRIT_RATING_RE = re.compile(r"\b(H\d{2})\b[^\n\[\]()]{0,100}\(\s*(red|green|yellow)\b[^()\n]*\)", re.I)
CRIT_ID_RE = re.compile(r"\bH\d{2}\b")

Progress = Callable[[str], None]


def eid(evidence_id: int) -> str:
    return f"E{evidence_id}"


# ---------------------------------------------------------------------------
# Profile JSON (the only input of the report model)
# ---------------------------------------------------------------------------

def report_plan(criteria: list[dict]) -> dict[str, list[str]]:
    """Fixed assignment of criteria to report sections (author-approved defaults):
    strengths = sufficient evidence and score > 0; weaknesses = sufficient and score <= 0;
    thin = status thin; no_evidence = no_statement."""
    plan = {"strengths": [], "weaknesses": [], "thin": [], "no_evidence": []}
    for c in criteria:
        if c["status"] == "no_statement":
            plan["no_evidence"].append(c["id"])
        elif c["status"] == "thin":
            plan["thin"].append(c["id"])
        elif c["score"] is not None and c["score"] > 0:
            plan["strengths"].append(c["id"])
        else:
            plan["weaknesses"].append(c["id"])
    return plan


def build_profile_json(conn: sqlite3.Connection, scoring_run_id: int) -> dict:
    run = conn.execute("SELECT * FROM scoring_run WHERE scoring_run_id = ?", (scoring_run_id,)).fetchone()
    if run is None:
        raise LookupError(f"scoring run {scoring_run_id} not found")
    ext = repo.get_extraction_run(conn, run["extraction_run_id"])
    corpus = repo.get_corpus(conn, ext["corpus_id"])
    names = {r["main_id"]: (r["name"], r["factor"]) for r in conn.execute("SELECT * FROM criterion")}

    criteria, red_flags, cited = [], [], set()
    for r in conn.execute("SELECT * FROM criterion_score WHERE scoring_run_id = ? ORDER BY main_id", (scoring_run_id,)):
        ids = json.loads(r["evidence_ids_json"] or "[]")
        cited.update(ids)
        criteria.append({
            "id": r["main_id"], "name": names[r["main_id"]][0], "factor": names[r["main_id"]][1],
            "score": None if r["score"] is None else round(r["score"], 1), "rating": r["rating"],
            "status": r["status"], "trend": r["trend"], "evidence_ids": [eid(i) for i in ids],
        })
        for f in json.loads(r["red_flags_json"] or "[]"):
            red_flags.append({"rule": f["rule"], "criterion": f["main_id"], "description": f["description"],
                              "evidence_ids": [eid(i) for i in f["evidence_ids"]]})

    evidence = {}
    if cited:
        marks = ",".join("?" * len(cited))
        for e in conn.execute(
            f"SELECT e.*, a.outlet, a.title FROM evidence e LEFT JOIN article a USING (article_id) "
            f"WHERE e.evidence_id IN ({marks}) ORDER BY e.evidence_id", sorted(cited),
        ):
            evidence[eid(e["evidence_id"])] = {
                "criterion": e["main_id"], "sentence": e["sentence_text"], "outlet": e["outlet"],
                "date": e["event_date"], "direction": e["direction"], "intensity": e["intensity"],
                "type": e["evidence_type"], "needs_review": bool(e["needs_review"]),
            }

    overall = conn.execute("SELECT * FROM overall_score WHERE scoring_run_id = ? ORDER BY overall_id LIMIT 1",
                           (scoring_run_id,)).fetchone()
    return {
        "location": json.loads(corpus["location_json"]),
        "period": {"from": corpus["period_from"], "to": corpus["period_to"]},
        "overall_score": None if overall is None or overall["score"] is None else round(overall["score"], 2),
        "red_flags": red_flags,
        "report_plan": report_plan(criteria),
        "criteria": criteria,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# 10.2 Validation (rules)
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    rule: str
    passed: bool
    violations: list[str] = field(default_factory=list)


def split_sections(text: str) -> dict[str, str]:
    """Body text per '## <Section>' heading (only the six fixed sections)."""
    parts: dict[str, str] = {}
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, re.M))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts[m.group(1).strip()] = text[m.end():end].strip()
    return parts


def paragraphs(body: str) -> list[str]:
    """Paragraphs and list items of a section."""
    out = []
    for block in re.split(r"\n\s*\n", body.strip()):
        lines = [l for l in block.splitlines() if l.strip()]
        if any(re.match(r"\s*([-*]|\d+\.)\s", l) for l in lines):
            item: list[str] = []
            for l in lines:
                if re.match(r"\s*([-*]|\d+\.)\s", l) and item:
                    out.append(" ".join(item))
                    item = []
                item.append(l.strip())
            if item:
                out.append(" ".join(item))
        elif lines:
            out.append(" ".join(l.strip() for l in lines))
    return out


def _is_none(p: str) -> bool:
    return p.strip().rstrip(".").lower() in ("none", "no red flags", "-")


def _count_sentences(text: str) -> int:
    import pysbd

    return len([s for s in pysbd.Segmenter(language="en", clean=False).segment(text) if s.strip()])


def validate_report(text: str, profile: dict) -> list[CheckResult]:
    plan = profile["report_plan"]
    known_ids = set(profile["evidence"])
    ratings = {c["id"]: c["rating"] for c in profile["criteria"]}
    sections = split_sections(text)
    results: list[CheckResult] = []

    # structure: the six fixed sections, in order
    headings = [m.group(1).strip() for m in re.finditer(r"^##\s+(.+?)\s*$", text, re.M)]
    v = [] if headings == list(SECTIONS) else [f"headings must be exactly {list(SECTIONS)} in this order, got {headings}"]
    results.append(CheckResult("structure", not v, v))

    # all cited IDs exist (and citations use the fixed format)
    v = []
    for b in BRACKET_E_RE.findall(text):
        if not CITATION_RE.fullmatch(b):
            v.append(f"malformed citation {b} (use the format [E1, E2])")
    for c in CITATION_RE.findall(text):
        for i in re.findall(r"E\d+", c):
            if i not in known_ids:
                v.append(f"cited evidence {i} does not exist in the profile")
    results.append(CheckResult("cited_ids_exist", not v, v))

    # every paragraph in Red flags / Strengths / Weaknesses cites at least one ID
    v = []
    for s in CITED_SECTIONS:
        for p in paragraphs(sections.get(s, "")):
            if not CITATION_RE.search(p) and not _is_none(p):
                v.append(f"{s}: paragraph without evidence citation: {p[:80]!r}")
    results.append(CheckResult("paragraphs_cite", not v, v))

    # stated ratings match the profile
    v = []
    covered = set()
    for m in CRIT_RATING_RE.finditer(text):
        covered.add(text.rfind("(", 0, m.start(2)))
        stated = m.group(2).lower()
        if m.group(1) not in ratings:
            v.append(f"{m.group(1)} is not a criterion of the profile")
        elif ratings[m.group(1)] != stated:
            v.append(f"{m.group(1)} stated as ({stated}), profile says ({ratings[m.group(1)]})")
    for m in RATING_RE.finditer(text):
        if m.start() not in covered:
            v.append(f"rating ({m.group(1)}) without a criterion ID before it on the same line")
    results.append(CheckResult("ratings_match", not v, v))

    # all red flags appear under Red flags
    rf_text = sections.get("Red flags", "")
    rf_ids = set(re.findall(r"E\d+", " ".join(CITATION_RE.findall(rf_text))))
    v = [f"red flag {f['rule']} for {f['criterion']} ({', '.join(f['evidence_ids'])}) not cited under Red flags"
         for f in profile["red_flags"] if not set(f["evidence_ids"]) <= rf_ids]
    results.append(CheckResult("red_flags_complete", not v, v))

    # "No evidence" complete, and no_statement criteria appear nowhere else (no speculation)
    ne_text = sections.get("No evidence", "")
    v = [f"{c} missing under No evidence" for c in plan["no_evidence"] if c not in CRIT_ID_RE.findall(ne_text)]
    for s, body in sections.items():
        if s != "No evidence":
            v += [f"{c} has no evidence but appears under {s}" for c in plan["no_evidence"]
                  if c in CRIT_ID_RE.findall(body)]
    results.append(CheckResult("no_evidence_complete", not v, v))

    # every criterion of the plan appears in its section
    v = []
    for key, s in (("strengths", "Strengths"), ("weaknesses", "Weaknesses"), ("thin", "Thinly evidenced criteria")):
        found = set(CRIT_ID_RE.findall(sections.get(s, "")))
        v += [f"{c} missing under {s}" for c in plan[key] if c not in found]
    results.append(CheckResult("sections_complete", not v, v))

    # summary of at most three sentences
    n = _count_sentences(sections.get("Summary", ""))
    v = [] if 1 <= n <= SUMMARY_MAX_SENTENCES else [f"Summary has {n} sentences (1–{SUMMARY_MAX_SENTENCES} allowed)"]
    results.append(CheckResult("summary_length", not v, v))
    return results


# ---------------------------------------------------------------------------
# Generation with retries
# ---------------------------------------------------------------------------

def generate_report(conn: sqlite3.Connection, llm: LLM, scoring_run_id: int, max_attempts: int = 3,
                    temperature: float = 0, progress: Progress = lambda m: None) -> int:
    """Write, validate and (if needed) regenerate the report. Returns the report_id."""
    profile = build_profile_json(conn, scoring_run_id)
    profile_json = json.dumps(profile, ensure_ascii=False, indent=1)
    attempts: list[dict] = []
    violations: list[str] = []
    text, passed, prompt_version = "", False, None
    for attempt in range(1, max_attempts + 1):
        progress(f"report attempt {attempt}/{max_attempts}")
        block = ("Your previous version of this report broke these rules. Write the whole report again "
                 "and fix every one of them:\n" + "\n".join(f"- {x}" for x in violations) + "\n\n") if violations else ""
        result = llm.complete("report", {"profile_json": profile_json, "violations_block": block},
                              temperature=temperature)
        text, prompt_version = result.texts[0].strip(), result.prompt_version
        checks = validate_report(text, profile)
        violations = [x for c in checks for x in c.violations]
        attempts.append({"attempt": attempt, "passed": not violations,
                         "checks": [asdict(c) for c in checks], "call_ids": result.call_ids})
        progress(f"  {'passed' if not violations else f'{len(violations)} violation(s)'}")
        if not violations:
            passed = True
            break
    with conn:
        cur = conn.execute(
            "INSERT INTO report (scoring_run_id, text, attempts, passed, checks_json, prompt_version) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (scoring_run_id, text, len(attempts), int(passed), json.dumps(attempts, ensure_ascii=False), prompt_version),
        )
    return cur.lastrowid
