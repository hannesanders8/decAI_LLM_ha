"""Versioned query definition for Stage 1 (SPEC §8).

One place builds the state text and the Query A / Query B questions — with instructions,
options and option descriptions taken from the codebook — so that every decision model
(jev, openai_decisions, llm_structured) receives exactly the same input. Any change to the
wording here must increase QUERY_DEFINITION_VERSION.
"""

from __future__ import annotations

import re

from locassess.criteria import Codebook, MainCriterion
from locassess.providers.decision import Option, Question

QUERY_DEFINITION_VERSION = "1.0"

OVERARCHING = "overarching"


def render_state(location: dict, title: str | None, published_at: str | None,
                 sentences: list[tuple[str, str]], section: int = 1, n_sections: int = 1) -> str:
    """The state every model sees: location and one article section with sentence IDs."""
    place = ", ".join(p for p in (location.get("name"), location.get("region"), location.get("country")) if p)
    aliases = ", ".join(location.get("aliases") or [])
    head = [f"Location under assessment: {place}" + (f" (also: {aliases})" if aliases else ""),
            f"Article title: {title or '(none)'}", f"Publication date: {published_at or '(unknown)'}"]
    if n_sections > 1:
        head.append(f"Article section {section} of {n_sections}")
    body = "\n".join(f"[{k}] {t}" for k, t in sentences)
    return "\n".join(head) + "\n\nArticle text:\n" + body


def _affected_key(main_id: str) -> str:
    return f"{main_id}_affected"


def query_a_questions(codebook: Codebook, criteria: list[MainCriterion], location_name: str) -> list[Question]:
    """Query A: one yes/no question per main criterion."""
    out = []
    for m in criteria:
        e = codebook.criteria[m.main_id]
        out.append(Question(
            key=_affected_key(m.main_id),
            instructions=(f"Criterion {m.main_id} {m.name}. Definition: {e.definition}\n"
                          f"Does the article contain evidence on this criterion that concerns {location_name} "
                          f"itself (a fact, an announcement or an opinion)?"),
            options=(Option("yes", f"The article contains such evidence about {location_name}."),
                     Option("no", f"The criterion is not addressed for {location_name}, "
                                  "or only mentioned in passing.")),
        ))
    return out


def _direction_options(codebook: Codebook) -> tuple[Option, ...]:
    """Option descriptions parsed from the codebook's direction_definition ('+1 = …; -1 = …; 0 = …')."""
    parts = {}
    for chunk in codebook.direction_definition.split(";"):
        m = re.match(r"\s*([+-]?[01])\s*=\s*(.+)", chunk)
        if m:
            parts[str(int(m.group(1)))] = m.group(2).strip()
    return tuple(Option(v, parts.get(v, "")) for v in ("-1", "0", "1"))


def query_b_questions(codebook: Codebook, main: MainCriterion, sentence_keys: list[str],
                      location_name: str) -> list[Question]:
    """Query B: details of the evidence for one affected main criterion."""
    e = codebook.criteria[main.main_id]
    context = f"Criterion {main.main_id} {main.name}. Definition: {e.definition}"
    if e.notes:
        context += f" Boundaries: {e.notes}"
    examples = ""
    if e.positive_examples or e.negative_examples:
        examples = ("\nExamples of positive evidence: " + " | ".join(e.positive_examples) +
                    "\nExamples of negative evidence: " + " | ".join(e.negative_examples))

    questions = [
        Question("direction",
                 f"{context}{examples}\nWhat is the direction of the article's evidence on this criterion "
                 f"for {location_name} as a production location?",
                 _direction_options(codebook)),
        Question("intensity",
                 f"{context}\nHow strong is the effect described in the evidence on this criterion "
                 f"for {location_name}?",
                 tuple(Option(str(k), codebook.intensity_definition[k]) for k in (1, 2, 3))),
        Question("evidence_type",
                 f"{context}\nWhat kind of statement is the evidence on this criterion?",
                 tuple(Option(k, codebook.evidence_type_definition[k]) for k in ("fact", "announcement", "opinion"))),
    ]
    if main.sub_criteria:
        questions.append(Question(
            "sub_criterion",
            f"{context}\nWhich sub-criterion does the evidence mainly concern?",
            tuple(Option(s.sub_id, s.name) for s in main.sub_criteria)
            + (Option(OVERARCHING, "the criterion as a whole, or several sub-criteria equally"),),
        ))
    if len(sentence_keys) >= 2:   # a single-sentence section needs no question (rule: that sentence)
        questions.append(Question(
            "evidence_sentence",
            f"{context}\nWhich sentence of the article is the best evidence on this criterion for {location_name}?",
            tuple(Option(k) for k in sentence_keys),
        ))
    return questions
