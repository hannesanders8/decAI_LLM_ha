"""Criteria framework (Hoang-Thi et al. 2026): 6 factors, 26 main criteria, 94 sub-criteria."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

EXPECTED_MAIN = 26
EXPECTED_SUB = 94

_MAIN_RE = re.compile(r"^H\d{2}$")


class CriteriaError(ValueError):
    pass


@dataclass(frozen=True)
class SubCriterion:
    sub_id: str
    main_id: str
    name: str
    name_de: str | None = None


@dataclass
class MainCriterion:
    main_id: str
    factor: str
    name: str
    name_de: str | None = None
    sub_criteria: list[SubCriterion] = field(default_factory=list)


def load_criteria_csv(
    path: str | Path,
    expected_main: int | None = EXPECTED_MAIN,
    expected_sub: int | None = EXPECTED_SUB,
) -> list[MainCriterion]:
    """Read criteria.csv into main criteria with their sub-criteria, validating structure.

    Only the `_en` columns are used as names; `_de` is kept as an optional tooltip.
    """
    mains: dict[str, MainCriterion] = {}
    seen_sub: set[str] = set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        for lineno, row in enumerate(csv.DictReader(f), start=2):
            main_id = (row.get("main_id") or "").strip()
            if not _MAIN_RE.match(main_id):
                raise CriteriaError(f"line {lineno}: invalid main_id {main_id!r}")
            factor = (row.get("factor_en") or "").strip()
            name = (row.get("main_criterion_en") or "").strip()
            if not factor or not name:
                raise CriteriaError(f"line {lineno}: missing factor_en or main_criterion_en")

            main = mains.get(main_id)
            if main is None:
                main = mains[main_id] = MainCriterion(
                    main_id, factor, name, (row.get("main_criterion_de") or "").strip() or None
                )
            elif (main.factor, main.name) != (factor, name):
                raise CriteriaError(f"line {lineno}: inconsistent factor/name for {main_id}")

            sub_id = (row.get("sub_id") or "").strip()
            if not sub_id:
                continue
            if not re.fullmatch(rf"{main_id}\.\d{{2}}", sub_id):
                raise CriteriaError(f"line {lineno}: sub_id {sub_id!r} does not belong to {main_id}")
            if sub_id in seen_sub:
                raise CriteriaError(f"line {lineno}: duplicate sub_id {sub_id}")
            sub_name = (row.get("sub_criterion_en") or "").strip()
            if not sub_name:
                raise CriteriaError(f"line {lineno}: missing sub_criterion_en for {sub_id}")
            seen_sub.add(sub_id)
            main.sub_criteria.append(
                SubCriterion(sub_id, main_id, sub_name, (row.get("sub_criterion_de") or "").strip() or None)
            )

    if expected_main is not None and len(mains) != expected_main:
        raise CriteriaError(f"expected {expected_main} main criteria, found {len(mains)}")
    if expected_sub is not None and len(seen_sub) != expected_sub:
        raise CriteriaError(f"expected {expected_sub} sub-criteria, found {len(seen_sub)}")
    return sorted(mains.values(), key=lambda m: m.main_id)


# ---------------------------------------------------------------------------
# Codebook (data/codebook.yaml): model instruction (Stage 1) and annotation guideline
# ---------------------------------------------------------------------------

CODEBOOK_STATUSES = ("draft", "reviewed")


@dataclass(frozen=True)
class CodebookEntry:
    main_id: str
    status: str
    name: str
    definition: str
    sub_criteria: tuple[str, ...]
    positive_examples: tuple[str, ...]
    negative_examples: tuple[str, ...]
    search_terms: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class Codebook:
    version: str
    direction_definition: str
    intensity_definition: dict[int, str]
    evidence_type_definition: dict[str, str]
    criteria: dict[str, CodebookEntry]

    @property
    def is_reviewed(self) -> bool:
        return all(e.status == "reviewed" for e in self.criteria.values())

    def drafts(self) -> list[str]:
        return [k for k, e in self.criteria.items() if e.status != "reviewed"]


def _str_list(value, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise CriteriaError(f"{where}: expected a list of strings")
    return tuple(v.strip() for v in value if v.strip())


def load_codebook(path: str | Path, criteria: list[MainCriterion] | None = None) -> Codebook:
    """Load and validate codebook.yaml; with `criteria`, check it covers exactly the framework."""
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for key in ("version", "direction_definition", "intensity_definition", "evidence_type_definition", "criteria"):
        if key not in data:
            raise CriteriaError(f"{path}: missing top-level key {key!r}")
    intensity = {int(k): str(v) for k, v in data["intensity_definition"].items()}
    if set(intensity) != {1, 2, 3}:
        raise CriteriaError(f"{path}: intensity_definition needs levels 1, 2, 3")
    types = {str(k): str(v) for k, v in data["evidence_type_definition"].items()}
    if set(types) != {"fact", "announcement", "opinion"}:
        raise CriteriaError(f"{path}: evidence_type_definition needs fact, announcement, opinion")

    entries: dict[str, CodebookEntry] = {}
    for main_id, e in (data["criteria"] or {}).items():
        where = f"{path}: criteria.{main_id}"
        if not isinstance(e, dict):
            raise CriteriaError(f"{where}: expected a mapping")
        status = str(e.get("status", "draft"))
        if status not in CODEBOOK_STATUSES:
            raise CriteriaError(f"{where}: status must be one of {CODEBOOK_STATUSES}")
        definition = str(e.get("definition") or "").strip()
        if not definition:
            raise CriteriaError(f"{where}: definition is empty")
        entries[str(main_id)] = CodebookEntry(
            main_id=str(main_id), status=status, name=str(e.get("name") or "").strip(), definition=definition,
            sub_criteria=_str_list(e.get("sub_criteria"), where),
            positive_examples=_str_list(e.get("positive_examples"), where),
            negative_examples=_str_list(e.get("negative_examples"), where),
            search_terms=_str_list(e.get("search_terms"), where),
            notes=str(e.get("notes") or "").strip(),
        )

    if criteria is not None:
        expected = {m.main_id: m for m in criteria}
        missing, extra = sorted(set(expected) - set(entries)), sorted(set(entries) - set(expected))
        if missing or extra:
            raise CriteriaError(f"{path}: missing criteria {missing}, unknown criteria {extra}")
        for main_id, e in entries.items():
            m = expected[main_id]
            if e.name != m.name:
                raise CriteriaError(f"{path}: {main_id} name {e.name!r} differs from criteria.csv ({m.name!r})")
            ids = [s.split()[0] for s in e.sub_criteria]
            if ids != [s.sub_id for s in m.sub_criteria]:
                raise CriteriaError(f"{path}: {main_id} sub_criteria do not match criteria.csv")
            if not e.search_terms:
                raise CriteriaError(f"{path}: {main_id} has no search_terms")

    return Codebook(
        version=str(data["version"]), direction_definition=str(data["direction_definition"]),
        intensity_definition=intensity, evidence_type_definition=types,
        criteria=dict(sorted(entries.items())),
    )
