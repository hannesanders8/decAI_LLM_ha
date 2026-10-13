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


def load_codebook(path: str | Path) -> dict:
    """Codebook loading is implemented in milestone M5."""
    raise NotImplementedError("codebook loading arrives in milestone M5")
