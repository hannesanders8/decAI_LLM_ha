"""Stage services shared by the CLI and the Streamlit UI (one function per pipeline step)."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import date
from pathlib import Path

from locassess.config import AppConfig, LocationCfg, ScoringCfg
from locassess.criteria import Codebook, MainCriterion, load_codebook, load_criteria_csv
from locassess.db import repo
from locassess.stage2_score import Profile, ScoringParams, build_profile, month_index, month_str

CRITERIA_CSV = Path("data/criteria.csv")
CODEBOOK = Path("data/codebook.yaml")

Progress = Callable[[str], None]


def _quiet(_: str) -> None:
    pass


def criteria() -> list[MainCriterion]:
    return load_criteria_csv(CRITERIA_CSV)


def codebook_or_none(crit: list[MainCriterion], path: Path = CODEBOOK) -> Codebook | None:
    return load_codebook(path, crit) if path.exists() else None


def with_location(cfg: AppConfig, name: str, country: str, local_language: str, region: str | None = None,
                  aliases: list[str] | None = None, years_back: int | None = None,
                  run_date: date | None = None) -> AppConfig:
    """A copy of the configuration for another location / period (used by the UI)."""
    loc = LocationCfg(name=name, region=region or None, country=country, local_language=local_language,
                      aliases=aliases or [])
    period = cfg.period.model_copy(update={k: v for k, v in
                                           {"years_back": years_back, "run_date": run_date}.items() if v is not None})
    return cfg.model_copy(update={"location": loc, "period": period})


# --- Stage 0 ------------------------------------------------------------------

def build_corpus(conn: sqlite3.Connection, cfg: AppConfig, criteria_ids: list[str] | None = None,
                 note: str | None = None, progress: Progress = _quiet) -> int:
    from locassess.providers.llm import make_llm
    from locassess.providers.search import make_search
    from locassess.stage0_corpus import HttpFetcher
    from locassess.stage0_corpus import build_corpus as _build

    crit = criteria()
    if criteria_ids:
        crit = [m for m in crit if m.main_id in set(criteria_ids)]
    llm = make_llm(conn, cfg.models.llm_provider, cfg.models.llm_model)
    search = make_search(conn, cfg.search_provider, cfg.models.llm_model, cfg.models.temperature,
                         cfg.search_context_size)
    return _build(conn, cfg, crit, llm, search, HttpFetcher(conn), note=note, progress=progress,
                  codebook=codebook_or_none(criteria()))


def freeze_corpus(conn: sqlite3.Connection, cfg: AppConfig, corpus_id: int) -> int:
    from locassess.stage0_corpus import freeze

    return freeze(conn, corpus_id, cfg.models.max_options_per_question)


# --- Stage 1 ------------------------------------------------------------------

def extract(conn: sqlite3.Connection, cfg: AppConfig, corpus_id: int, model: str | None = None,
            repeat: int = 1, codebook_path: Path = CODEBOOK, progress: Progress = _quiet) -> list[int]:
    from locassess.providers.decision import make_decision_model
    from locassess.stage1_extract import run_extraction

    crit = criteria()
    cb = load_codebook(codebook_path, crit)
    dm = make_decision_model(conn, cfg, model)
    return [run_extraction(conn, corpus_id, dm, cb, codebook_path, crit, cfg.scoring.extraction, progress=progress)
            for _ in range(repeat)]


# --- Stage 2 ------------------------------------------------------------------

def score(conn: sqlite3.Connection, cfg: AppConfig, extraction_run_id: int, scoring_cfg: ScoringCfg | None = None,
          label: str | None = None) -> tuple[int, Profile]:
    """Score an extraction run; the period comes from its corpus (fallback: config)."""
    period_from, period_to = repo.extraction_run_period(conn, extraction_run_id)
    if period_to is None:
        period_to = cfg.resolved_run_date()
    if period_from is None:
        period_from = month_str(month_index(period_to) - 12 * cfg.period.years_back)
    params = ScoringParams.from_config(scoring_cfg or cfg.scoring, run_date=period_to, period_from=period_from)
    factors = repo.criteria_factors(conn)
    profile = build_profile(repo.load_evidence(conn, extraction_run_id), factors.keys(), params,
                            default_slider=cfg.weights_default, factors=factors)
    return repo.save_scoring_run(conn, extraction_run_id, profile, label), profile


# --- Stage 3 ------------------------------------------------------------------

def report(conn: sqlite3.Connection, cfg: AppConfig, scoring_run_id: int, progress: Progress = _quiet) -> int:
    from locassess.providers.llm import make_llm
    from locassess.stage3_report import generate_report

    llm = make_llm(conn, cfg.models.llm_provider, cfg.models.llm_model)
    return generate_report(conn, llm, scoring_run_id, cfg.report.max_attempts, cfg.models.temperature,
                           progress=progress)
