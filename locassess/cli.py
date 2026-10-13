"""Command-line interface (SPEC §14). Further commands are added with their milestones."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from locassess.config import ScoringCfg, load_config
from locassess.criteria import load_criteria_csv
from locassess.db import repo
from locassess.stage2_score import NO_STATEMENT, ScoringParams, build_profile, month_index, month_str

app = typer.Typer(help="LocAssess — news-based assessment of production locations.", no_args_is_help=True)

DbOption = typer.Option(repo.DEFAULT_DB_PATH, "--db", help="SQLite database file.")


@app.command("init-db")
def init_db(db: Path = DbOption) -> None:
    """Create the database schema (idempotent)."""
    conn = repo.connect(db)
    repo.init_db(conn)
    conn.close()
    typer.echo(f"Database initialised: {db}")


@app.command("load-criteria")
def load_criteria(
    db: Path = DbOption,
    csv: Path = typer.Option(Path("data/criteria.csv"), "--csv", help="Criteria CSV file."),
) -> None:
    """Load the criteria framework from criteria.csv into the database."""
    criteria = load_criteria_csv(csv)
    conn = repo.connect(db)
    repo.init_db(conn)
    repo.upsert_criteria(conn, criteria)
    n_main, n_sub = repo.count_criteria(conn)
    conn.close()
    typer.echo(f"{n_main} main criteria, {n_sub} sub-criteria")


@app.command("score")
def score(
    extraction_run: int = typer.Option(..., "--extraction-run", help="Extraction run to score."),
    config: Path = typer.Option(Path("config.yaml"), "--config", help="Configuration file."),
    params: Path | None = typer.Option(None, "--params", help="YAML overriding the `scoring` section."),
    label: str | None = typer.Option(None, "--label", help="Label for this scoring run."),
    db: Path = DbOption,
) -> None:
    """Stage 2: score an extraction run with fixed rules (no model calls)."""
    cfg = load_config(config)
    scoring_cfg = cfg.scoring
    if params is not None:
        override = yaml.safe_load(params.read_text(encoding="utf-8")) or {}
        override = override.get("scoring", override)
        scoring_cfg = ScoringCfg.model_validate(_deep_merge(cfg.scoring.model_dump(), override))

    conn = repo.connect(db)
    period_from, period_to = repo.extraction_run_period(conn, extraction_run)
    if period_to is None:
        period_to = cfg.resolved_run_date()
    if period_from is None:
        period_from = month_str(month_index(period_to) - 12 * cfg.period.years_back)
    p = ScoringParams.from_config(scoring_cfg, run_date=period_to, period_from=period_from)

    profile = build_profile(
        repo.load_evidence(conn, extraction_run), repo.criteria_factors(conn).keys(), p,
        default_slider=cfg.weights_default, factors=repo.criteria_factors(conn),
    )
    scoring_run_id = repo.save_scoring_run(conn, extraction_run, profile, label)
    conn.close()

    typer.echo(f"scoring_run_id={scoring_run_id}")
    for c in profile.criteria.values():
        if c.status != NO_STATEMENT:
            s = "n/a" if c.score is None else f"{c.score:+.1f}"
            typer.echo(f"  {c.main_id}  {s:>4}  {c.rating or 'n/a':<6}  {c.status:<10}  trend: {c.trend}")
    for f in profile.red_flags:
        typer.echo(f"  red flag {f.rule} {f.main_id} evidence {list(f.evidence_ids)}")
    g = profile.overall.score
    typer.echo(f"  overall: {'n/a' if g is None else f'{g:+.2f}'}")


corpus_app = typer.Typer(help="Stage 0: build, import, freeze and inspect corpora.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")

ConfigOption = typer.Option(Path("config.yaml"), "--config", help="Configuration file.")


def _stage0_setup(config: Path, db: Path):
    from locassess.providers.llm import make_llm

    cfg = load_config(config)
    conn = repo.connect(db)
    repo.init_db(conn)
    llm = make_llm(conn, cfg.models.llm_provider, cfg.models.llm_model)
    criteria = load_criteria_csv(Path("data/criteria.csv"))
    return cfg, conn, llm, criteria


def _report_corpus(conn, corpus_id: int) -> None:
    from locassess.stage0_corpus import selection_log

    summary, _ = selection_log(conn, corpus_id)
    for row in summary:
        typer.echo(f"  {row['step']:<14} {row['count']:>5}  {row['detail']}")
    typer.echo(f"corpus_id={corpus_id}  (review it, then: locassess corpus freeze --corpus {corpus_id})")


@corpus_app.command("build")
def corpus_build(config: Path = ConfigOption, db: Path = DbOption,
                 note: str | None = typer.Option(None, "--note")) -> None:
    """Generate queries, search, fetch, filter, rate relevance and select articles."""
    from locassess.providers.search import make_search
    from locassess.stage0_corpus import HttpFetcher, build_corpus

    cfg, conn, llm, criteria = _stage0_setup(config, db)
    search = make_search(conn, cfg.search_provider, cfg.models.llm_model, cfg.models.temperature)
    corpus_id = build_corpus(conn, cfg, criteria, llm, search, HttpFetcher(conn), note=note, progress=typer.echo)
    _report_corpus(conn, corpus_id)


@corpus_app.command("import")
def corpus_import(csv: Path = typer.Option(..., "--csv", help="CSV with columns url, note."),
                  config: Path = ConfigOption, db: Path = DbOption,
                  note: str | None = typer.Option(None, "--note")) -> None:
    """Import hand-picked URLs; fetch, filter, relevance and selection run as in `build`."""
    from locassess.stage0_corpus import HttpFetcher, import_corpus

    cfg, conn, llm, criteria = _stage0_setup(config, db)
    corpus_id = import_corpus(conn, cfg, criteria, llm, csv, HttpFetcher(conn), note=note, progress=typer.echo)
    _report_corpus(conn, corpus_id)


@corpus_app.command("freeze")
def corpus_freeze(corpus: int = typer.Option(..., "--corpus"), config: Path = ConfigOption,
                  db: Path = DbOption) -> None:
    """Split selected articles into sentences and make the corpus read-only."""
    from locassess.stage0_corpus import freeze

    cfg = load_config(config)
    conn = repo.connect(db)
    n = freeze(conn, corpus, cfg.models.max_options_per_question)
    typer.echo(f"corpus {corpus} frozen with {n} articles")


@corpus_app.command("log")
def corpus_log(corpus: int = typer.Option(..., "--corpus"), out: Path = typer.Option(Path("results"), "--out"),
               db: Path = DbOption) -> None:
    """Export the selection log (summary + per-article) as CSV."""
    from locassess.stage0_corpus import export_selection_log

    conn = repo.connect(db)
    for p in export_selection_log(conn, corpus, out):
        typer.echo(f"wrote {p}")


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


if __name__ == "__main__":
    app()
