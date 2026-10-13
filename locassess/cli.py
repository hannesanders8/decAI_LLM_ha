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


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


if __name__ == "__main__":
    app()
