"""Command-line interface (SPEC §14). Further commands are added with their milestones."""

from __future__ import annotations

from pathlib import Path

import typer

from locassess.criteria import load_criteria_csv
from locassess.db import repo

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


if __name__ == "__main__":
    app()
