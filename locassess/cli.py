"""Command-line interface (SPEC §14). Further commands are added with their milestones."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from locassess.config import ScoringCfg, load_config
from locassess.criteria import load_criteria_csv
from locassess.db import repo
from locassess.stage2_score import NO_STATEMENT

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


@app.command("codebook-draft")
def codebook_draft(
    out: Path = typer.Option(Path("data/codebook.yaml"), "--out", help="Codebook file to write."),
    force: bool = typer.Option(False, "--force", help="Regenerate draft entries of an existing file."),
    config: Path = typer.Option(Path("config.yaml"), "--config", help="Configuration file."),
    db: Path = DbOption,
) -> None:
    """Draft data/codebook.yaml with the LLM (status: draft) for the author to review."""
    from locassess.codebook_draft import write_codebook_draft
    from locassess.criteria import load_codebook
    from locassess.providers.llm import make_llm

    cfg = load_config(config)
    conn = repo.connect(db)
    repo.init_db(conn)
    criteria = load_criteria_csv(Path("data/criteria.csv"))
    llm = make_llm(conn, cfg.models.llm_provider, cfg.models.llm_model)
    try:
        generated = write_codebook_draft(out, llm, criteria, cfg.models.llm_model, force=force,
                                         temperature=cfg.models.temperature, progress=typer.echo)
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc), param_hint="--force") from exc
    cb = load_codebook(out, criteria)
    typer.echo(f"wrote {out}: {len(generated)} entries drafted, {len(cb.criteria) - len(cb.drafts())} reviewed kept")


@app.command("extract")
def extract(
    corpus: int = typer.Option(..., "--corpus", help="Frozen corpus to extract evidence from."),
    model: str | None = typer.Option(None, "--model", help="jev | openai_decisions | llm_structured "
                                                           "(default: models.decision_provider)."),
    repeat: int = typer.Option(1, "--repeat", min=1, help="Number of repetitions (one extraction run each)."),
    config: Path = typer.Option(Path("config.yaml"), "--config", help="Configuration file."),
    codebook_path: Path = typer.Option(Path("data/codebook.yaml"), "--codebook", help="Codebook file."),
    db: Path = DbOption,
) -> None:
    """Stage 1: extract evidence with a decision model (Query A + Query B per article)."""
    from locassess.criteria import load_codebook
    from locassess.providers.decision import make_decision_model
    from locassess.stage1_extract import run_extraction, run_summary

    cfg = load_config(config)
    criteria = load_criteria_csv(Path("data/criteria.csv"))
    cb = load_codebook(codebook_path, criteria)
    if cb.drafts():
        typer.echo(f"note: codebook has {len(cb.drafts())} unreviewed (draft) entries")
    conn = repo.connect(db)
    repo.init_db(conn)
    dm = make_decision_model(conn, cfg, model)
    for _ in range(repeat):
        run_id = run_extraction(conn, corpus, dm, cb, codebook_path, criteria, cfg.scoring.extraction,
                                progress=typer.echo)
        s = run_summary(conn, run_id)
        typer.echo(f"extraction_run_id={run_id}  model={s['model']} ({s['model_version']})  "
                   f"repetition={s['repetition']}  method={s['confidence_method']}  "
                   f"A={s['responses_A']} B={s['responses_B']}  evidence={s['evidence']} "
                   f"(needs review: {s['needs_review']})")


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

    from locassess.pipeline import score as score_run

    conn = repo.connect(db)
    scoring_run_id, profile = score_run(conn, cfg, extraction_run, scoring_cfg, label)
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


def _load_codebook_for_search(criteria):
    from locassess.criteria import load_codebook

    path = Path("data/codebook.yaml")
    if not path.exists():
        typer.echo("note: data/codebook.yaml not found; queries use the sub-criteria names")
        return None
    cb = load_codebook(path, criteria)
    if cb.drafts():
        typer.echo(f"note: codebook has {len(cb.drafts())} unreviewed (draft) entries")
    return cb


def _report_corpus(conn, corpus_id: int) -> None:
    from locassess.stage0_corpus import selection_log

    summary, _ = selection_log(conn, corpus_id)
    for row in summary:
        typer.echo(f"  {row['step']:<14} {row['count']:>5}  {row['detail']}")
    typer.echo(f"corpus_id={corpus_id}  (review it, then: locassess corpus freeze --corpus {corpus_id})")


@corpus_app.command("build")
def corpus_build(config: Path = ConfigOption, db: Path = DbOption,
                 criteria_ids: str | None = typer.Option(
                     None, "--criteria", help="Comma-separated main criteria for a small test run, e.g. H02,H20."),
                 note: str | None = typer.Option(None, "--note")) -> None:
    """Generate queries, search, fetch, filter, rate relevance and select articles."""
    from locassess.providers.search import make_search
    from locassess.stage0_corpus import HttpFetcher, build_corpus

    cfg, conn, llm, criteria = _stage0_setup(config, db)
    if criteria_ids:
        wanted = [c.strip().upper() for c in criteria_ids.split(",") if c.strip()]
        unknown = sorted(set(wanted) - {m.main_id for m in criteria})
        if unknown:
            raise typer.BadParameter(f"unknown criteria: {', '.join(unknown)}", param_hint="--criteria")
        criteria = [m for m in criteria if m.main_id in wanted]
        note = note or f"test run: {', '.join(m.main_id for m in criteria)}"
    search = make_search(conn, cfg.search_provider, cfg.models.llm_model, cfg.models.temperature,
                         cfg.search_context_size)
    codebook = _load_codebook_for_search(criteria)
    corpus_id = build_corpus(conn, cfg, criteria, llm, search, HttpFetcher(conn), note=note,
                             progress=typer.echo, codebook=codebook)
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


@corpus_app.command("process")
def corpus_process(corpus: int = typer.Option(..., "--corpus"), config: Path = ConfigOption,
                   db: Path = DbOption) -> None:
    """Continue an unfinished corpus without new searches: fetch, filter, relevance, selection."""
    from locassess.stage0_corpus import HttpFetcher, process_corpus

    cfg, conn, llm, criteria = _stage0_setup(config, db)
    process_corpus(conn, corpus, cfg, criteria, llm, HttpFetcher(conn), progress=typer.echo)
    _report_corpus(conn, corpus)


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


@app.command("report")
def report(
    scoring_run: int = typer.Option(..., "--scoring-run", help="Scoring run to report on."),
    out: Path = typer.Option(Path("results"), "--out", help="Folder for the Markdown export."),
    config: Path = ConfigOption,
    db: Path = DbOption,
) -> None:
    """Stage 3: write the report from the profile, validate it, retry on violations."""
    from locassess.providers.llm import make_llm
    from locassess.stage3_report import generate_report

    cfg = load_config(config)
    conn = repo.connect(db)
    llm = make_llm(conn, cfg.models.llm_provider, cfg.models.llm_model)
    report_id = generate_report(conn, llm, scoring_run, cfg.report.max_attempts, cfg.models.temperature,
                                progress=typer.echo)
    row = conn.execute("SELECT * FROM report WHERE report_id = ?", (report_id,)).fetchone()
    if row["passed"]:
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"report_scoring{scoring_run}.md"
        path.write_text(row["text"] + "\n", encoding="utf-8")
        typer.echo(f"report_id={report_id} passed after {row['attempts']} attempt(s); wrote {path}")
    else:
        typer.echo(f"report_id={report_id} FAILED validation after {row['attempts']} attempts "
                   "(stored as failed; the UI shows the profile without report text)")
        for a in json.loads(row["checks_json"])[-1]["checks"]:
            for v in a["violations"]:
                typer.echo(f"  [{a['rule']}] {v}")


@app.command("baseline")
def baseline(
    corpus: int = typer.Option(..., "--corpus", help="Frozen corpus."),
    repeat: int = typer.Option(3, "--repeat", min=1, help="Repetitions (SPEC: 3–5)."),
    config: Path = ConfigOption,
    db: Path = DbOption,
) -> None:
    """Baseline (SPEC §11): the LLM assesses the frozen corpus directly, without rules."""
    from locassess.baseline import run_baseline
    from locassess.providers.llm import make_llm

    cfg = load_config(config)
    conn = repo.connect(db)
    repo.init_db(conn)
    llm = make_llm(conn, cfg.models.llm_provider, cfg.models.llm_model)
    crit = load_criteria_csv(Path("data/criteria.csv"))
    for _ in range(repeat):
        bid = run_baseline(conn, llm, corpus, crit, cfg.models.llm_model, cfg.models.temperature, progress=typer.echo)
        row = conn.execute("SELECT * FROM baseline_assessment WHERE baseline_id = ?", (bid,)).fetchone()
        data = json.loads(row["per_criterion_json"])
        rated = sum(1 for c in data.values() if c.get("score") is not None)
        overall = json.loads(row["overall_json"]).get("score")
        typer.echo(f"baseline_id={bid} repetition={row['repetition']}  criteria rated={rated}/26  overall={overall}")


annotate_app = typer.Typer(help="Manual gold standard: export a template, import the filled CSV.",
                           no_args_is_help=True)
app.add_typer(annotate_app, name="annotate")


@annotate_app.command("template")
def annotate_template(corpus: int = typer.Option(..., "--corpus"),
                      annotator: str = typer.Option(..., "--annotator", help="Your name or initials."),
                      out: Path = typer.Option(None, "--out", help="CSV to write (default results/annotation_...)."),
                      db: Path = DbOption) -> None:
    """Write a CSV with one row per selected article × main criterion to fill in."""
    from locassess.annotation import write_template

    out = out or Path("results") / f"annotation_corpus{corpus}_{annotator}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = write_template(repo.connect(db), corpus, load_criteria_csv(Path("data/criteria.csv")), annotator, out)
    typer.echo(f"wrote {out} ({n} rows). Fill affected (0/1); for affected = 1 also direction, intensity, "
               "evidence_type, sub_id (or overarching) and sentence_keys (e.g. S3,S7). Guideline: data/codebook.yaml")


@annotate_app.command("import")
def annotate_import(csv_path: Path = typer.Option(..., "--csv"), db: Path = DbOption) -> None:
    """Validate and import a filled annotation CSV (rows with empty `affected` are skipped)."""
    from locassess.annotation import AnnotationError, import_annotations

    try:
        n = import_annotations(repo.connect(db), csv_path, load_criteria_csv(Path("data/criteria.csv")))
    except AnnotationError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(f"imported {n} annotations")


@app.command("evaluate")
def evaluate(
    all_: bool = typer.Option(False, "--all", help="Run every metric for which data exists."),
    metric: list[str] = typer.Option([], "--metric", help="Run selected metrics (repeatable)."),
    out: Path = typer.Option(Path("results"), "--out"),
    config: Path = ConfigOption,
    db: Path = DbOption,
) -> None:
    """Evaluation (SPEC §13): metrics as CSV files in results/."""
    from locassess.evaluation.run_eval import METRICS, run_metrics

    unknown = [m for m in metric if m not in METRICS]
    if unknown or not (all_ or metric):
        raise typer.BadParameter(f"use --all or --metric with one of: {', '.join(METRICS)}", param_hint="--metric")
    cfg = load_config(config)
    for name, outcome in run_metrics(repo.connect(db), cfg, None if all_ else metric, out):
        typer.echo(f"{name:<24} {outcome}")


@app.command("ui")
def ui(db: Path = DbOption, config: Path = ConfigOption) -> None:
    """Start the Streamlit web UI."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "LOCASSESS_DB": str(db), "LOCASSESS_CONFIG": str(config)}
    app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path),
                                      "--server.showEmailPrompt", "false",      # no first-run email question
                                      "--browser.gatherUsageStats", "false"],   # no telemetry to Streamlit
                                     env=env))


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


if __name__ == "__main__":
    app()
