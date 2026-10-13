# LocAssess

News-based assessment of production locations (Master's thesis, Universität Hamburg).
The full build specification is in [SPEC.md](SPEC.md).

**Status:** M1 (skeleton), M2 (Stage 2 scoring), M3 (Stage 0 corpus) and M5 (codebook draft) done; next M4 (Stage 1).

## Setup

Requires Python ≥ 3.11.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"         # add ",anthropic" for the optional Anthropic adapter
cp config.example.yaml config.yaml
cp .env.example .env            # then fill in the API keys you need
```

`config.yaml`, `.env`, `data/locassess.db` and `results/` are git-ignored. Never commit API keys.

## Commands

```bash
locassess init-db          # create data/locassess.db with all tables (idempotent)
locassess load-criteria    # load data/criteria.csv → 26 main criteria, 94 sub-criteria
```

### Codebook

```bash
locassess codebook-draft            # LLM drafts data/codebook.yaml (status: draft per criterion)
locassess codebook-draft --force    # regenerate draft entries; entries with status: reviewed are kept
```

The codebook is the model instruction for Stage 1 and the annotation guideline for the gold
standard. Review each entry in `data/codebook.yaml`, edit it, and set `status: reviewed`.
Stage 0 takes its search terms from the codebook when the file exists.

### Stage 0 — corpus

Fill `data/whitelist.csv` (columns `domain, outlet_name, media_type, language, country`) and set
`OPENAI_API_KEY` (or `BRAVE_API_KEY` with `search_provider: brave`) first.

```bash
locassess corpus build                       # queries → search → fetch → filter → relevance → selection
locassess corpus build --criteria H02,H20    # small test run
locassess corpus process --corpus 2          # finish an interrupted build without new searches
locassess corpus import --csv urls.csv       # hand-picked URLs (columns url, note); same filter/selection
locassess corpus log --corpus 1              # results/selection_log_corpus1.csv + per-article CSV
locassess corpus freeze --corpus 1           # split into sentences, make the corpus read-only
```

`build` and `import` print the selection log and do not freeze, so the corpus can be reviewed first.

### Stage 2 — scoring

```bash
locassess score --extraction-run 4 [--params params.yaml] [--label NAME] [--config config.yaml]
```

`score` runs Stage 2 (pure rules, no model calls) on the evidence of an extraction run and stores a
scoring run with all 26 criterion scores, red flags and the equal-weight overall score. The period
comes from the corpus (`period_from`/`period_to`), falling back to `config.yaml`. `--params` takes a
YAML that overrides parts of the `scoring` section, e.g. `scoring: {half_life_months: 12}`.

All commands accept `--db PATH`; `load-criteria` also accepts `--csv PATH`.

## Tests

```bash
pytest
```

All model and search calls are mocked in tests.

## Layout

```
data/criteria.csv        criteria framework (Hoang-Thi et al. 2026)
data/whitelist.csv       allowed news domains (maintained by the author)
locassess/config.py      config.yaml + .env loading, config hash
locassess/criteria.py    criteria + codebook loader (with validation)
locassess/codebook_draft.py  LLM draft of the codebook
locassess/db/            SQLite schema (incl. frozen-corpus triggers) and data access
locassess/providers/     provider interfaces and the call_log wrapper
locassess/prompts.py     versioned prompt loader (prompts/*.md)
locassess/stage0_corpus.py Stage 0 search, fetch, rule filter, selection, freeze, selection log
locassess/stage2_score.py  Stage 2 scoring (pure functions, SPEC §9)
locassess/cli.py         command-line interface
```

## Note on article texts

Full article texts are stored only locally for research (text and data mining). Do not redistribute them.
