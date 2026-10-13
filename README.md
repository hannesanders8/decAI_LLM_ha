# LocAssess

News-based assessment of production locations (Master's thesis, Universität Hamburg).
The full build specification is in [SPEC.md](SPEC.md).

**Status:** milestone M1 (skeleton): configuration, database schema, criteria loader, call logging.

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

## Commands (M1)

```bash
locassess init-db          # create data/locassess.db with all tables (idempotent)
locassess load-criteria    # load data/criteria.csv → 26 main criteria, 94 sub-criteria
```

Both accept `--db PATH`; `load-criteria` also accepts `--csv PATH`.

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
locassess/criteria.py    criteria loader
locassess/db/            SQLite schema (incl. frozen-corpus triggers) and data access
locassess/providers/     provider interfaces and the call_log wrapper
locassess/cli.py         command-line interface
```

## Note on article texts

Full article texts are stored only locally for research (text and data mining). Do not redistribute them.
