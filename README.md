# LocAssess

News-based assessment of production locations (Master's thesis, Universität Hamburg).
The full build specification is in [SPEC.md](SPEC.md).

**Status:** all milestones M1–M8 done (Stage 0–3, codebook draft, UI, baseline and evaluation).

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

To save API cost, `corpus.max_found_articles` (e.g. 15) stops the search once that many unique
articles have been found; they then go through fetch, filter, relevance and selection as usual.
Queries run round-robin over the criteria (first query of every criterion, then the second …),
so the limit does not favour the first criteria; further hits are logged as `over_limit`.

### Stage 1 — evidence extraction

```bash
locassess extract --corpus 3 --model openai_decisions          # OpenAI Decisions API (gpt-6-luna)
locassess extract --corpus 3 --model llm_structured --repeat 3 # gpt-4.1 + JSON schema, logprobs or sampling
locassess extract --corpus 3 --model jev                       # TypeSafe Jev; needs TYPESAFE_API_KEY
```

Runs on a frozen corpus. Per article section: Query A (26 yes/no questions), then Query B for every
criterion with P(yes) ≥ `affected_threshold`. All decision models receive the same questions,
options and codebook text (`locassess/query_definition.py`, versioned). The full probability
distributions are stored in `model_response`; fixed rules derive `evidence` (SPEC §8.4).

### Stage 2 — scoring

```bash
locassess score --extraction-run 4 [--params params.yaml] [--label NAME] [--config config.yaml]
```

`score` runs Stage 2 (pure rules, no model calls) on the evidence of an extraction run and stores a
scoring run with all 26 criterion scores, red flags and the equal-weight overall score. The period
comes from the corpus (`period_from`/`period_to`), falling back to `config.yaml`. `--params` takes a
YAML that overrides parts of the `scoring` section, e.g. `scoring: {half_life_months: 12}`.

### Stage 3 — report

```bash
locassess report --scoring-run 2      # writes results/report_scoring2.md if validation passes
```

The LLM receives only the profile JSON (scores, ratings, red flags, evidence sentences with outlet
and date) and writes six fixed sections. Rule-based checks (SPEC §10.2 plus structure, section
completeness and summary length) validate the text; on violations the report is regenerated with
the list of violations, up to `report.max_attempts`. Attempts and per-rule results are stored.

### Baseline, gold standard and evaluation

```bash
locassess baseline --corpus 3 --repeat 3                     # direct LLM assessment of the frozen corpus (SPEC §11)
locassess annotate template --corpus 3 --annotator HA         # CSV: one row per article × criterion
locassess annotate import --csv results/annotation_corpus3_HA.csv
locassess evaluate --all                                      # all metrics of SPEC §13 → results/eval_*.csv
locassess evaluate --metric hallucination --metric cost_latency
```

The baseline also cites verbatim quotes per criterion; the evaluation checks them against the
article texts (hallucination, share of statements backed by evidence). Metrics without the
necessary data are skipped with a reason: extraction quality and calibration need annotations,
stability needs ≥ 2 repetitions, search reproducibility two corpora of the same location.
Add current model prices to `prices` in `config.yaml` for the cost evaluation.

| Metric (`--metric`) | Output | SQ |
|---|---|---|
| `search_reproducibility` | Jaccard of selected URLs between corpora of the same location | — |
| `extraction_quality` | precision / recall / F1 / κ of "affected", agreement on direction, intensity, type, sub-criterion, sentence (+ per criterion) | SQ1 |
| `hallucination` | share of hybrid evidence sentences / baseline quotes not found in the article | SQ1 |
| `extraction_stability` | identical cells and Fleiss' κ across repetitions | SQ1 |
| `calibration` | Brier score, PR-AUC, reliability-diagram bins from stored P(affected) | SQ1 |
| `scoring_robustness` | rating stability over 36 parameter variants; ranking Spearman over 1000 random slider draws | SQ2 |
| `baseline_comparison` | score spread per criterion, share backed by evidence, rating agreement | SQ3 |
| `report_reliability` | pass rate, attempts, failure rate per rule | SQ3 |
| `cost_latency` | tokens, cost, latency per purpose and per run / article | — |

### Web UI

```bash
locassess ui          # opens the Streamlit app in the browser
```

Tabs: **Profile** (overall and factor scores, red flags, 26 criteria by factor with rating colour,
status and trend, drill-down criterion → sub-criterion → evidence sentence → article link),
**Compare locations** (ranking and side-by-side table), **Report**, **Selection log**,
**Run pipeline** (build → freeze → extract → score from the browser) and **Export** (Markdown,
PDF, CSV). The 26 weight sliders in the sidebar recalculate the overall score, factor scores and
ranking live (SPEC §9.7; no model calls).

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
locassess/providers/     LLM, search and decision-model adapters; call_log wrapper
locassess/prompts.py     versioned prompt loader (prompts/*.md)
locassess/stage0_corpus.py Stage 0 search, fetch, rule filter, selection, freeze, selection log
locassess/query_definition.py  versioned Query A/B questions built from the codebook
locassess/stage1_extract.py  Stage 1 extraction rules and runs
locassess/stage2_score.py  Stage 2 scoring (pure functions, SPEC §9)
locassess/stage3_report.py Stage 3 profile JSON, report validation and retries
locassess/pipeline.py    stage services shared by CLI and UI
locassess/ui_data.py     data functions behind the UI (tables, sliders, ranking, exports)
locassess/baseline.py    baseline (direct LLM assessment)
locassess/annotation.py  gold-standard template export / import
locassess/evaluation/    metric functions (metrics.py) and runner (run_eval.py)
locassess/cli.py         command-line interface
ui/app.py                Streamlit app
```

## Note on article texts

Full article texts are stored only locally for research (text and data mining). Do not redistribute them.
