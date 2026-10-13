# SPEC — News-Based Location Assessment (LocAssess)

Version 1.0 · 2026-10-13 · Author: Hannes (Master's thesis, Universität Hamburg)

This document is the build specification for Claude Code. It describes **what** to build, **how each stage must behave**, the **data model**, the **formulas**, and the **milestones with acceptance criteria**. Where this spec says MUST, the behaviour is part of the research design and must not be changed without asking. Where it says SHOULD or "default", it is a sensible default that can be adjusted in `config.yaml`.

---

## 1. Purpose

LocAssess assesses a candidate **production location** (a city or region anywhere in the world) using **news articles**. It:

1. searches the web for ~15 relevant articles from reputable outlets and freezes them as a corpus,
2. extracts structured evidence per article with a **decision model** (typed answers + probabilities),
3. turns the evidence into a location profile with **deterministic, rule-based scoring** (no LLM),
4. lets an LLM write an explanatory report that is **automatically checked** against the profile,
5. stores everything so that a **scientific evaluation** (model comparison, baseline comparison, robustness) can be run with SQL.

### Core principle (MUST)

**Models suggest and interpret; fixed rules filter and decide.**

- The only place where a judgement about the location is made is Stage 2, and Stage 2 contains no model calls.
- After Stage 0, every stage reads only the **frozen corpus**. A web search is never repeated inside a run.
- Every number shown to the user must be traceable to evidence → sentence → article → URL.

### Research question (context for design decisions)

> How can a hybrid architecture combining model-based information extraction with rule-based evaluation transform news articles into a traceable assessment of production locations, and how does it perform compared to a direct LLM-based assessment?

- SQ1: How accurately can a model extract location-relevant evidence from news articles and assign it to predefined criteria?
- SQ2: How can the extracted evidence be aggregated into criterion-level assessments using transparent, rule-based logic?
- SQ3: How does the hybrid approach compare to a direct LLM assessment in terms of consistency, traceability and plausibility?

### Language (MUST)

The program, code, prompts, codebook, UI, reports and database content are **English**. Articles are searched in **English and in the local language of the location**; non-English articles are processed as they are (the models receive the original text; prompts and answer options are English).

---

## 2. Pipeline overview

| Stage | Task | Executed by | Output |
| --- | --- | --- | --- |
| 0 | Generate queries, search, fetch, filter, select, freeze | LLM (queries, relevance) + rules (filters, selection) | Frozen corpus (~15 articles) |
| 1 | Classify each article against the criteria | Decision model + rules | Evidence records |
| 2 | Score criteria, trends, red flags, optional overall score | Pure Python rules | Location profile |
| 3 | Write and validate report; web UI | LLM (text) + rules (validation) + Streamlit | Report with evidence references |
| E | Evaluation and baseline | Python + LLM (baseline only) | Metrics tables (CSV) |

```
config + criteria ──► [0] corpus ──(frozen)──► [1] evidence ──► [2] profile ──► [3] report + UI
                                                   ▲                                │
                                    other models / repetitions          [E] evaluation & baseline
```

---

## 3. Tech stack (defaults)

| Component | Choice |
| --- | --- |
| Language | Python ≥ 3.11 |
| Packaging | `pyproject.toml` (uv or pip) |
| Database | SQLite (single file `data/locassess.db`), schema in `locassess/db/schema.sql` |
| Validation / models | `pydantic` v2, `pydantic-settings` |
| Config | `config.yaml` + `.env` (secrets) via `python-dotenv` |
| Article extraction | `trafilatura` |
| Sentence splitting | `pysbd` (multilingual) |
| LLM SDKs | `openai` (default); `anthropic` optional |
| HTTP | `httpx` |
| UI | Streamlit |
| CLI | `typer` |
| Tests | `pytest` |
| Analysis | `pandas`, `numpy`, `scikit-learn` (metrics), `scipy` |

No web framework, no auth, no deployment. Runs locally.

---

## 4. Repository layout

```
locassess/
  pyproject.toml
  config.example.yaml
  .env.example
  README.md
  data/
    criteria.csv            # provided: 6 factors, 26 main criteria, 94 sub-criteria
    codebook.yaml           # generated draft, then reviewed by the author
    whitelist.csv           # provided by the author: allowed news domains
    locassess.db            # created at runtime (git-ignored)
  prompts/                  # versioned prompt templates (Markdown with a version header)
    query_generation.md
    relevance.md
    llm_structured_extraction.md
    report.md
    baseline.md
  locassess/
    config.py
    criteria.py             # loads criteria.csv + codebook.yaml
    db/
      schema.sql
      repo.py               # thin data-access layer (sqlite3)
    providers/
      llm.py                # LLM interface + OpenAI/Anthropic adapters
      search.py             # search interface + adapters
      decision.py           # decision-model interface + adapters
    stage0_corpus.py
    stage1_extract.py
    stage2_score.py
    stage3_report.py
    baseline.py
    evaluation/
      metrics.py
      run_eval.py
    cli.py
  ui/
    app.py                  # Streamlit entry point
  tests/
    test_stage2_worked_example.py
    test_rules_*.py
```

---

## 5. Inputs

### 5.1 Criteria framework (`data/criteria.csv`, provided)

Source: Hoang-Thi et al. (2026), *Sustainability* 18, 8227, doi:10.3390/su18168227.

Three levels, each with a fixed role:

| Level | Count | Role in the program |
| --- | --- | --- |
| Factor | 6 | Grouping in UI and report; optional factor sub-score |
| Main criterion (`H01`–`H26`) | 26 | **Unit of assessment**: Query A, score, rating, trend, red flags, weight slider, evaluation unit |
| Sub-criterion (`H01.01` …) | 94 | Defines the main criterion in the codebook, seeds search terms, extra field in Query B to show which aspect is evidenced. **Never scored on its own.** |

Four main criteria have no sub-criteria (H05, H06, H07, H08; empty `sub_id` in the CSV); for those the sub-criterion field in Query B is skipped.

CSV columns: `factor_en, factor_de, main_id, main_criterion_en, main_criterion_de, sub_id, sub_criterion_en, sub_criterion_de, paper_sources, paper_table`. Use only the `_en` columns in the program; `_de` columns may be shown as an optional tooltip.

Main criteria:

| ID | Factor | Main criterion |
| --- | --- | --- |
| H01 | Physical | Natural resources |
| H02 | Physical | Transportation system |
| H03 | Physical | Distance |
| H04 | Physical | Spatial relation |
| H05 | Physical | Urban agglomeration |
| H06 | Environmental | Environmental pollution |
| H07 | Environmental | Requirements for environmental protection |
| H08 | Environmental | Climate change |
| H09 | Political | Tax |
| H10 | Political | Local and national government legislation/Policy |
| H11 | Political | Political status |
| H12 | Economic | Cost/expenditure |
| H13 | Economic | Economic environment |
| H14 | Economic | Marketing |
| H15 | Economic | Capital |
| H16 | Economic | Market |
| H17 | Economic | External economies |
| H18 | Economic | Business related service |
| H19 | Economic | Economic agglomeration |
| H20 | Social | Population potential |
| H21 | Social | Chance and purely personal factors |
| H22 | Social | Quality of life |
| H23 | Social | Education |
| H24 | Public health | Epidemiology |
| H25 | Public health | Health system |
| H26 | Public health | Public Health Surveillance |

### 5.2 Codebook (`data/codebook.yaml`)

One entry per main criterion. It is used **both** as model instruction (Stage 1) **and** as the annotation guideline for the manual gold standard. Claude Code generates a **draft** from `criteria.csv` (marked `status: draft`); the author reviews and edits it. Structure:

```yaml
version: "0.1"
direction_definition: "+1 = makes the location more attractive for a production site; -1 = less attractive; 0 = mentioned without a clear direction"
intensity_definition:
  1: "minor or local effect, single event"
  2: "noticeable effect on companies in the region"
  3: "major effect that would change a location decision"
evidence_type_definition:
  fact: "has happened / is in force"
  announcement: "planned, announced, decided but not yet implemented"
  opinion: "assessment, forecast, criticism, expectation"
criteria:
  H02:
    status: draft
    name: "Transportation system"
    definition: "Availability and quality of transport infrastructure at the location: railway, highway, seaport, airport."
    sub_criteria: ["H02.01 Railway", "H02.02 Highway", "H02.03 Seaport", "H02.04 Airport"]
    positive_examples: ["New motorway junction opened next to the industrial park."]
    negative_examples: ["Rail freight terminal closed for two years."]
    search_terms: ["railway", "freight rail", "motorway", "port", "airport", "logistics hub"]
```

### 5.3 Configuration (`config.yaml`)

```yaml
location:
  name: "Leipzig"
  region: "Saxony"
  country: "Germany"
  local_language: "de"
  aliases: ["Leipzig", "Region Leipzig"]
period:
  years_back: 5                # hard filter in Stage 0
  run_date: null               # null = today; fixed date allowed for reproducibility
corpus:
  target_articles: 15
  max_per_year: 4
  min_words: 200
  queries_per_criterion: 2     # 2–3
  hits_per_query: 10
  min_relevance: 1             # 0–2 scale
whitelist_file: "data/whitelist.csv"
models:
  llm_provider: "openai"       # openai | anthropic
  llm_model: "<set in .env or here>"
  temperature: 0
  decision_provider: "llm_structured"   # jev | openai_decisions | llm_structured
  llm_confidence_method: "logprobs"     # logprobs | sampling
  sampling_n: 5
  max_options_per_question: 255         # smallest option limit of all compared models
search_provider: "openai_web_search"    # openai_web_search | brave | manual_csv
scoring:
  type_weights: {fact: 1.0, announcement: 0.6, opinion: 0.3}
  half_life_months: 24
  rating_thresholds: {green: 1.0, red: -1.0}
  trend_threshold: 1.0
  status: {sufficient_min_evidence: 3, sufficient_min_articles: 2}
  extraction: {affected_threshold: 0.5, review_threshold: 0.6, second_sentence_gap: 0.2}
  red_flags:
    recent_months: 12
    knockout_criteria: []      # e.g. ["H01", "H20"]; author decides
weights_default: 1             # slider default for every main criterion (0–10)
report:
  max_attempts: 3              # first try + 2 retries
```

### 5.4 Secrets (`.env`, never committed)

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=        # optional
TYPESAFE_API_KEY=         # Jev, if early access is granted
BRAVE_API_KEY=            # only if search_provider = brave
```

### 5.5 Whitelist (`data/whitelist.csv`)

Columns: `domain, outlet_name, media_type (national|regional|trade|agency), language, country`. Maintained by the author. Must be extensible for any country.

---

## 6. Provider interfaces

All external services sit behind small interfaces so that models can be swapped and compared. **Do not invent API endpoints or request formats.** Before implementing an adapter, read the provider's current official documentation; if something is unclear or access is missing, implement the adapter skeleton, raise a clear `ProviderNotConfigured` error, and tell the author.

### 6.1 LLM (`providers/llm.py`)

```python
class LLM(Protocol):
    def complete(self, prompt_id: str, variables: dict, json_schema: dict | None = None,
                 temperature: float = 0, n: int = 1, logprobs: bool = False) -> LLMResult: ...
```

Used for: query generation (Stage 0), relevance rating (Stage 0), report (Stage 3), baseline (E), and the `llm_structured` decision adapter. Adapters: OpenAI (default), Anthropic (optional). Every call is written to `call_log`.

### 6.2 Search (`providers/search.py`)

```python
class SearchProvider(Protocol):
    def search(self, query: str, allowed_domains: list[str], date_from: date, date_to: date,
               max_results: int) -> list[SearchHit]: ...   # SearchHit: url, title, snippet, rank, published_at|None
```

Adapters:
- `openai_web_search` — the LLM provider's web search tool, with domain filtering if the current API supports it.
- `brave` — Brave Search API, domains via `site:` operators.
- `manual_csv` — imports a hand-picked list of URLs (`url, note`). Required: the thesis may use a hand-selected set of articles; also the path for importing articles from another database later.

Search-level domain filtering is only an optimisation. The whitelist is **always enforced again** by the rule filter in Stage 0.4.

### 6.3 Decision model (`providers/decision.py`)

A decision model receives a state and a set of questions with fixed answer options, and returns a probability for every option.

```python
class DecisionModel(Protocol):
    name: str
    def decide(self, state: dict, questions: dict[str, list[str]]) -> dict[str, dict[str, float]]:
        """questions: {question_key: [option, ...]} → {question_key: {option: probability}}"""
```

Adapters:

| Adapter | Description |
| --- | --- |
| `jev` | TypeSafe Jev (System One model). Docs: https://docs.typesafe.ai/ · Early access via https://console.typesafe.ai/ |
| `openai_decisions` | OpenAI Decisions API on GPT-6 Luna (announced 2026-09-29, limited preview at time of writing). Use current OpenAI docs. |
| `llm_structured` | **Fallback that must work end to end.** Any chat LLM with JSON-schema structured output; options become enums. Probabilities via token logprobs if available, else `sampling_n` samples at temperature 1 → relative frequency. The method used is stored per extraction run. |

All three adapters MUST receive exactly the same questions, options and codebook text (one versioned query definition), so their results are comparable.

---

## 7. Stage 0 — Web search and corpus

Output: one frozen `corpus` with ~15 `article` rows marked `selected = 1`, plus a full selection log.

1. **Generate queries (LLM, `prompts/query_generation.md`).** Per main criterion, `queries_per_criterion` queries in English and in the local language, built from location name/aliases and the codebook `search_terms` (derived from sub-criteria). Store in `search_query`.
2. **Search (SearchProvider).** Run every query with whitelist domains and the period. Store the top `hits_per_query` hits with rank in `search_hit`.
3. **Fetch articles.** Download, extract full text with trafilatura; store title, publication date (from the article, not the search hit), URL, fetch timestamp, text, SHA-256 of the text. Paywall/empty text/fetch error → excluded with reason.
4. **Rule filter (MUST).** An article passes only if all hold:
   - domain is on the whitelist;
   - publication date within the period;
   - location name or an alias appears in the text (case-insensitive);
   - word count ≥ `min_words`;
   - not a duplicate (same URL, or text similarity ≥ 0.9 by e.g. shingled Jaccard — wire-agency copies).
   Every exclusion is stored with its reason.
5. **Relevance (LLM, suggestion only, `prompts/relevance.md`).** Rating 0–2 of how strongly the article concerns the location with respect to the criteria, plus a one-sentence justification.
6. **Selection (rules, MUST).** Keep relevance ≥ `min_relevance`; at most `max_per_year` per calendar year; fill criteria that have no article yet first (based on which criterion's query found the article); ties → higher relevance, then newer date; stop at `target_articles`.
7. **Freeze.** Mark the corpus `frozen = 1`. Afterwards the corpus and its articles are read-only (enforce in `repo.py`).
8. **Selection log.** PRISMA-style counts: hits → fetched → passed filter → relevant → selected, with exclusion reasons; exportable as CSV and shown in the UI.

Note for the thesis: full texts are stored only locally for research (text-and-data-mining); never redistribute them.

---

## 8. Stage 1 — Evidence extraction

Runs on a frozen corpus. One **extraction run** = one decision model × one repetition number.

1. **Prepare article.** Split into sentences with pysbd; store `sentence` rows (`S1`, `S2`, … per article). Articles with more sentences than `max_options_per_question` are split into sections so that every model sees identical sections. The publication date is the event date (`YYYY-MM`).
2. **Query A — which criteria are affected?** One call per article (section). State: location, article text with sentence IDs, codebook definitions. Questions: `{H01_affected: [yes, no], …, H26_affected: [yes, no]}`. Criteria with `P(yes) ≥ affected_threshold` go to Query B.
3. **Query B — details per affected criterion.** One call per affected criterion. Questions:

| Question | Options |
| --- | --- |
| `direction` | `-1`, `0`, `1` |
| `intensity` | `1`, `2`, `3` |
| `evidence_type` | `fact`, `announcement`, `opinion` |
| `sub_criterion` | the criterion's sub-criterion IDs + `overarching` (skip if the criterion has none) |
| `evidence_sentence` | sentence IDs of the article (section) |

4. **Rules (MUST).**
   - Each field takes its most probable option.
   - `confidence = min(P(affected=yes), P(direction), P(intensity))` — conservative: evidence is only as certain as its weakest core field.
   - `confidence < review_threshold` → `needs_review = 1`.
   - If the second-best sentence is within `second_sentence_gap` of the best, store both sentences.
5. **Store.** Every raw model response with the **full probability distribution of every question** (also for criteria not affected) → `model_response`. Every derived evidence → `evidence`. One evidence at most per article × main criterion (per section; merge sections by taking the higher-confidence result).

Comparison unit for evaluation: **article × main criterion**.

Expected volume: ~15 articles × (1 Query A + ~3 Query B) ≈ 60 calls per location.

---

## 9. Stage 2 — Rule-based scoring (no model calls)

Input: evidence of one extraction run. Output: one **scoring run** (a parameter set) with criterion scores, red flags, overall scores. Same input + same parameters → identical output (MUST, covered by tests).

### 9.1 Evidence value

```
e_i      = r_i · s_i · w_type · c_i · w_time
w_time   = 0.5 ^ (age_months / half_life_months)
```

`r` direction (−1/0/+1), `s` intensity (1–3), `c` confidence, `w_type` from `type_weights`. `age_months` = whole months between event date and run date.

### 9.2 Criterion score (range −3 … +3)

```
S_k = Σ e_i / Σ (w_type · c_i · w_time)      over all evidence of criterion k
```

### 9.3 Evidence status

- no evidence → `no_statement` (never treated as neutral, no score)
- 1–2 evidence or only one article → `thin`
- ≥ 3 evidence from ≥ 2 articles → `sufficient`

### 9.4 Rating

`S ≥ +1` → `green`, `S ≤ −1` → `red`, else `yellow`. `thin` is shown visibly.

### 9.5 Trend

Split the period at its midpoint. Compute the score of each half **without** `w_time` (otherwise the older half is artificially discounted). Difference ≥ +1 → `improving`, ≤ −1 → `deteriorating`, else `stable`. Only if both halves have evidence, else `n/a`.

### 9.6 Red flags (independent of the score)

- R1: a `fact` with direction −1 and intensity 3 within the last `recent_months`.
- R2: ≥ 2 negative evidence with intensity ≥ 2 for the same criterion from different articles/outlets.
- R3: a criterion listed in `knockout_criteria` with `S ≤ −1`.

### 9.7 Overall score and weight sliders

Each of the 26 main criteria has a slider `g_k ∈ {0, …, 10}`, default `weights_default` (1). Let `E` be the set of criteria with evidence:

```
w_k = g_k / Σ_{j∈E} g_j        G = Σ_{k∈E} w_k · S_k
```

All sliders at 1 = equal weighting. A slider at 0 excludes a criterion. If all sliders in `E` are 0 → no overall score. Optional **factor sub-score**: same formula restricted to the criteria of one factor.

### 9.8 Worked example (MUST be a unit test)

Run date 2026-10, period 2021-10 – 2026-10, criterion **H20 Population potential**, default parameters:

| Evidence | Content | r | s | type | c | event date |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | 30 % of apprenticeships unfilled | −1 | 2 | fact | 0.62 | 2025-03 |
| E2 | Plant cuts production due to staff shortage | −1 | 3 | fact | 0.85 | 2026-06 |
| E3 | State programme to recruit skilled workers announced | +1 | 2 | announcement | 0.80 | 2022-11 |

Expected: `w_time` ≈ 0.58 / 0.89 / 0.26; weights ≈ 0.36 / 0.76 / 0.12; `S_H20 = −2.212` (shown as −2.2) → `red`; status `sufficient`; trend: older half +2.000, newer half −2.578 → `deteriorating`; red flags R1 (E2) and R2 (E1+E2).

Overall example: with H10 = +2.6 and H01 = +0.4 (thin) and no other evidence, equal weights give `G = +0.263`; sliders H20 = 2, H10 = 1, H01 = 1 give `G = −0.356`. Assert with a tolerance of 0.005 on these unrounded values.

---

## 10. Stage 3 — Report and UI

### 10.1 Report (LLM, `prompts/report.md`)

- Input: **only** the profile JSON (criteria with score, rating, status, trend, evidence IDs; red flags; for each evidence the sentence text, outlet and date). Never raw articles.
- Fixed structure: Summary (max. 3 sentences) · Red flags · Strengths · Weaknesses · Thinly evidenced criteria · No evidence.
- Every content statement ends with evidence IDs in square brackets, e.g. `[E1, E2]`.
- No new judgements; ratings, trends and red flags are taken over exactly; criteria with `no_statement` are only listed under "No evidence", without speculation.

### 10.2 Report validation (rules, MUST)

| Check | Method |
| --- | --- |
| All cited IDs exist | regex `\[E\d+(, E\d+)*\]`, compare with profile |
| Every paragraph in Red flags / Strengths / Weaknesses cites ≥ 1 ID | per paragraph |
| Stated ratings match | `(red)`, `(green)`, `(yellow)` next to a criterion name vs profile |
| All red flags appear | each red-flag evidence ID appears under Red flags |
| "No evidence" complete | every `no_statement` criterion is listed |

Failure → regenerate with the list of violations in the prompt, up to `max_attempts`. Still failing → store as failed; UI shows the profile without report text. Store attempts and per-rule results.

### 10.3 Streamlit UI (`ui/app.py`)

- Input: location (name, region, country, local language, aliases), period; start a run with per-stage progress.
- Profile overview: 26 main criteria grouped by the 6 factors, each with rating colour, score, evidence status, trend.
- Red flags highlighted at the top.
- Drill-down: main criterion → sub-criterion → evidence → evidence sentence → original article (link).
- 26 sliders (0–10, default 1), grouped by factor, live recalculation of overall score, factor sub-scores and ranking.
- Side-by-side comparison of several locations.
- Selection log of Stage 0; report; export as Markdown/PDF and CSV.

---

## 11. Baseline (for SQ3)

`baseline.py`: give an LLM the same frozen corpus (full texts) and the criteria list and ask it **directly** for a score −3 … +3, rating and short justification per main criterion plus an overall judgement (same scale as Stage 2). Repeat 3–5 times. Store in `baseline_assessment`. Prompt in `prompts/baseline.md`.

---

## 12. Data model (SQLite)

`locassess/db/schema.sql` MUST create at least the following tables (names may get `_id` suffixes, types may be refined). All timestamps ISO-8601 UTC. JSON columns as TEXT.

```sql
CREATE TABLE criterion (            -- loaded from criteria.csv
  main_id TEXT PRIMARY KEY, factor TEXT NOT NULL, name TEXT NOT NULL, name_de TEXT);
CREATE TABLE sub_criterion (
  sub_id TEXT PRIMARY KEY, main_id TEXT NOT NULL REFERENCES criterion, name TEXT NOT NULL, name_de TEXT);

CREATE TABLE corpus (
  corpus_id INTEGER PRIMARY KEY, location_json TEXT NOT NULL, period_from TEXT, period_to TEXT,
  created_at TEXT NOT NULL, config_hash TEXT NOT NULL, frozen INTEGER NOT NULL DEFAULT 0, note TEXT);
CREATE TABLE search_query (
  query_id INTEGER PRIMARY KEY, corpus_id INTEGER REFERENCES corpus, main_id TEXT, query_text TEXT,
  language TEXT, hit_count INTEGER, prompt_version TEXT);
CREATE TABLE search_hit (
  hit_id INTEGER PRIMARY KEY, query_id INTEGER REFERENCES search_query, url TEXT, title TEXT,
  rank INTEGER, status TEXT, exclusion_reason TEXT);
CREATE TABLE article (
  article_id INTEGER PRIMARY KEY, corpus_id INTEGER REFERENCES corpus, url TEXT, domain TEXT,
  outlet TEXT, title TEXT, published_at TEXT, fetched_at TEXT, language TEXT, text TEXT,
  text_sha256 TEXT, word_count INTEGER, relevance INTEGER, relevance_reason TEXT,
  selected INTEGER DEFAULT 0, exclusion_reason TEXT);
CREATE TABLE sentence (
  article_id INTEGER REFERENCES article, sentence_key TEXT, section INTEGER, text TEXT,
  PRIMARY KEY (article_id, sentence_key));

CREATE TABLE annotation (           -- manual gold standard, incl. "not affected"
  article_id INTEGER REFERENCES article, main_id TEXT REFERENCES criterion, annotator TEXT,
  affected INTEGER, direction INTEGER, intensity INTEGER, evidence_type TEXT, sub_id TEXT,
  sentence_keys TEXT, PRIMARY KEY (article_id, main_id, annotator));

CREATE TABLE extraction_run (
  extraction_run_id INTEGER PRIMARY KEY, corpus_id INTEGER REFERENCES corpus, model TEXT,
  model_version TEXT, query_definition_version TEXT, codebook_version TEXT, repetition INTEGER,
  confidence_method TEXT, created_at TEXT);
CREATE TABLE model_response (
  response_id INTEGER PRIMARY KEY, extraction_run_id INTEGER REFERENCES extraction_run,
  article_id INTEGER, section INTEGER, query_type TEXT CHECK (query_type IN ('A','B')),
  main_id TEXT, distributions_json TEXT NOT NULL, call_id INTEGER REFERENCES call_log);
CREATE TABLE evidence (
  evidence_id INTEGER PRIMARY KEY, extraction_run_id INTEGER REFERENCES extraction_run,
  article_id INTEGER, main_id TEXT, sub_id TEXT, direction INTEGER, intensity INTEGER,
  evidence_type TEXT, event_date TEXT, confidence REAL, sentence_keys TEXT, sentence_text TEXT,
  needs_review INTEGER DEFAULT 0, UNIQUE (extraction_run_id, article_id, main_id));

CREATE TABLE scoring_run (
  scoring_run_id INTEGER PRIMARY KEY, extraction_run_id INTEGER REFERENCES extraction_run,
  params_json TEXT NOT NULL, label TEXT, created_at TEXT);
CREATE TABLE criterion_score (
  scoring_run_id INTEGER REFERENCES scoring_run, main_id TEXT, score REAL, rating TEXT,
  status TEXT, trend TEXT, red_flags_json TEXT, evidence_ids_json TEXT,
  PRIMARY KEY (scoring_run_id, main_id));
CREATE TABLE overall_score (
  overall_id INTEGER PRIMARY KEY, scoring_run_id INTEGER REFERENCES scoring_run, variant TEXT,
  sliders_json TEXT, weights_json TEXT, score REAL, factor_scores_json TEXT);
CREATE TABLE report (
  report_id INTEGER PRIMARY KEY, scoring_run_id INTEGER REFERENCES scoring_run, text TEXT,
  attempts INTEGER, passed INTEGER, checks_json TEXT, prompt_version TEXT);

CREATE TABLE baseline_assessment (
  baseline_id INTEGER PRIMARY KEY, corpus_id INTEGER REFERENCES corpus, model TEXT,
  repetition INTEGER, per_criterion_json TEXT, overall_json TEXT, call_id INTEGER);

CREATE TABLE call_log (             -- every model / search call
  call_id INTEGER PRIMARY KEY, provider TEXT, model TEXT, model_version TEXT, purpose TEXT,
  prompt_version TEXT, request_json TEXT, response_json TEXT, input_tokens INTEGER,
  output_tokens INTEGER, cost_usd REAL, latency_ms INTEGER, created_at TEXT, error TEXT);
```

Read-only rule: once `corpus.frozen = 1`, no INSERT/UPDATE/DELETE on that corpus's `article` or `sentence` rows (enforce in `repo.py`; a SQLite trigger is welcome).

---

## 13. Evaluation (`locassess/evaluation/`)

Each metric is a function plus a CLI command that writes a CSV to `results/`.

| Check | Procedure | Metric | SQ |
| --- | --- | --- | --- |
| Search reproducibility | Stage 0 twice for one location on different days (two corpora) | Jaccard overlap of selected URLs | justifies freezing |
| Extraction quality | Compare evidence with `annotation` per article × main criterion (incl. not affected) | Precision, recall, F1 of "affected"; agreement on direction; Cohen's kappa | SQ1 |
| Hallucination | Share of evidence sentences that do not exist in the article | % | SQ1 |
| Extraction stability | 3–5 repetitions per model on the same corpus | Agreement between runs (e.g. Fleiss' kappa / % identical cells) | SQ1 |
| Threshold-free model comparison | Use stored distributions | Brier score, reliability diagram data, PR-AUC | SQ1 |
| Scoring robustness | Vary type weights, half-life, thresholds; draw random sliders (N = 1000) | Stability of ratings and of location ranking (Spearman) | SQ2 |
| Baseline comparison | Hybrid vs. `baseline_assessment` over repetitions | Spread of scores per criterion; share of statements backed by evidence | SQ3 |
| Report reliability | From `report` | Attempts needed, failure rate per rule | SQ3 |
| Cost & latency | From `call_log` | Cost and time per article / location per model | — |

Model comparison in Stage 1: Jev vs. OpenAI Decisions vs. `llm_structured` on the same annotated corpus. Report precision/recall aggregated over all criteria; per-criterion results only descriptively (few positive cases per location).

---

## 14. CLI

```
locassess init-db
locassess load-criteria
locassess codebook-draft                     # writes data/codebook.yaml (status: draft)
locassess corpus build --config config.yaml  # Stage 0, prints corpus_id
locassess corpus import --csv urls.csv       # manual_csv path
locassess corpus freeze --corpus 1
locassess extract --corpus 1 --model llm_structured --repeat 3
locassess score --extraction-run 4 [--params params.yaml]
locassess report --scoring-run 7
locassess baseline --corpus 1 --repeat 3
locassess evaluate --all
locassess ui                                  # streamlit run ui/app.py
```

---

## 15. Engineering rules

- All prompts live in `prompts/` with a version header; the version is stored with every call.
- Temperature 0 everywhere except the `sampling` confidence method.
- Every external call goes through one wrapper that writes `call_log`, retries with backoff on rate limits, and never logs API keys.
- Stage 2 is pure functions over plain data classes; no DB or network access inside the formulas.
- Tests: the worked example (9.8), every rule in 7.4/7.6/8.4/9.6/10.2, and determinism of Stage 2. Model calls are mocked in tests.
- Small, reviewable commits per milestone. Update `README.md` with setup and run instructions.

---

## 16. Milestones (build in this order)

| # | Milestone | Acceptance criteria |
| --- | --- | --- |
| M1 | Skeleton | Repo layout, `pyproject.toml`, config loading, `init-db`, `load-criteria` (26 main / 94 sub), `call_log` wrapper, tests run green |
| M2 | Stage 2 first | Scoring module with all rules; worked example test passes; deterministic |
| M3 | Stage 0 | All search adapters incl. `manual_csv`; fetch, filter, selection, freeze, selection log CSV; frozen corpus is read-only |
| M4 | Stage 1 | `llm_structured` adapter end to end (Query A/B, rules, distributions stored); `jev` and `openai_decisions` adapters per official docs or clean `ProviderNotConfigured` |
| M5 | Codebook draft | `codebook-draft` command; author reviews the YAML |
| M6 | Stage 3 | Report generation + validation + retries |
| M7 | UI | Streamlit app as in 10.3 |
| M8 | Baseline & evaluation | All metrics in 13 as CSV outputs |

Stop after each milestone, summarise what was built, and wait for the author's go-ahead.

---

## 17. Out of scope

AHP/TOPSIS weighting, user accounts, cloud deployment, scheduling, redistribution of article texts.

## 18. Open points (ask the author, do not decide alone)

- Access to Jev and to the OpenAI Decisions API (early access / limited preview).
- Contents of `whitelist.csv` and of `knockout_criteria`.
- Final codebook texts (the draft is only a starting point).
- Which 5 locations are evaluated and for which period.
