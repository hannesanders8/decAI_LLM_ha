-- LocAssess schema (SPEC §12). All timestamps ISO-8601 UTC; JSON columns as TEXT.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS criterion (            -- loaded from criteria.csv
  main_id TEXT PRIMARY KEY, factor TEXT NOT NULL, name TEXT NOT NULL, name_de TEXT);
CREATE TABLE IF NOT EXISTS sub_criterion (
  sub_id TEXT PRIMARY KEY, main_id TEXT NOT NULL REFERENCES criterion, name TEXT NOT NULL, name_de TEXT);

CREATE TABLE IF NOT EXISTS corpus (
  corpus_id INTEGER PRIMARY KEY, location_json TEXT NOT NULL, period_from TEXT, period_to TEXT,
  created_at TEXT NOT NULL, config_hash TEXT NOT NULL, frozen INTEGER NOT NULL DEFAULT 0, note TEXT);
CREATE TABLE IF NOT EXISTS search_query (
  query_id INTEGER PRIMARY KEY, corpus_id INTEGER REFERENCES corpus, main_id TEXT, query_text TEXT,
  language TEXT, hit_count INTEGER, prompt_version TEXT);
CREATE TABLE IF NOT EXISTS search_hit (
  hit_id INTEGER PRIMARY KEY, query_id INTEGER REFERENCES search_query, url TEXT, title TEXT,
  rank INTEGER, status TEXT, exclusion_reason TEXT);
CREATE TABLE IF NOT EXISTS article (
  article_id INTEGER PRIMARY KEY, corpus_id INTEGER REFERENCES corpus, url TEXT, domain TEXT,
  outlet TEXT, title TEXT, published_at TEXT, fetched_at TEXT, language TEXT, text TEXT,
  text_sha256 TEXT, word_count INTEGER, relevance INTEGER, relevance_reason TEXT,
  selected INTEGER DEFAULT 0, exclusion_reason TEXT);
CREATE TABLE IF NOT EXISTS sentence (
  article_id INTEGER REFERENCES article, sentence_key TEXT, section INTEGER, text TEXT,
  PRIMARY KEY (article_id, sentence_key));

CREATE TABLE IF NOT EXISTS annotation (           -- manual gold standard, incl. "not affected"
  article_id INTEGER REFERENCES article, main_id TEXT REFERENCES criterion, annotator TEXT,
  affected INTEGER, direction INTEGER, intensity INTEGER, evidence_type TEXT, sub_id TEXT,
  sentence_keys TEXT, PRIMARY KEY (article_id, main_id, annotator));

CREATE TABLE IF NOT EXISTS extraction_run (
  extraction_run_id INTEGER PRIMARY KEY, corpus_id INTEGER REFERENCES corpus, model TEXT,
  model_version TEXT, query_definition_version TEXT, codebook_version TEXT, repetition INTEGER,
  confidence_method TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS model_response (
  response_id INTEGER PRIMARY KEY, extraction_run_id INTEGER REFERENCES extraction_run,
  article_id INTEGER, section INTEGER, query_type TEXT CHECK (query_type IN ('A','B')),
  main_id TEXT, distributions_json TEXT NOT NULL, call_id INTEGER REFERENCES call_log);
CREATE TABLE IF NOT EXISTS evidence (
  evidence_id INTEGER PRIMARY KEY, extraction_run_id INTEGER REFERENCES extraction_run,
  article_id INTEGER, main_id TEXT, sub_id TEXT, direction INTEGER, intensity INTEGER,
  evidence_type TEXT, event_date TEXT, confidence REAL, sentence_keys TEXT, sentence_text TEXT,
  needs_review INTEGER DEFAULT 0, UNIQUE (extraction_run_id, article_id, main_id));

CREATE TABLE IF NOT EXISTS scoring_run (
  scoring_run_id INTEGER PRIMARY KEY, extraction_run_id INTEGER REFERENCES extraction_run,
  params_json TEXT NOT NULL, label TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS criterion_score (
  scoring_run_id INTEGER REFERENCES scoring_run, main_id TEXT, score REAL, rating TEXT,
  status TEXT, trend TEXT, red_flags_json TEXT, evidence_ids_json TEXT,
  PRIMARY KEY (scoring_run_id, main_id));
CREATE TABLE IF NOT EXISTS overall_score (
  overall_id INTEGER PRIMARY KEY, scoring_run_id INTEGER REFERENCES scoring_run, variant TEXT,
  sliders_json TEXT, weights_json TEXT, score REAL, factor_scores_json TEXT);
CREATE TABLE IF NOT EXISTS report (
  report_id INTEGER PRIMARY KEY, scoring_run_id INTEGER REFERENCES scoring_run, text TEXT,
  attempts INTEGER, passed INTEGER, checks_json TEXT, prompt_version TEXT);

CREATE TABLE IF NOT EXISTS baseline_assessment (
  baseline_id INTEGER PRIMARY KEY, corpus_id INTEGER REFERENCES corpus, model TEXT,
  repetition INTEGER, per_criterion_json TEXT, overall_json TEXT, call_id INTEGER);

CREATE TABLE IF NOT EXISTS call_log (             -- every model / search call
  call_id INTEGER PRIMARY KEY, provider TEXT, model TEXT, model_version TEXT, purpose TEXT,
  prompt_version TEXT, request_json TEXT, response_json TEXT, input_tokens INTEGER,
  output_tokens INTEGER, cost_usd REAL, latency_ms INTEGER, created_at TEXT, error TEXT);

-- Read-only rule: once corpus.frozen = 1, its article and sentence rows cannot change.
CREATE TRIGGER IF NOT EXISTS article_frozen_insert BEFORE INSERT ON article
WHEN (SELECT frozen FROM corpus WHERE corpus_id = NEW.corpus_id) = 1
BEGIN SELECT RAISE(ABORT, 'corpus is frozen: article is read-only'); END;

CREATE TRIGGER IF NOT EXISTS article_frozen_update BEFORE UPDATE ON article
WHEN (SELECT frozen FROM corpus WHERE corpus_id = OLD.corpus_id) = 1
  OR (SELECT frozen FROM corpus WHERE corpus_id = NEW.corpus_id) = 1
BEGIN SELECT RAISE(ABORT, 'corpus is frozen: article is read-only'); END;

CREATE TRIGGER IF NOT EXISTS article_frozen_delete BEFORE DELETE ON article
WHEN (SELECT frozen FROM corpus WHERE corpus_id = OLD.corpus_id) = 1
BEGIN SELECT RAISE(ABORT, 'corpus is frozen: article is read-only'); END;

CREATE TRIGGER IF NOT EXISTS sentence_frozen_insert BEFORE INSERT ON sentence
WHEN (SELECT c.frozen FROM article a JOIN corpus c USING (corpus_id)
      WHERE a.article_id = NEW.article_id) = 1
BEGIN SELECT RAISE(ABORT, 'corpus is frozen: sentence is read-only'); END;

CREATE TRIGGER IF NOT EXISTS sentence_frozen_update BEFORE UPDATE ON sentence
WHEN (SELECT c.frozen FROM article a JOIN corpus c USING (corpus_id)
      WHERE a.article_id = OLD.article_id) = 1
  OR (SELECT c.frozen FROM article a JOIN corpus c USING (corpus_id)
      WHERE a.article_id = NEW.article_id) = 1
BEGIN SELECT RAISE(ABORT, 'corpus is frozen: sentence is read-only'); END;

CREATE TRIGGER IF NOT EXISTS sentence_frozen_delete BEFORE DELETE ON sentence
WHEN (SELECT c.frozen FROM article a JOIN corpus c USING (corpus_id)
      WHERE a.article_id = OLD.article_id) = 1
BEGIN SELECT RAISE(ABORT, 'corpus is frozen: sentence is read-only'); END;
