"""Stage 0 — web search and corpus (SPEC §7).

Models only suggest (queries, relevance); fixed rules filter and select. After freezing,
the corpus and its articles are read-only.

Exclusion reasons carry the step that excluded the article as a prefix
(`fetch:`, `filter:`, `relevance:`, `selection:`), so the selection log can count them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
import trafilatura

from locassess.config import AppConfig, config_hash
from locassess.criteria import MainCriterion
from locassess.db import repo
from locassess.prompts import load_prompt
from locassess.providers.calls import CallOutcome, RateLimited, logged_call
from locassess.providers.llm import LLM
from locassess.providers.search import ManualCSVSearch, SearchHit, SearchProvider

DUPLICATE_SIMILARITY = 0.9      # shingled Jaccard threshold for wire-agency copies (SPEC §7.4)
SHINGLE_SIZE = 5
RELEVANCE_MAX_CHARS = 20000     # article text passed to the relevance prompt
USER_AGENT = "LocAssess/0.1 (academic research; Universitaet Hamburg)"

Progress = Callable[[str], None]


def _quiet(_: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Period
# ---------------------------------------------------------------------------

def period(run_date: date, years_back: int) -> tuple[date, date]:
    """(date_from, date_to): the same calendar day `years_back` years before the run date."""
    try:
        start = run_date.replace(year=run_date.year - years_back)
    except ValueError:  # 29 February
        start = run_date.replace(year=run_date.year - years_back, day=28)
    return start, run_date


# ---------------------------------------------------------------------------
# Whitelist and URLs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WhitelistEntry:
    domain: str
    outlet_name: str
    media_type: str
    language: str
    country: str


class WhitelistError(ValueError):
    pass


def load_whitelist(path: str | Path) -> list[WhitelistEntry]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        entries = [
            WhitelistEntry(
                domain=_bare_host(row["domain"]), outlet_name=(row.get("outlet_name") or "").strip(),
                media_type=(row.get("media_type") or "").strip(), language=(row.get("language") or "").strip(),
                country=(row.get("country") or "").strip(),
            )
            for row in csv.DictReader(f) if (row.get("domain") or "").strip()
        ]
    if not entries:
        raise WhitelistError(f"{path} contains no domains; add the allowed news outlets first")
    return entries


def _bare_host(value: str) -> str:
    v = value.strip().lower()
    if "://" in v:
        v = urlsplit(v).hostname or ""
    v = v.split("/")[0]
    return v[4:] if v.startswith("www.") else v


def host_of(url: str) -> str:
    return _bare_host(urlsplit(url).hostname or "")


def match_whitelist(url: str, whitelist: Iterable[WhitelistEntry]) -> WhitelistEntry | None:
    """Entry whose domain equals the URL host or is a parent domain of it (longest match)."""
    host = host_of(url)
    matches = [e for e in whitelist if host == e.domain or host.endswith("." + e.domain)]
    return max(matches, key=lambda e: len(e.domain), default=None)


_TRACKING = re.compile(r"^(utm_\w+|fbclid|gclid|mc_cid|mc_eid|ref|cmpid)$", re.I)


def normalize_url(url: str) -> str:
    """Canonical form for duplicate detection: https, bare host, no fragment/tracking/trailing slash."""
    parts = urlsplit(url.strip())
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query) if not _TRACKING.match(k)))
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(("https", _bare_host(parts.netloc), path, query, ""))


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    status: int
    final_url: str
    html: str | None


Fetcher = Callable[[str], FetchResult]


class HttpFetcher:
    """Downloads a page; every request is written to call_log (without the HTML)."""

    def __init__(self, conn: sqlite3.Connection, client: httpx.Client | None = None):
        self.conn = conn
        self.client = client or httpx.Client(timeout=30, follow_redirects=True,
                                             headers={"User-Agent": USER_AGENT})

    def __call__(self, url: str) -> FetchResult:
        holder: dict[str, FetchResult] = {}

        def call() -> CallOutcome:
            resp = self.client.get(url)
            if resp.status_code == 429:
                raise RateLimited(f"HTTP 429 for {url}")
            html = resp.text if resp.status_code == 200 else None
            holder["r"] = FetchResult(resp.status_code, str(resp.url), html)
            return CallOutcome({"status": resp.status_code, "final_url": str(resp.url),
                                "bytes": len(resp.content)})

        logged_call(self.conn, provider="http", model=None, purpose="fetch", request={"url": url}, fn=call)
        return holder["r"]


@dataclass
class ExtractedArticle:
    title: str | None
    published_at: date | None
    text: str
    language: str | None


_HTML_LANG = re.compile(r"<html[^>]*\blang=[\"']?([a-zA-Z]{2})", re.I)


def extract_article(html: str, url: str) -> ExtractedArticle | None:
    """Full text and metadata via trafilatura; None if no text could be extracted."""
    raw = trafilatura.extract(html, url=url, output_format="json", with_metadata=True,
                              include_comments=False, include_tables=False)
    if not raw:
        return None
    data = json.loads(raw)
    text = (data.get("text") or "").strip()
    if not text:
        return None
    published = None
    if data.get("date"):
        try:
            published = date.fromisoformat(str(data["date"])[:10])
        except ValueError:
            published = None
    lang = data.get("language")
    if not lang:
        m = _HTML_LANG.search(html[:5000])
        lang = m.group(1).lower() if m else None
    return ExtractedArticle(data.get("title"), published, text, lang)


# ---------------------------------------------------------------------------
# 7.4 Rule filter
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(text.split())


def mentions_location(text: str, names: Iterable[str]) -> bool:
    """Case-insensitive whole-word match of the location name or an alias."""
    return any(re.search(rf"(?<!\w){re.escape(n.strip())}(?!\w)", text, re.I) for n in names if n.strip())


def shingles(text: str, k: int = SHINGLE_SIZE) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass
class FilterInput:
    article_id: int
    url: str
    published_at: date | None
    text: str
    title: str | None = None


def filter_reason(a: FilterInput, whitelist: list[WhitelistEntry], date_from: date, date_to: date,
                  names: list[str], min_words: int) -> str | None:
    """First failing rule (fixed order), or None if the article passes rules 1–4."""
    if match_whitelist(a.url, whitelist) is None:
        return "filter:domain_not_whitelisted"
    if a.published_at is None:
        return "filter:no_publication_date"
    if not date_from <= a.published_at <= date_to:
        return "filter:outside_period"
    if not mentions_location(f"{a.title or ''}\n{a.text}", names):   # title or text (author decision)
        return "filter:location_not_mentioned"
    if word_count(a.text) < min_words:
        return "filter:too_short"
    return None


def find_duplicates(articles: list[FilterInput],
                    threshold: float = DUPLICATE_SIMILARITY) -> dict[int, int]:
    """article_id → article_id of the earlier article it duplicates (same URL or similar text).

    Articles are compared in the given order; the first occurrence is kept.
    """
    kept: list[tuple[FilterInput, str, set]] = []
    dup: dict[int, int] = {}
    for a in articles:
        norm, sh = normalize_url(a.url), shingles(a.text)
        for k, k_norm, k_sh in kept:
            if norm == k_norm or jaccard(sh, k_sh) >= threshold:
                dup[a.article_id] = k.article_id
                break
        else:
            kept.append((a, norm, sh))
    return dup


# ---------------------------------------------------------------------------
# 7.6 Selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    article_id: int
    relevance: int
    published_at: date
    criteria: frozenset[str]


def select_articles(candidates: Iterable[Candidate], min_relevance: int, max_per_year: int,
                    target: int) -> list[int]:
    """Greedy selection by fixed rules.

    1. Only relevance >= min_relevance.
    2. At most max_per_year articles per calendar year.
    3. Articles that cover a criterion without a selected article yet come first.
    4. Ties: higher relevance, then newer date, then lower article_id.
    5. Stop at `target` articles.
    """
    pool = [c for c in candidates if c.relevance >= min_relevance]
    selected: list[int] = []
    covered: set[str] = set()
    per_year: Counter[int] = Counter()

    def key(c: Candidate):
        return (not (c.criteria - covered), -c.relevance, -c.published_at.toordinal(), c.article_id)

    while len(selected) < target:
        open_ = [c for c in pool if c.article_id not in selected and per_year[c.published_at.year] < max_per_year]
        if not open_:
            break
        best = min(open_, key=key)
        selected.append(best.article_id)
        covered |= best.criteria
        per_year[best.published_at.year] += 1
    return selected


# ---------------------------------------------------------------------------
# 7.1 Query generation
# ---------------------------------------------------------------------------

def criterion_search_info(main: MainCriterion) -> tuple[str, list[str]]:
    """(definition, search terms) from the criteria framework.

    The codebook (M5) will provide reviewed definitions and search terms; until then the
    sub-criteria names (or the main criterion name) seed the queries.
    """
    terms = [s.name for s in main.sub_criteria] or [main.name]
    definition = f"{main.name} ({main.factor})" + (f": {', '.join(terms)}" if main.sub_criteria else "")
    return definition, terms


def query_languages(cfg: AppConfig) -> list[str]:
    langs = ["en"]
    if cfg.location.local_language and cfg.location.local_language.lower() != "en":
        langs.append(cfg.location.local_language.lower())
    return langs


def location_names(cfg: AppConfig) -> list[str]:
    names = [cfg.location.name, *cfg.location.aliases]
    return list(dict.fromkeys(n.strip() for n in names if n and n.strip()))


def query_schema(languages: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"language": {"type": "string", "enum": languages},
                                   "query": {"type": "string"}},
                    "required": ["language", "query"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["queries"],
        "additionalProperties": False,
    }


def generate_queries(llm: LLM, cfg: AppConfig, main: MainCriterion, date_from: date,
                     date_to: date) -> tuple[list[tuple[str, str]], str | None]:
    """[(language, query)] with exactly queries_per_criterion queries per language (rules applied)."""
    langs = query_languages(cfg)
    names = location_names(cfg)
    n = cfg.corpus.queries_per_criterion
    definition, terms = criterion_search_info(main)
    result = llm.complete("query_generation", {
        "location_name": cfg.location.name, "region": cfg.location.region or "-",
        "country": cfg.location.country, "aliases": ", ".join(names),
        "date_from": date_from.isoformat(), "date_to": date_to.isoformat(),
        "main_id": main.main_id, "criterion_name": main.name, "definition": definition,
        "search_terms": ", ".join(terms), "n": n, "languages": ", ".join(langs),
    }, json_schema=query_schema(langs), temperature=cfg.models.temperature)

    out: list[tuple[str, str]] = []
    for lang in langs:
        seen: set[str] = set()
        for q in result.parsed[0]["queries"]:
            text = " ".join(str(q["query"]).split())
            if q["language"] != lang or not text or text.lower() in seen:
                continue
            if not mentions_location(text, names):   # rule: every query names the location
                text = f"{cfg.location.name} {text}"
            seen.add(text.lower())
            out.append((lang, text))
            if len(seen) == n:
                break
    return out, result.prompt_version


# ---------------------------------------------------------------------------
# 7.5 Relevance
# ---------------------------------------------------------------------------

RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {"relevance": {"type": "integer", "enum": [0, 1, 2]},
                   "justification": {"type": "string"}},
    "required": ["relevance", "justification"],
    "additionalProperties": False,
}


def rate_relevance(llm: LLM, cfg: AppConfig, criteria: list[MainCriterion], title: str | None,
                   published_at: str | None, text: str) -> tuple[int, str]:
    result = llm.complete("relevance", {
        "location_name": cfg.location.name, "region": cfg.location.region or "-",
        "country": cfg.location.country, "aliases": ", ".join(location_names(cfg)),
        "criteria_list": "\n".join(f"- {m.main_id} {m.name} ({m.factor})" for m in criteria),
        "title": title or "", "published_at": published_at or "", "text": text[:RELEVANCE_MAX_CHARS],
    }, json_schema=RELEVANCE_SCHEMA, temperature=cfg.models.temperature)
    parsed = result.parsed[0]
    rel = int(parsed["relevance"])
    if rel not in (0, 1, 2):
        raise ValueError(f"relevance out of range: {rel}")
    return rel, str(parsed["justification"]).strip()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _new_corpus(conn: sqlite3.Connection, cfg: AppConfig, run_date: date, note: str | None) -> tuple[int, date, date]:
    date_from, date_to = period(run_date, cfg.period.years_back)
    corpus_id = repo.create_corpus(
        conn, json.dumps(cfg.location.model_dump(mode="json"), sort_keys=True),
        date_from.isoformat(), date_to.isoformat(), config_hash(cfg), note,
    )
    return corpus_id, date_from, date_to


def _record_hits(conn: sqlite3.Connection, corpus_id: int, query_id: int, hits: list[SearchHit],
                 known: dict[str, int]) -> None:
    """Store hits; create one article row per new normalized URL."""
    for h in hits:
        norm = normalize_url(h.url)
        if norm in known:
            repo.insert_search_hit(conn, query_id, h.url, h.title, h.rank, "duplicate_url",
                                   f"same URL as article {known[norm]}")
            continue
        known[norm] = repo.insert_article(conn, corpus_id, url=h.url, domain=host_of(h.url),
                                          title=h.title or None)
        repo.insert_search_hit(conn, query_id, h.url, h.title, h.rank, "new_url")


def build_corpus(conn: sqlite3.Connection, cfg: AppConfig, criteria: list[MainCriterion], llm: LLM,
                 search: SearchProvider, fetch: Fetcher, run_date: date | None = None,
                 note: str | None = None, progress: Progress = _quiet) -> int:
    """Stage 0 steps 1–6. Returns the corpus_id; the corpus is not frozen yet."""
    whitelist = load_whitelist(cfg.whitelist_file)
    corpus_id, date_from, date_to = _new_corpus(conn, cfg, run_date or cfg.resolved_run_date(), note)
    domains = sorted({e.domain for e in whitelist})
    known: dict[str, int] = {}
    for main in criteria:
        progress(f"queries for {main.main_id} {main.name}")
        queries, prompt_version = generate_queries(llm, cfg, main, date_from, date_to)
        for lang, text in queries:
            qid = repo.insert_search_query(conn, corpus_id, main.main_id, text, lang, prompt_version)
            try:
                hits = search.search(text, domains, date_from, date_to, cfg.corpus.hits_per_query)
            except Exception as exc:  # logged in call_log; one failed query must not stop the run
                progress(f"  search failed for {text!r}: {exc}")
                continue
            repo.set_query_hit_count(conn, qid, len(hits))
            _record_hits(conn, corpus_id, qid, hits, known)
    _process_articles(conn, corpus_id, cfg, criteria, whitelist, llm, fetch, date_from, date_to, progress)
    return corpus_id


def import_corpus(conn: sqlite3.Connection, cfg: AppConfig, criteria: list[MainCriterion], llm: LLM,
                  csv_path: str | Path, fetch: Fetcher, run_date: date | None = None,
                  note: str | None = None, progress: Progress = _quiet) -> int:
    """Manual path: hand-picked URLs replace steps 1–2; steps 3–6 run unchanged."""
    whitelist = load_whitelist(cfg.whitelist_file)
    corpus_id, date_from, date_to = _new_corpus(conn, cfg, run_date or cfg.resolved_run_date(), note)
    hits = ManualCSVSearch(csv_path).search("", [], date_from, date_to, 0)
    qid = repo.insert_search_query(conn, corpus_id, None, f"manual_csv:{Path(csv_path).name}", None, None)
    repo.set_query_hit_count(conn, qid, len(hits))
    _record_hits(conn, corpus_id, qid, hits, {})
    _process_articles(conn, corpus_id, cfg, criteria, whitelist, llm, fetch, date_from, date_to, progress)
    return corpus_id


def criteria_by_url(conn: sqlite3.Connection, corpus_id: int) -> dict[str, set[str]]:
    """Normalized URL → main criteria whose queries found it (basis for criterion coverage)."""
    out: dict[str, set[str]] = {}
    for url, main_id in repo.hit_criteria(conn, corpus_id):
        s = out.setdefault(normalize_url(url), set())
        if main_id:
            s.add(main_id)
    return out


def _passed_filter(row) -> bool:
    """Article has text and was not excluded by fetch or filter (later steps may still exclude it)."""
    reason = row["exclusion_reason"] or ""
    return row["text"] is not None and (not reason or reason.startswith(("relevance:", "selection:")))


def _process_articles(conn: sqlite3.Connection, corpus_id: int, cfg: AppConfig,
                      criteria: list[MainCriterion], whitelist: list[WhitelistEntry], llm: LLM,
                      fetch: Fetcher, date_from: date, date_to: date, progress: Progress) -> None:
    """Stage 0 steps 3–6. Resumable: only unprocessed articles are fetched and only unrated
    ones are rated; duplicates and the selection are recomputed over the current state."""
    names = location_names(cfg)

    # 7.3 fetch (only whitelisted domains are downloaded; the rest is excluded by rule 1) + 7.4 rules 1–4
    todo = [a for a in repo.get_articles(conn, corpus_id) if a["fetched_at"] is None and not a["exclusion_reason"]]
    for i, a in enumerate(todo, start=1):
        entry = match_whitelist(a["url"], whitelist)
        if entry is None:
            repo.update_article(conn, a["article_id"], exclusion_reason="filter:domain_not_whitelisted")
            continue
        progress(f"fetch {i}/{len(todo)} {a['url']}")
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            res = fetch(a["url"])
        except Exception as exc:
            repo.update_article(conn, a["article_id"], fetched_at=fetched_at,
                                exclusion_reason=f"fetch:error:{type(exc).__name__}")
            continue
        if res.status != 200 or not res.html:
            repo.update_article(conn, a["article_id"], fetched_at=fetched_at,
                                exclusion_reason=f"fetch:http_{res.status}")
            continue
        ext = extract_article(res.html, res.final_url)
        if ext is None:
            repo.update_article(conn, a["article_id"], fetched_at=fetched_at, exclusion_reason="fetch:empty_text")
            continue
        # the final URL counts, so redirects cannot bypass the whitelist
        title = ext.title or a["title"]
        fi = FilterInput(a["article_id"], res.final_url or a["url"], ext.published_at, ext.text, title)
        repo.update_article(
            conn, a["article_id"], outlet=entry.outlet_name or entry.domain, title=title,
            published_at=ext.published_at.isoformat() if ext.published_at else None, fetched_at=fetched_at,
            language=ext.language or entry.language or None, text=ext.text,   # page language first
            text_sha256=hashlib.sha256(ext.text.encode("utf-8")).hexdigest(), word_count=word_count(ext.text),
            exclusion_reason=filter_reason(fi, whitelist, date_from, date_to, names, cfg.corpus.min_words),
        )

    # 7.4 rule 5: duplicates (same URL or near-identical text); the earlier article is kept
    passed = [FilterInput(r["article_id"], r["url"], None, r["text"])
              for r in repo.get_articles(conn, corpus_id) if _passed_filter(r)]
    dups = find_duplicates(passed)
    for aid, original in dups.items():
        repo.update_article(conn, aid, selected=0, exclusion_reason=f"filter:duplicate_of_{original}")

    # 7.5 relevance (LLM suggestion); articles rated in an earlier run keep their rating
    rows = [r for r in repo.get_articles(conn, corpus_id) if _passed_filter(r)]
    unrated = [r for r in rows if r["relevance"] is None]
    for i, r in enumerate(unrated, start=1):
        progress(f"relevance {i}/{len(unrated)} {r['url']}")
        try:
            rel, why = rate_relevance(llm, cfg, criteria, r["title"], r["published_at"], r["text"])
        except Exception as exc:
            repo.update_article(conn, r["article_id"], exclusion_reason=f"relevance:error:{type(exc).__name__}")
            continue
        repo.update_article(conn, r["article_id"], relevance=rel, relevance_reason=why,
                            exclusion_reason="relevance:below_min" if rel < cfg.corpus.min_relevance else None)

    # 7.6 selection (rules), recomputed from scratch over all relevant articles
    rows = {r["article_id"]: r for r in repo.get_articles(conn, corpus_id)}
    rated = [aid for aid, r in rows.items() if _passed_filter(r) and r["relevance"] is not None
             and r["relevance"] >= cfg.corpus.min_relevance and not (r["exclusion_reason"] or "").startswith("relevance:")]
    crit_by_url = criteria_by_url(conn, corpus_id)
    candidates = [
        Candidate(aid, rows[aid]["relevance"], date.fromisoformat(rows[aid]["published_at"]),
                  frozenset(crit_by_url.get(normalize_url(rows[aid]["url"]), set())))
        for aid in rated
    ]
    chosen = select_articles(candidates, cfg.corpus.min_relevance, cfg.corpus.max_per_year,
                             cfg.corpus.target_articles)
    years = Counter(rows[aid]["published_at"][:4] for aid in chosen)
    for aid in rated:
        if aid in chosen:
            repo.update_article(conn, aid, selected=1, exclusion_reason=None)
        elif years[rows[aid]["published_at"][:4]] >= cfg.corpus.max_per_year:
            repo.update_article(conn, aid, selected=0, exclusion_reason="selection:max_per_year")
        else:
            repo.update_article(conn, aid, selected=0, exclusion_reason="selection:target_reached")


def process_corpus(conn: sqlite3.Connection, corpus_id: int, cfg: AppConfig, criteria: list[MainCriterion],
                   llm: LLM, fetch: Fetcher, progress: Progress = _quiet) -> None:
    """Continue an unfinished corpus (e.g. after an interrupted build) without new searches.

    The corpus' own location and period are used; corpus rules (min_words, selection) come from `cfg`.
    """
    corpus = repo.get_corpus(conn, corpus_id)
    if corpus["frozen"]:
        raise repo.CorpusFrozenError(f"corpus {corpus_id} is frozen and read-only")
    cfg = cfg.model_copy(update={"location": type(cfg.location).model_validate(json.loads(corpus["location_json"]))})
    _process_articles(conn, corpus_id, cfg, criteria, load_whitelist(cfg.whitelist_file), llm, fetch,
                      date.fromisoformat(corpus["period_from"]), date.fromisoformat(corpus["period_to"]), progress)


# ---------------------------------------------------------------------------
# 7.7 Freeze (sentences are split here so that frozen articles never change later)
# ---------------------------------------------------------------------------

def split_sentences(text: str, language: str | None) -> list[str]:
    import pysbd
    from pysbd.languages import LANGUAGE_CODES

    lang = (language or "en").lower()[:2]
    seg = pysbd.Segmenter(language=lang if lang in LANGUAGE_CODES else "en", clean=False)
    out = []
    for para in text.splitlines():
        out.extend(s.strip() for s in seg.segment(para) if s.strip())
    return out


def sentence_rows(sentences: list[str], max_per_section: int) -> list[tuple[str, int, str]]:
    """(S1.., section starting at 1, text); each section holds at most max_per_section sentences."""
    return [(f"S{i + 1}", i // max_per_section + 1, s) for i, s in enumerate(sentences)]


def freeze(conn: sqlite3.Connection, corpus_id: int, max_options_per_question: int) -> int:
    """Split the selected articles into sentences and mark the corpus frozen. Returns #articles."""
    selected = repo.get_articles(conn, corpus_id, selected_only=True)
    if not selected:
        raise ValueError(f"corpus {corpus_id} has no selected articles")
    for a in selected:
        rows = sentence_rows(split_sentences(a["text"], a["language"]), max_options_per_question)
        repo.replace_sentences(conn, a["article_id"], rows)
    repo.freeze_corpus(conn, corpus_id)
    return len(selected)


# ---------------------------------------------------------------------------
# 7.8 Selection log
# ---------------------------------------------------------------------------

def selection_log(conn: sqlite3.Connection, corpus_id: int) -> tuple[list[dict], list[dict]]:
    """PRISMA-style (summary rows, per-article rows)."""
    n_queries = conn.execute("SELECT COUNT(*) FROM search_query WHERE corpus_id = ?", (corpus_id,)).fetchone()[0]
    n_hits = conn.execute(
        "SELECT COUNT(*) FROM search_hit JOIN search_query USING (query_id) WHERE corpus_id = ?", (corpus_id,),
    ).fetchone()[0]
    articles = repo.get_articles(conn, corpus_id)
    reasons = Counter(a["exclusion_reason"] for a in articles if a["exclusion_reason"])

    # Counts come from what actually happened to each article, so an unfinished run is reported truthfully.
    def reason_of(a) -> str:
        return a["exclusion_reason"] or ""

    n_unique = len(articles)
    n_whitelisted = n_unique - reasons.get("filter:domain_not_whitelisted", 0)
    n_pending = sum(1 for a in articles if a["fetched_at"] is None and not a["exclusion_reason"])
    n_fetched = sum(1 for a in articles if a["text"] is not None)
    n_passed = sum(1 for a in articles if a["text"] is not None
                   and (not a["exclusion_reason"] or reason_of(a).startswith(("relevance:", "selection:"))))
    n_relevant = sum(1 for a in articles if a["relevance"] is not None and not reason_of(a).startswith("relevance:"))
    n_selected = sum(1 for a in articles if a["selected"])

    summary = [
        {"step": "queries", "count": n_queries, "detail": ""},
        {"step": "search_hits", "count": n_hits, "detail": ""},
        {"step": "unique_urls", "count": n_unique, "detail": f"{n_hits - n_unique} duplicate URLs across queries"},
        {"step": "whitelisted", "count": n_whitelisted,
         "detail": f"{n_pending} not processed yet (run unfinished)" if n_pending else ""},
        {"step": "fetched", "count": n_fetched, "detail": ""},
        {"step": "passed_filter", "count": n_passed, "detail": ""},
        {"step": "relevant", "count": n_relevant, "detail": ""},
        {"step": "selected", "count": n_selected, "detail": ""},
    ]
    summary += [{"step": "excluded", "count": n, "detail": r} for r, n in sorted(reasons.items())]

    crit_by_url = criteria_by_url(conn, corpus_id)
    per_article = [
        {"article_id": a["article_id"], "url": a["url"], "domain": a["domain"], "outlet": a["outlet"],
         "title": a["title"], "published_at": a["published_at"], "language": a["language"],
         "word_count": a["word_count"], "relevance": a["relevance"], "relevance_reason": a["relevance_reason"],
         "criteria": " ".join(sorted(crit_by_url.get(normalize_url(a["url"]), set()))), "selected": a["selected"],
         "exclusion_reason": a["exclusion_reason"]}
        for a in articles
    ]
    return summary, per_article


def export_selection_log(conn: sqlite3.Connection, corpus_id: int, out_dir: str | Path) -> tuple[Path, Path]:
    summary, per_article = selection_log(conn, corpus_id)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = (out_dir / f"selection_log_corpus{corpus_id}.csv",
             out_dir / f"selection_articles_corpus{corpus_id}.csv")
    for path, rows, fields in ((paths[0], summary, ["step", "count", "detail"]),
                               (paths[1], per_article, list(per_article[0]) if per_article else ["article_id"])):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return paths
