"""Search interface (SPEC §6.2) and adapters: openai_web_search, brave, manual_csv.

Search-level domain filtering is only an optimisation; Stage 0 always re-applies the whitelist.
"""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import httpx

from locassess.config import Secrets
from locassess.prompts import load_prompt
from locassess.providers import ProviderNotConfigured
from locassess.providers.calls import CallOutcome, RateLimited, logged_call
from locassess.providers.llm import is_rate_limit


@dataclass
class SearchHit:
    url: str
    title: str
    snippet: str
    rank: int
    published_at: date | None = None


class SearchProvider(Protocol):
    def search(self, query: str, allowed_domains: list[str], date_from: date, date_to: date,
               max_results: int) -> list[SearchHit]: ...


def merge_ranked(lists: list[list[SearchHit]], max_results: int) -> list[SearchHit]:
    """Interleave ranked lists (round robin), drop duplicate URLs, re-rank from 1."""
    out: list[SearchHit] = []
    seen: set[str] = set()
    for i in range(max((len(x) for x in lists), default=0)):
        for hits in lists:
            if i < len(hits) and hits[i].url not in seen:
                seen.add(hits[i].url)
                out.append(hits[i])
    return [SearchHit(h.url, h.title, h.snippet, rank, h.published_at)
            for rank, h in enumerate(out[:max_results], start=1)]


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)] or [[]]


# ---------------------------------------------------------------------------
# OpenAI web search (Responses API `web_search` tool)
# Docs: https://developers.openai.com/api/docs/guides/tools-web-search
# ---------------------------------------------------------------------------

class OpenAIWebSearch:
    provider = "openai_web_search"
    MAX_ALLOWED_DOMAINS = 100   # documented limit of filters.allowed_domains

    def __init__(self, conn: sqlite3.Connection, model: str, client: Any = None,
                 secrets: Secrets | None = None, temperature: float | None = 0):
        self.conn, self.model, self.temperature = conn, model, temperature
        if client is None:
            key = (secrets or Secrets()).openai_api_key
            if key is None or not key.get_secret_value():
                raise ProviderNotConfigured("OPENAI_API_KEY is not set in .env")
            import openai
            client = openai.OpenAI(api_key=key.get_secret_value())
        self.client = client

    def search(self, query: str, allowed_domains: list[str], date_from: date, date_to: date,
               max_results: int) -> list[SearchHit]:
        prompt = load_prompt("web_search")
        text = prompt.render({"query": query, "date_from": date_from.isoformat(),
                              "date_to": date_to.isoformat(), "max_results": max_results})
        lists = []
        for chunk in _chunks(sorted(allowed_domains), self.MAX_ALLOWED_DOMAINS):
            tool: dict[str, Any] = {"type": "web_search"}
            if chunk:
                tool["filters"] = {"allowed_domains": chunk}
            request: dict[str, Any] = {"model": self.model, "input": text, "tools": [tool],
                                       "include": ["web_search_call.action.sources"]}
            if self.temperature is not None:
                request["temperature"] = self.temperature
            outcome, _ = logged_call(
                self.conn, provider=self.provider, model=self.model, purpose="web_search",
                prompt_version=prompt.version, request=request, fn=lambda r=request: self._call(r),
            )
            lists.append(parse_openai_web_search(outcome.response))
        return merge_ranked(lists, max_results)

    def _call(self, request: dict) -> CallOutcome:
        try:
            resp = self.client.responses.create(**request)
        except Exception as exc:
            if is_rate_limit(exc):
                raise RateLimited(str(exc)) from exc
            raise
        usage = getattr(resp, "usage", None)
        data = resp.model_dump(mode="json") if hasattr(resp, "model_dump") else resp
        return CallOutcome(response=data, input_tokens=getattr(usage, "input_tokens", None),
                           output_tokens=getattr(usage, "output_tokens", None),
                           model_version=getattr(resp, "model", None))


def parse_openai_web_search(data: dict) -> list[SearchHit]:
    """Cited URLs first (in citation order), then the other consulted sources."""
    cited: list[tuple[str, str]] = []
    sources: list[str] = []
    for item in data.get("output") or []:
        if item.get("type") == "web_search_call":
            for src in ((item.get("action") or {}).get("sources") or []):
                if src.get("url"):
                    sources.append(src["url"])
        elif item.get("type") == "message":
            for part in item.get("content") or []:
                for ann in part.get("annotations") or []:
                    if ann.get("type") == "url_citation" and ann.get("url"):
                        cited.append((ann["url"], ann.get("title") or ""))
    hits, seen = [], set()
    for url, title in cited + [(u, "") for u in sources]:
        if url not in seen:
            seen.add(url)
            hits.append(SearchHit(url, title, "", len(hits) + 1))
    return hits


# ---------------------------------------------------------------------------
# Brave Search API
# Docs: https://api-dashboard.search.brave.com/app/documentation/web-search/query
# ---------------------------------------------------------------------------

class BraveSearch:
    provider = "brave"
    ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
    MAX_COUNT = 20              # documented maximum of `count`
    MAX_QUERY_CHARS = 400       # conservative length budget for q incl. site: operators

    def __init__(self, conn: sqlite3.Connection, secrets: Secrets | None = None,
                 http: httpx.Client | None = None):
        key = (secrets or Secrets()).brave_api_key
        if key is None or not key.get_secret_value():
            raise ProviderNotConfigured("BRAVE_API_KEY is not set in .env")
        self.conn, self._key = conn, key.get_secret_value()
        self.http = http or httpx.Client(timeout=30)

    def search(self, query: str, allowed_domains: list[str], date_from: date, date_to: date,
               max_results: int) -> list[SearchHit]:
        lists = []
        for q in self._queries(query, sorted(allowed_domains)):
            params = {"q": q, "count": min(max_results, self.MAX_COUNT),
                      "freshness": f"{date_from.isoformat()}to{date_to.isoformat()}"}
            outcome, _ = logged_call(
                self.conn, provider=self.provider, model=None, purpose="web_search",
                request={"url": self.ENDPOINT, "params": params}, fn=lambda p=params: self._call(p),
            )
            lists.append(parse_brave(outcome.response))
        return merge_ranked(lists, max_results)

    def _queries(self, query: str, domains: list[str]) -> list[str]:
        """Split the domain list into site: groups that keep q within the length budget."""
        if not domains:
            return [query]
        out, group = [], []
        for d in domains:
            candidate = group + [d]
            if group and len(self._q(query, candidate)) > self.MAX_QUERY_CHARS:
                out.append(self._q(query, group))
                candidate = [d]
            group = candidate
        out.append(self._q(query, group))
        return out

    @staticmethod
    def _q(query: str, domains: list[str]) -> str:
        return f"{query} ({' OR '.join(f'site:{d}' for d in domains)})"

    def _call(self, params: dict) -> CallOutcome:
        resp = self.http.get(self.ENDPOINT, params=params,
                             headers={"Accept": "application/json", "X-Subscription-Token": self._key})
        if resp.status_code == 429:
            raise RateLimited("brave: HTTP 429")
        resp.raise_for_status()
        return CallOutcome(response=resp.json())


def parse_brave(data: dict) -> list[SearchHit]:
    hits = []
    for r in ((data.get("web") or {}).get("results") or []):
        if not r.get("url"):
            continue
        published = None
        page_age = r.get("page_age")  # not in the documented example; used only if present
        if isinstance(page_age, str):
            try:
                published = date.fromisoformat(page_age[:10])
            except ValueError:
                published = None
        hits.append(SearchHit(r["url"], r.get("title") or "", r.get("description") or "",
                              len(hits) + 1, published))
    return hits


# ---------------------------------------------------------------------------
# Manual CSV (hand-picked URLs: columns url, note)
# ---------------------------------------------------------------------------

class ManualCSVSearch:
    provider = "manual_csv"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows = load_manual_csv(self.path)

    def search(self, query: str, allowed_domains: list[str], date_from: date, date_to: date,
               max_results: int) -> list[SearchHit]:
        """Returns every listed URL in file order; the query is ignored."""
        return [SearchHit(url, "", note, rank) for rank, (url, note) in enumerate(self.rows, start=1)]


def load_manual_csv(path: str | Path) -> list[tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "url" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: needs a 'url' column (and optionally 'note')")
        for row in reader:
            url = (row.get("url") or "").strip()
            if url:
                rows.append((url, (row.get("note") or "").strip()))
    return rows


def make_search(conn: sqlite3.Connection, provider: str, model: str,
                temperature: float | None = 0) -> SearchProvider:
    if provider == "openai_web_search":
        return OpenAIWebSearch(conn, model, temperature=temperature)
    if provider == "brave":
        return BraveSearch(conn)
    if provider == "manual_csv":
        raise ProviderNotConfigured("manual_csv has no queries; use `locassess corpus import --csv`")
    raise ProviderNotConfigured(f"unknown search provider {provider!r}")
