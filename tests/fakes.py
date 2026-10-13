"""Test doubles for model, search and fetch calls (no network)."""

from __future__ import annotations

from collections.abc import Callable

from locassess.prompts import load_prompt
from locassess.providers.llm import LLMResult
from locassess.providers.search import SearchHit
from locassess.stage0_corpus import FetchResult


class FakeLLM:
    """Answers by prompt id. Renders the real template so missing variables are caught."""

    def __init__(self, relevance: Callable[[dict], int] = lambda v: 2, queries: Callable[[dict], list] | None = None):
        self.relevance = relevance
        self.queries = queries
        self.calls: list[tuple[str, dict]] = []

    def complete(self, prompt_id, variables, json_schema=None, temperature=0, n=1, logprobs=False):
        prompt = load_prompt(prompt_id)
        prompt.render(variables)
        self.calls.append((prompt_id, variables))
        if prompt_id == "query_generation":
            langs = variables["languages"].split(", ")
            qs = self.queries(variables) if self.queries else [
                {"language": lang, "query": f"Leipzig {variables['main_id']} {lang} {i}"}
                for lang in langs for i in range(int(variables["n"]))
            ]
            parsed = {"queries": qs}
        elif prompt_id == "codebook_draft":
            mid = variables["main_id"]
            parsed = {"definition": f"Definition of {mid}.", "notes": f"Not {mid}.",
                      "positive_examples": [f"{mid} good news.", f"{mid} good news."],
                      "negative_examples": [f"{mid} bad news."],
                      "search_terms": [f"{mid} term a", f"{mid}  term b"]}
        elif prompt_id == "relevance":
            parsed = {"relevance": self.relevance(variables), "justification": "Test justification."}
        else:
            raise AssertionError(f"unexpected prompt {prompt_id}")
        return LLMResult(texts=[str(parsed)], parsed=[parsed], prompt_version=prompt.version)


class FakeSearch:
    def __init__(self, results: Callable[[str], list[str]]):
        self.results = results
        self.calls: list[dict] = []

    def search(self, query, allowed_domains, date_from, date_to, max_results):
        self.calls.append(dict(query=query, domains=allowed_domains, date_from=date_from, date_to=date_to))
        return [SearchHit(u, f"hit {u}", "", i) for i, u in enumerate(self.results(query)[:max_results], start=1)]


def article_html(title: str, published: str | None, body: str, lang: str | None = "en") -> str:
    meta = f'<meta property="article:published_time" content="{published}T08:00:00Z">' if published else ""
    paras = "".join(f"<p>{p}</p>" for p in body.split("\n\n"))
    lang_attr = f' lang="{lang}"' if lang else ""
    return (f'<html{lang_attr}><head><title>{title}</title>{meta}</head>'
            f"<body><article><h1>{title}</h1>{paras}</article></body></html>")


class FakeFetcher:
    def __init__(self, pages: dict[str, str | int]):
        self.pages = pages
        self.calls: list[str] = []

    def __call__(self, url):
        self.calls.append(url)
        page = self.pages.get(url, 404)
        if isinstance(page, int):
            return FetchResult(page, url, None)
        return FetchResult(200, url, page)
