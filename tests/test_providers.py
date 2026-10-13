"""LLM and search adapters with mocked clients (no network)."""

import json
from datetime import date
from types import SimpleNamespace

import httpx
import pytest

from locassess.config import Secrets
from locassess.providers import ProviderNotConfigured
from locassess.providers.llm import OpenAILLM
from locassess.providers.search import (
    BraveSearch, ManualCSVSearch, OpenAIWebSearch, load_manual_csv, merge_ranked, parse_openai_web_search,
)

FROM, TO = date(2021, 10, 13), date(2026, 10, 13)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("locassess.providers.calls.time.sleep", lambda s: None)


class RateLimit(Exception):
    status_code = 429


class FakeResponses:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def fake_client(*responses):
    return SimpleNamespace(responses=FakeResponses(responses))


# --- OpenAI LLM ------------------------------------------------------------

def llm_response(text):
    return SimpleNamespace(output_text=text, id="resp_1", model="gpt-test-2026", output=[],
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def test_openai_llm_structured_output_and_call_log(conn):
    client = fake_client(RateLimit("slow down"), llm_response('{"relevance": 2, "justification": "x"}'))
    llm = OpenAILLM(conn, "gpt-test", client=client)
    res = llm.complete("relevance", {k: "v" for k in [
        "location_name", "region", "country", "aliases", "criteria_list", "title", "published_at", "text"]},
        json_schema={"type": "object"})
    assert res.parsed == [{"relevance": 2, "justification": "x"}]
    assert res.prompt_version == "1.0" and res.model_version == "gpt-test-2026"
    req = client.responses.requests[-1]
    assert req["text"]["format"] == {"type": "json_schema", "name": "relevance", "schema": {"type": "object"},
                                     "strict": True}
    assert req["temperature"] == 0 and len(client.responses.requests) == 2  # retried after 429
    row = conn.execute("SELECT * FROM call_log").fetchone()
    assert (row["provider"], row["purpose"], row["prompt_version"], row["input_tokens"]) == \
        ("openai", "relevance", "1.0", 10)


def test_openai_llm_requires_key(conn, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderNotConfigured):
        OpenAILLM(conn, "gpt-test")


# --- OpenAI web search -----------------------------------------------------

WEB_SEARCH_OUTPUT = {
    "output": [
        {"type": "web_search_call", "action": {"type": "search", "sources": [
            {"type": "url", "url": "https://lvz.de/a"}, {"type": "url", "url": "https://mdr.de/b"}]}},
        {"type": "message", "content": [{"type": "output_text", "text": "...", "annotations": [
            {"type": "url_citation", "url": "https://mdr.de/b", "title": "B"},
            {"type": "url_citation", "url": "https://lvz.de/c", "title": "C"}]}]},
    ]
}


def test_parse_openai_web_search_cited_first():
    hits = parse_openai_web_search(WEB_SEARCH_OUTPUT)
    assert [(h.url, h.title, h.rank) for h in hits] == [
        ("https://mdr.de/b", "B", 1), ("https://lvz.de/c", "C", 2), ("https://lvz.de/a", "", 3)]


def test_openai_web_search_request_and_domain_chunks(conn):
    resp = SimpleNamespace(model_dump=lambda mode=None: WEB_SEARCH_OUTPUT, model="m", usage=None)
    client = fake_client(resp, resp)
    domains = [f"d{i}.de" for i in range(150)]
    hits = OpenAIWebSearch(conn, "m", client=client).search("Leipzig port", domains, FROM, TO, 2)
    reqs = client.responses.requests
    assert len(reqs) == 2  # 100 + 50 domains
    assert reqs[0]["tools"][0]["type"] == "web_search"
    assert len(reqs[0]["tools"][0]["filters"]["allowed_domains"]) == 100
    assert reqs[0]["include"] == ["web_search_call.action.sources"]
    assert "Leipzig port" in reqs[0]["input"] and "2021-10-13" in reqs[0]["input"]
    assert [h.rank for h in hits] == [1, 2]


# --- Brave -----------------------------------------------------------------

def test_brave_request_and_parse(conn):
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"web": {"results": [
            {"url": "https://lvz.de/a", "title": "A", "description": "d", "page_age": "2025-04-12T14:22:41"},
            {"url": "https://mdr.de/b", "title": "B", "description": "e"}]}})

    brave = BraveSearch(conn, secrets=Secrets(brave_api_key="brave-secret"),
                        http=httpx.Client(transport=httpx.MockTransport(handler)))
    hits = brave.search("Leipzig port", ["lvz.de", "mdr.de"], FROM, TO, 50)
    req = seen[-1]
    assert req.headers["X-Subscription-Token"] == "brave-secret"
    assert req.url.params["q"] == "Leipzig port (site:lvz.de OR site:mdr.de)"
    assert req.url.params["freshness"] == "2021-10-13to2026-10-13"
    assert req.url.params["count"] == "20"
    assert [(h.url, h.published_at) for h in hits] == [("https://lvz.de/a", date(2025, 4, 12)), ("https://mdr.de/b", None)]
    logged = " ".join(str(dict(r)) for r in conn.execute("SELECT * FROM call_log"))
    assert "brave-secret" not in logged


def test_brave_splits_long_domain_lists(conn):
    brave = BraveSearch(conn, secrets=Secrets(brave_api_key="k"))
    qs = brave._queries("Leipzig port", [f"domain{i}.example.de" for i in range(40)])
    assert len(qs) > 1 and all(len(q) <= BraveSearch.MAX_QUERY_CHARS for q in qs)
    assert sum(q.count("site:") for q in qs) == 40


def test_brave_requires_key(conn, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    with pytest.raises(ProviderNotConfigured):
        BraveSearch(conn)


# --- manual CSV / merge ----------------------------------------------------

def test_manual_csv(tmp_path):
    p = tmp_path / "urls.csv"
    p.write_text("url,note\nhttps://lvz.de/a,first\n,\nhttps://mdr.de/b,\n", encoding="utf-8")
    assert load_manual_csv(p) == [("https://lvz.de/a", "first"), ("https://mdr.de/b", "")]
    assert [h.rank for h in ManualCSVSearch(p).search("", [], FROM, TO, 0)] == [1, 2]
    bad = tmp_path / "bad.csv"
    bad.write_text("link\nx\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_manual_csv(bad)


def test_merge_ranked_round_robin():
    from locassess.providers.search import SearchHit
    a = [SearchHit("a1", "", "", 1), SearchHit("a2", "", "", 2)]
    b = [SearchHit("b1", "", "", 1), SearchHit("a1", "", "", 2), SearchHit("b3", "", "", 3)]
    assert [(h.url, h.rank) for h in merge_ranked([a, b], 4)] == [("a1", 1), ("b1", 2), ("a2", 3), ("b3", 4)]
