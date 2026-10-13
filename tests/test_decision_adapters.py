"""Decision-model adapters with mocked clients (no network)."""

import json
import math
from types import SimpleNamespace

import httpx
import pytest

from locassess.config import Secrets
from locassess.providers import ProviderNotConfigured
from locassess.providers.decision import (
    JevDecisions, LLMStructured, OpenAIDecisions, Option, Question, answer_schema,
    distributions_from_logprobs, distributions_from_samples,
)

QS = [Question("direction", "Direction?", (Option("-1", "less"), Option("0", "none"), Option("1", "more"))),
      Question("evidence_sentence", "Which sentence?", tuple(Option(f"S{i}") for i in range(1, 13))),
      Question("evidence_type", "Type?", (Option("fact"), Option("announcement"), Option("opinion")))]
STATE = "[S1] Leipzig approved a new motorway junction. [S2] The rail terminal closes."
ln = math.log


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("locassess.providers.calls.time.sleep", lambda s: None)


# --- OpenAI Decisions --------------------------------------------------------

class FakeDecisionsClient:
    def __init__(self, response):
        self.requests, self.response = [], response
        self.decisions = SimpleNamespace(create=self.create)

    def create(self, **kw):
        self.requests.append(kw)
        return SimpleNamespace(model_dump=lambda mode=None: self.response)


def test_openai_decisions_request_and_parse(conn):
    resp = {"model": "gpt-6-luna-2026", "usage": {"input_tokens": 300, "output_tokens": 0}, "answers": [
        {"type": "choice", "name": "direction", "choice": "1", "confidence": 0.4,
         "probabilities": [{"value": "-1", "probability": 0.1}, {"value": "0", "probability": 0.2},
                           {"value": "1", "probability": 0.7}]},
        {"type": "refusal", "name": "evidence_sentence"},
        {"type": "choice", "name": "evidence_type", "choice": "fact", "confidence": 0.9,
         "probabilities": [{"value": "fact", "probability": 0.95}, {"value": "announcement", "probability": 0.05}]}]}
    client = FakeDecisionsClient(resp)
    res = OpenAIDecisions(conn, client=client).decide(STATE, QS, "extraction_B", version="1.0")
    req = client.requests[0]
    assert req["model"] == "gpt-6-luna" and req["input"] == STATE
    assert req["questions"][0] == {"type": "choice", "name": "direction", "instructions": "Direction?", "choices": [
        {"value": "-1", "description": "less"}, {"value": "0", "description": "none"},
        {"value": "1", "description": "more"}]}
    assert req["questions"][1]["choices"][0] == {"value": "S1"}      # no empty descriptions
    assert res.distributions["direction"] == {"-1": 0.1, "0": 0.2, "1": 0.7}
    assert res.distributions["evidence_type"] == {"fact": 0.95, "announcement": 0.05, "opinion": 0.0}
    assert res.refusals == ["evidence_sentence"] and res.confidence["direction"] == 0.4
    assert res.model_version == "gpt-6-luna-2026" and len(res.call_ids) == 1
    row = conn.execute("SELECT provider, purpose, prompt_version, input_tokens FROM call_log").fetchone()
    assert tuple(row) == ("openai_decisions", "extraction_B", "1.0", 300)


def test_option_limits_checked(conn):
    too_many = [Question("x", "x", tuple(Option(f"S{i}") for i in range(256)))]
    with pytest.raises(ValueError, match="2..255"):
        OpenAIDecisions(conn, client=FakeDecisionsClient({})).decide(STATE, too_many, "t")
    with pytest.raises(ValueError, match="unique"):
        OpenAIDecisions(conn, client=FakeDecisionsClient({})).decide(
            STATE, [Question("x", "x", (Option("a"), Option("a")))], "t")


# --- Jev -----------------------------------------------------------------------

def test_jev_request_and_parse(conn):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"model": "jev-1.13.0", "usage": {"input_tokens": 50, "output_tokens": 0},
                                         "answers": {"direction": {"type": "choice", "choice": "-1", "confidence": 0.8,
                                                                   "probabilities": {"-1": 0.9, "0": 0.1, "1": 0.0}}}})

    jev = JevDecisions(conn, secrets=Secrets(typesafe_api_key="ts-key"),
                       http=httpx.Client(transport=httpx.MockTransport(handler)))
    res = jev.decide(STATE, QS, "extraction_B")
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://api.typesafe.ai/v1/systemone"
    assert seen[0].headers["Authorization"] == "Bearer ts-key"
    assert body["model"] == "jev-latest" and body["state"] == STATE
    assert body["questions"]["direction"] == {"type": "choice", "instructions": "Direction?",
                                              "criteria": {"-1": "less", "0": "none", "1": "more"}}
    assert body["questions"]["evidence_type"]["criteria"] == {"fact": None, "announcement": None, "opinion": None}
    assert res.distributions == {"direction": {"-1": 0.9, "0": 0.1, "1": 0.0}}
    assert res.refusals == ["evidence_sentence", "evidence_type"] and res.model_version == "jev-1.13.0"
    logged = " ".join(str(tuple(r)) for r in conn.execute("SELECT * FROM call_log"))
    assert "ts-key" not in logged


def test_jev_requires_key(conn, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ProviderNotConfigured, match="TYPESAFE_API_KEY"):
        JevDecisions(conn)


# --- llm_structured: logprobs ---------------------------------------------------

def tok(token, p, alts=()):
    return {"token": token, "logprob": ln(p), "top": [[token, ln(p)], *[[a, ln(q)] for a, q in alts]]}


# Token sequence recorded from a live gpt-4.1 call (structured output + logprobs, 2026-10-13), shortened;
# the alternatives for the sentence ID were added to exercise the ambiguous case.
LIVE = [tok('{"', 1.0), tok("direction", 1.0), tok('":"', 1.0),
        tok("-", 0.533, [("1", 0.384), ("0", 0.083)]), tok("1", 1.0),
        tok('","', 1.0), tok("evidence_sentence", 1.0), tok('":"', 1.0),
        tok("S", 1.0), tok("2", 0.9, [("1", 0.1)]),
        tok('","', 1.0), tok("evidence_type", 1.0), tok('":"', 1.0),
        tok("fact", 0.997, [("announcement", 0.003)]), tok('"}', 1.0)]


def live_response(tokens=LIVE):
    return {"output_text": "".join(t["token"] for t in tokens), "logprobs": tokens}


def test_logprobs_distribution_from_live_tokens():
    res = distributions_from_logprobs(live_response(), QS)
    d = res.distributions
    assert d["direction"] == pytest.approx({"-1": 0.533, "0": 0.083, "1": 0.384}, abs=1e-6)
    assert d["evidence_type"]["fact"] == pytest.approx(0.997)
    assert d["evidence_type"]["announcement"] == pytest.approx(0.003)
    # "1" after "S" fits S1, S10, S11, S12 → split equally and flagged as ambiguous
    assert d["evidence_sentence"]["S2"] == pytest.approx(0.9)
    assert d["evidence_sentence"]["S1"] == pytest.approx(0.025)
    assert d["evidence_sentence"]["S12"] == pytest.approx(0.025)
    assert res.meta["ambiguous"]["evidence_sentence"] == pytest.approx(0.1)
    assert res.meta["residual"]["direction"] == pytest.approx(0.0, abs=1e-6)


def test_logprobs_value_extension_at_closing_quote():
    # chosen "S1"; at the closing quote the model also considered continuing to "S12"
    tokens = [tok('{"', 1.0), tok("evidence_sentence", 1.0), tok('":"', 1.0), tok("S", 1.0), tok("1", 1.0),
              tok('"}', 0.8, [("2", 0.2)])]
    d = distributions_from_logprobs(live_response(tokens), [QS[1]]).distributions["evidence_sentence"]
    assert d["S1"] == pytest.approx(0.8) and d["S12"] == pytest.approx(0.2)


def test_logprobs_text_mismatch_raises():
    bad = {"output_text": '{"direction":"1"}', "logprobs": [tok('{"direction":"', 1.0)]}
    with pytest.raises(ValueError, match="reproduce"):
        distributions_from_logprobs(bad, [QS[0]])


def test_sampling_relative_frequency():
    samples = [{"direction": "1"}, {"direction": "1"}, {"direction": "-1"}, {"direction": "1"}, {"direction": "0"}]
    d = distributions_from_samples(samples, [QS[0]]).distributions["direction"]
    assert d == {"-1": 0.2, "0": 0.2, "1": 0.6}


class FakeResponsesClient:
    def __init__(self, outputs):
        self.outputs, self.requests = list(outputs), []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kw):
        self.requests.append(kw)
        text, tokens = self.outputs.pop(0)
        lp = [SimpleNamespace(token=t["token"], logprob=t["logprob"],
                              top_logprobs=[SimpleNamespace(token=a, logprob=l) for a, l in t["top"]])
              for t in tokens]
        part = SimpleNamespace(type="output_text", logprobs=lp)
        return SimpleNamespace(output_text=text, output=[SimpleNamespace(content=[part])], model="gpt-4.1-2026",
                               usage=SimpleNamespace(input_tokens=500, output_tokens=20))


def test_llm_structured_logprobs_request(conn):
    r = live_response()
    client = FakeResponsesClient([(r["output_text"], r["logprobs"])])
    res = LLMStructured(conn, "gpt-4.1", "logprobs", client=client).decide(STATE, QS, "extraction_B", "1.0")
    req = client.requests[0]
    assert req["temperature"] == 0 and req["top_logprobs"] == 20
    assert req["include"] == ["message.output_text.logprobs"]
    assert req["text"]["format"]["schema"] == answer_schema(QS)
    assert req["text"]["format"]["schema"]["properties"]["direction"]["enum"] == ["-1", "0", "1"]
    assert "### direction\nDirection?\nOptions:\n- -1: less" in req["input"] and STATE in req["input"]
    assert "Options: S1, S2, S3" in req["input"]
    assert res.distributions["direction"]["-1"] == pytest.approx(0.533)
    assert conn.execute("SELECT prompt_version FROM call_log").fetchone()[0] == "1.0+llm_structured_extraction@1.0"


def test_llm_structured_sampling(conn):
    outs = [(json.dumps({"direction": d, "evidence_sentence": "S1", "evidence_type": "fact"}), [])
            for d in ["1", "1", "0", "1", "-1"]]
    client = FakeResponsesClient(outs)
    res = LLMStructured(conn, "gpt-4.1", "sampling", sampling_n=5, client=client).decide(STATE, QS, "x")
    assert all(r["temperature"] == 1 and "top_logprobs" not in r for r in client.requests)
    assert res.distributions["direction"] == {"-1": 0.2, "0": 0.2, "1": 0.6} and len(res.call_ids) == 5


def test_all_adapters_receive_identical_questions(conn):
    """SPEC §6.3: same questions, options and texts for every decision model."""
    dec = FakeDecisionsClient({"answers": []})
    OpenAIDecisions(conn, client=dec).decide(STATE, QS, "t")
    jev_body = []

    def jev_handler(r):
        jev_body.append(json.loads(r.content))
        return httpx.Response(200, json={"answers": {}})

    JevDecisions(conn, secrets=Secrets(typesafe_api_key="k"),
                 http=httpx.Client(transport=httpx.MockTransport(jev_handler))).decide(STATE, QS, "t")
    r = live_response()
    llm = FakeResponsesClient([(r["output_text"], r["logprobs"])])
    LLMStructured(conn, "gpt-4.1", client=llm).decide(STATE, QS, "t")

    from_decisions = {q["name"]: (q["instructions"], [c["value"] for c in q["choices"]])
                      for q in dec.requests[0]["questions"]}
    from_jev = {k: (v["instructions"], list(v["criteria"])) for k, v in jev_body[0]["questions"].items()}
    schema = llm.requests[0]["text"]["format"]["schema"]["properties"]
    assert from_decisions == from_jev
    assert {k: v[1] for k, v in from_decisions.items()} == {k: v["enum"] for k, v in schema.items()}
    assert all(instr in llm.requests[0]["input"] for instr, _ in from_decisions.values())
