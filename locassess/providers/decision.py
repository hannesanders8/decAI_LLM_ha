"""Decision-model interface (SPEC §6.3) and adapters: openai_decisions, llm_structured, jev.

A decision model receives a state (the rendered article section) and a list of questions with
fixed answer options, and returns a probability for every option. All adapters receive exactly
the same questions, options and codebook text from one versioned query definition
(locassess/query_definition.py).
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from locassess.config import AppConfig, Secrets
from locassess.prompts import load_prompt
from locassess.providers import ProviderNotConfigured
from locassess.providers.calls import CallOutcome, RateLimited, logged_call
from locassess.providers.llm import is_rate_limit


@dataclass(frozen=True)
class Option:
    value: str
    description: str = ""


@dataclass(frozen=True)
class Question:
    key: str
    instructions: str
    options: tuple[Option, ...]

    @property
    def values(self) -> list[str]:
        return [o.value for o in self.options]


@dataclass
class DecisionResult:
    distributions: dict[str, dict[str, float]]          # question key → option → probability
    refusals: list[str] = field(default_factory=list)   # question keys the model declined
    confidence: dict[str, float] = field(default_factory=dict)  # provider-native confidence, if any
    call_ids: list[int] = field(default_factory=list)
    model_version: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)  # method details (e.g. logprob residual mass)

    def to_json(self) -> str:
        return json.dumps({"distributions": self.distributions, "refusals": self.refusals,
                           "confidence": self.confidence, "call_ids": self.call_ids,
                           "model_version": self.model_version, "meta": self.meta},
                          sort_keys=True, ensure_ascii=False)


class DecisionModel(Protocol):
    name: str
    confidence_method: str      # native | logprobs | sampling

    def decide(self, state: str, questions: list[Question], purpose: str,
               version: str | None = None) -> DecisionResult: ...


def _check_options(questions: list[Question], max_options: int = 255) -> None:
    for q in questions:
        values = q.values
        if not 2 <= len(values) <= max_options:
            raise ValueError(f"question {q.key}: needs 2..{max_options} options, got {len(values)}")
        if len(set(values)) != len(values):
            raise ValueError(f"question {q.key}: options must be unique")


# ---------------------------------------------------------------------------
# OpenAI Decisions API
# Docs: https://developers.openai.com/api/docs/guides/decisions (public beta, gpt-6-luna)
# ---------------------------------------------------------------------------

class OpenAIDecisions:
    name = "openai_decisions"
    confidence_method = "native"
    DEFAULT_MODEL = "gpt-6-luna"

    def __init__(self, conn: sqlite3.Connection, model: str | None = None, client: Any = None,
                 secrets: Secrets | None = None):
        self.conn, self.model = conn, model or self.DEFAULT_MODEL
        if client is None:
            key = (secrets or Secrets()).openai_api_key
            if key is None or not key.get_secret_value():
                raise ProviderNotConfigured("OPENAI_API_KEY is not set in .env")
            import openai
            client = openai.OpenAI(api_key=key.get_secret_value())
        self.client = client

    def decide(self, state: str, questions: list[Question], purpose: str,
               version: str | None = None) -> DecisionResult:
        _check_options(questions)
        request = {"model": self.model, "input": state, "questions": [
            {"type": "choice", "name": q.key, "instructions": q.instructions,
             "choices": [{"value": o.value, **({"description": o.description} if o.description else {})}
                         for o in q.options]}
            for q in questions]}
        outcome, call_id = logged_call(self.conn, provider=self.name, model=self.model, purpose=purpose,
                                       prompt_version=version, request=request, fn=lambda: self._call(request))
        result = parse_openai_decision(outcome.response, questions)
        result.call_ids, result.model_version = [call_id], outcome.model_version
        return result

    def _call(self, request: dict) -> CallOutcome:
        try:
            resp = self.client.decisions.create(**request)
        except Exception as exc:
            if is_rate_limit(exc):
                raise RateLimited(str(exc)) from exc
            raise
        data = resp.model_dump(mode="json") if hasattr(resp, "model_dump") else resp
        usage = data.get("usage") or {}
        return CallOutcome(response=data, input_tokens=usage.get("input_tokens"),
                           output_tokens=usage.get("output_tokens"), model_version=data.get("model"))


def parse_openai_decision(data: dict, questions: list[Question]) -> DecisionResult:
    by_name = {a.get("name"): a for a in data.get("answers") or []}
    result = DecisionResult(distributions={})
    for q in questions:
        a = by_name.get(q.key)
        if a is None or a.get("type") != "choice":
            result.refusals.append(q.key)
            continue
        probs = {str(p["value"]): float(p["probability"]) for p in a.get("probabilities") or []}
        result.distributions[q.key] = {v: probs.get(v, 0.0) for v in q.values}
        if a.get("confidence") is not None:
            result.confidence[q.key] = float(a["confidence"])
    return result


# ---------------------------------------------------------------------------
# TypeSafe Jev (System One)
# Docs: https://docs.typesafe.ai/introduction/quickstart, https://docs.typesafe.ai/primitives/choice
# Not tested against the live API (no access yet).
# ---------------------------------------------------------------------------

class JevDecisions:
    name = "jev"
    confidence_method = "native"
    ENDPOINT = "https://api.typesafe.ai/v1/systemone"
    DEFAULT_MODEL = "jev-latest"

    def __init__(self, conn: sqlite3.Connection, model: str | None = None, secrets: Secrets | None = None,
                 http: httpx.Client | None = None):
        key = (secrets or Secrets()).typesafe_api_key
        if key is None or not key.get_secret_value():
            raise ProviderNotConfigured("TYPESAFE_API_KEY is not set in .env (Jev access: https://console.typesafe.ai/)")
        self.conn, self.model, self._key = conn, model or self.DEFAULT_MODEL, key.get_secret_value()
        self.http = http or httpx.Client(timeout=120)

    def decide(self, state: str, questions: list[Question], purpose: str,
               version: str | None = None) -> DecisionResult:
        _check_options(questions)
        request = {"model": self.model, "state": state, "questions": {
            q.key: {"type": "choice", "instructions": q.instructions,
                    "criteria": {o.value: (o.description or None) for o in q.options}}
            for q in questions}}
        outcome, call_id = logged_call(self.conn, provider=self.name, model=self.model, purpose=purpose,
                                       prompt_version=version, request=request, fn=lambda: self._call(request))
        result = parse_jev(outcome.response, questions)
        result.call_ids, result.model_version = [call_id], outcome.model_version
        return result

    def _call(self, request: dict) -> CallOutcome:
        resp = self.http.post(self.ENDPOINT, json=request,
                              headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
        if resp.status_code == 429:
            raise RateLimited("jev: HTTP 429")
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        return CallOutcome(response=data, input_tokens=usage.get("input_tokens"),
                           output_tokens=usage.get("output_tokens"), model_version=data.get("model"))


def parse_jev(data: dict, questions: list[Question]) -> DecisionResult:
    answers = data.get("answers") or {}
    result = DecisionResult(distributions={})
    for q in questions:
        a = answers.get(q.key)
        if not a or a.get("type") != "choice" or not isinstance(a.get("probabilities"), dict):
            result.refusals.append(q.key)
            continue
        probs = a["probabilities"]
        result.distributions[q.key] = {v: float(probs.get(v, 0.0)) for v in q.values}
        if a.get("confidence") is not None:
            result.confidence[q.key] = float(a["confidence"])
    return result


# ---------------------------------------------------------------------------
# llm_structured: any chat LLM with JSON-schema output (fallback that must work end to end)
# Probabilities from token logprobs (temperature 0) or from sampling_n samples at temperature 1.
# ---------------------------------------------------------------------------

TOP_LOGPROBS = 20


def render_questions(questions: list[Question]) -> str:
    blocks = []
    for q in questions:
        if all(not o.description for o in q.options):
            opts = "Options: " + ", ".join(q.values)
        else:
            opts = "Options:\n" + "\n".join(f"- {o.value}: {o.description}" if o.description else f"- {o.value}"
                                           for o in q.options)
        blocks.append(f"### {q.key}\n{q.instructions}\n{opts}")
    return "\n\n".join(blocks)


def answer_schema(questions: list[Question]) -> dict:
    return {"type": "object",
            "properties": {q.key: {"type": "string", "enum": q.values} for q in questions},
            "required": [q.key for q in questions], "additionalProperties": False}


class LLMStructured:
    name = "llm_structured"

    def __init__(self, conn: sqlite3.Connection, model: str, method: str = "logprobs", sampling_n: int = 5,
                 client: Any = None, secrets: Secrets | None = None):
        if method not in ("logprobs", "sampling"):
            raise ValueError(f"unknown confidence method {method!r}")
        self.conn, self.model, self.method, self.sampling_n = conn, model, method, sampling_n
        self.confidence_method = method
        if client is None:
            key = (secrets or Secrets()).openai_api_key
            if key is None or not key.get_secret_value():
                raise ProviderNotConfigured("OPENAI_API_KEY is not set in .env")
            import openai
            client = openai.OpenAI(api_key=key.get_secret_value())
        self.client = client

    def decide(self, state: str, questions: list[Question], purpose: str,
               version: str | None = None) -> DecisionResult:
        _check_options(questions)
        prompt = load_prompt("llm_structured_extraction")
        text = prompt.render({"state": state, "questions": render_questions(questions)})
        base = {"model": self.model, "input": text,
                "text": {"format": {"type": "json_schema", "name": "answers",
                                    "schema": answer_schema(questions), "strict": True}}}
        pv = f"{version}+{prompt.id}@{prompt.version}" if version else prompt.version
        if self.method == "logprobs":
            request = {**base, "temperature": 0, "include": ["message.output_text.logprobs"],
                       "top_logprobs": TOP_LOGPROBS}
            outcome, call_id = logged_call(self.conn, provider="openai", model=self.model, purpose=purpose,
                                           prompt_version=pv, request=request, fn=lambda: self._call(request))
            result = distributions_from_logprobs(outcome.response, questions)
            result.call_ids, result.model_version = [call_id], outcome.model_version
            return result

        request = {**base, "temperature": 1}
        samples, call_ids, mv = [], [], None
        for _ in range(self.sampling_n):
            outcome, call_id = logged_call(self.conn, provider="openai", model=self.model, purpose=purpose,
                                           prompt_version=pv, request=request, fn=lambda: self._call(request))
            samples.append(json.loads(outcome.response["output_text"]))
            call_ids.append(call_id)
            mv = outcome.model_version
        result = distributions_from_samples(samples, questions)
        result.call_ids, result.model_version = call_ids, mv
        return result

    def _call(self, request: dict) -> CallOutcome:
        try:
            resp = self.client.responses.create(**request)
        except Exception as exc:
            if is_rate_limit(exc):
                raise RateLimited(str(exc)) from exc
            raise
        tokens = []
        for item in getattr(resp, "output", None) or []:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "refusal":
                    raise RuntimeError(f"model refused: {getattr(part, 'refusal', '')}")
                for t in getattr(part, "logprobs", None) or []:
                    tokens.append({"token": t.token, "logprob": t.logprob,
                                   "top": [[a.token, a.logprob] for a in (t.top_logprobs or [])]})
        usage = getattr(resp, "usage", None)
        return CallOutcome(response={"output_text": resp.output_text, "logprobs": tokens},
                           input_tokens=getattr(usage, "input_tokens", None),
                           output_tokens=getattr(usage, "output_tokens", None),
                           model_version=getattr(resp, "model", None))


def distributions_from_samples(samples: list[dict], questions: list[Question]) -> DecisionResult:
    result = DecisionResult(distributions={}, meta={"method": "sampling", "n": len(samples)})
    for q in questions:
        counts = Counter(str(s.get(q.key)) for s in samples)
        result.distributions[q.key] = {v: counts.get(v, 0) / len(samples) for v in q.values}
    return result


def distributions_from_logprobs(response: dict, questions: list[Question]) -> DecisionResult:
    """Option probabilities from the token logprobs of one JSON answer.

    For each question the chosen value is located in the output; walking its tokens, every
    alternative token (top_logprobs) is assigned to the option(s) it leads to. The chosen option
    gets the probability of the chosen path. If an alternative token fits several options (e.g.
    "1" for S1, S10, S11 … when S2 was chosen), its mass is split equally between them and
    recorded as `ambiguous`. Mass on tokens that fit no option, or outside the top alternatives,
    is recorded as `residual`; the distribution is renormalised over the options.
    """
    text: str = response["output_text"]
    tokens: list[dict] = response.get("logprobs") or []
    if "".join(t["token"] for t in tokens) != text:
        raise ValueError("token logprobs do not reproduce the output text; cannot derive probabilities")
    starts, pos = [], 0
    for t in tokens:
        starts.append(pos)
        pos += len(t["token"])
    answers = json.loads(text)

    result = DecisionResult(distributions={}, meta={"method": "logprobs", "top_logprobs": TOP_LOGPROBS,
                                                    "residual": {}, "ambiguous": {}})
    for q in questions:
        value = str(answers[q.key])
        m = re.search(rf'"{re.escape(q.key)}"\s*:\s*"', text)
        if m is None:
            raise ValueError(f"answer for {q.key} not found in output")
        probs, ambiguous = _walk_value(tokens, starts, text, m.end(), value, q.values)
        total = sum(probs.values())
        result.meta["residual"][q.key] = max(0.0, 1.0 - total)
        if ambiguous > 0:
            result.meta["ambiguous"][q.key] = ambiguous
        result.distributions[q.key] = {v: (p / total if total > 0 else 0.0) for v, p in probs.items()}
    return result


def _walk_value(tokens, starts, text, value_start, value, options) -> tuple[dict[str, float], float]:
    probs = {o: 0.0 for o in options}
    ambiguous = 0.0
    path_p = 1.0
    i = next(k for k in range(len(tokens)) if starts[k] <= value_start < starts[k] + len(tokens[k]["token"]))
    pos = value_start
    while i < len(tokens):
        tok = tokens[i]["token"]
        prefix = text[starts[i]:pos]                 # characters of this token before the current position
        done = text[value_start:pos]                 # part of the value already fixed on the chosen path
        chosen_rest = tok[len(prefix):]
        for alt, lp in tokens[i]["top"]:
            if not alt.startswith(prefix):
                continue
            rest = alt[len(prefix):]
            if rest == chosen_rest:
                continue
            alt_val, terminated = rest.split('"', 1)[0], '"' in rest
            cand = [o for o in options if (o == done + alt_val if terminated else o.startswith(done + alt_val))]
            p = path_p * math.exp(lp)
            if len(cand) == 1:
                probs[cand[0]] += p
            elif cand:
                ambiguous += p
                for o in cand:
                    probs[o] += p / len(cand)
        path_p *= math.exp(tokens[i]["logprob"])
        if '"' in chosen_rest:                      # the chosen token closes the value
            break
        pos = starts[i] + len(tok)
        i += 1
    if value in probs:
        probs[value] += path_p
    return probs, ambiguous


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_decision_model(conn: sqlite3.Connection, cfg: AppConfig, name: str | None = None) -> DecisionModel:
    name = name or cfg.models.decision_provider
    if name == "openai_decisions":
        return OpenAIDecisions(conn, cfg.models.decision_model)
    if name == "jev":
        return JevDecisions(conn, cfg.models.decision_model)
    if name == "llm_structured":
        if cfg.models.llm_provider != "openai":
            raise ProviderNotConfigured("llm_structured is implemented for the OpenAI provider only")
        return LLMStructured(conn, cfg.models.llm_model, cfg.models.llm_confidence_method, cfg.models.sampling_n)
    raise ProviderNotConfigured(f"unknown decision provider {name!r}")
