"""LLM interface (SPEC §6.1) and the OpenAI adapter (Responses API)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Protocol

from locassess.config import Secrets
from locassess.prompts import load_prompt
from locassess.providers import ProviderNotConfigured
from locassess.providers.calls import CallOutcome, RateLimited, logged_call


@dataclass
class LLMResult:
    texts: list[str]                                   # one per sample (n)
    parsed: list[Any] = field(default_factory=list)    # parsed JSON when a schema was given
    logprobs: list[Any] | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    call_ids: list[int] = field(default_factory=list)


class LLM(Protocol):
    def complete(self, prompt_id: str, variables: dict, json_schema: dict | None = None,
                 temperature: float = 0, n: int = 1, logprobs: bool = False) -> LLMResult: ...


def is_rate_limit(exc: Exception) -> bool:
    try:
        import openai
        if isinstance(exc, openai.RateLimitError):
            return True
    except ImportError:  # pragma: no cover
        pass
    return getattr(exc, "status_code", None) == 429


class OpenAILLM:
    """OpenAI Responses API with optional JSON-schema structured output.

    Docs: https://developers.openai.com/api/docs/guides/structured-outputs
    """

    provider = "openai"

    def __init__(self, conn: sqlite3.Connection, model: str, client: Any = None,
                 secrets: Secrets | None = None, purpose: str = "llm"):
        self.conn, self.model, self.purpose = conn, model, purpose
        if client is None:
            key = (secrets or Secrets()).openai_api_key
            if key is None or not key.get_secret_value():
                raise ProviderNotConfigured("OPENAI_API_KEY is not set in .env")
            import openai
            client = openai.OpenAI(api_key=key.get_secret_value())
        self.client = client

    def complete(self, prompt_id: str, variables: dict, json_schema: dict | None = None,
                 temperature: float = 0, n: int = 1, logprobs: bool = False) -> LLMResult:
        if logprobs:
            raise NotImplementedError("token logprobs are added with the llm_structured adapter (M4)")
        prompt = load_prompt(prompt_id)
        request: dict[str, Any] = {"model": self.model, "input": prompt.render(variables),
                                   "temperature": temperature}
        if json_schema is not None:
            request["text"] = {"format": {"type": "json_schema", "name": prompt_id,
                                          "schema": json_schema, "strict": True}}

        result = LLMResult(texts=[], prompt_version=prompt.version)
        for _ in range(n):  # the Responses API returns one sample per request
            outcome, call_id = logged_call(
                self.conn, provider=self.provider, model=self.model, purpose=prompt_id,
                prompt_version=prompt.version, request=request, fn=lambda: self._call(request),
            )
            text = outcome.response["output_text"]
            result.texts.append(text)
            if json_schema is not None:
                result.parsed.append(json.loads(text))
            result.model_version = outcome.model_version
            result.call_ids.append(call_id)
        return result

    def _call(self, request: dict) -> CallOutcome:
        try:
            resp = self.client.responses.create(**request)
        except Exception as exc:
            if is_rate_limit(exc):
                raise RateLimited(str(exc)) from exc
            raise
        for item in getattr(resp, "output", None) or []:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "refusal":
                    raise RuntimeError(f"model refused: {getattr(part, 'refusal', '')}")
        usage = getattr(resp, "usage", None)
        return CallOutcome(
            response={"output_text": resp.output_text, "id": getattr(resp, "id", None)},
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            model_version=getattr(resp, "model", None),
        )


def make_llm(conn: sqlite3.Connection, provider: str, model: str) -> LLM:
    if provider == "openai":
        return OpenAILLM(conn, model)
    raise ProviderNotConfigured(f"LLM provider {provider!r} is not implemented yet")
