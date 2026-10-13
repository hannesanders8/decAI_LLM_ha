"""LLM interface (SPEC §6.1). Adapters (OpenAI, Anthropic) arrive with the stages that use them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResult:
    texts: list[str]                                   # one per sample (n)
    parsed: list[Any] = field(default_factory=list)    # parsed JSON when a schema was given
    logprobs: list[Any] | None = None
    model_version: str | None = None
    call_id: int | None = None


class LLM(Protocol):
    def complete(self, prompt_id: str, variables: dict, json_schema: dict | None = None,
                 temperature: float = 0, n: int = 1, logprobs: bool = False) -> LLMResult: ...
