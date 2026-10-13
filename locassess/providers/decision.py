"""Decision-model interface (SPEC §6.3). Adapters arrive in milestone M4."""

from __future__ import annotations

from typing import Protocol


class DecisionModel(Protocol):
    name: str

    def decide(self, state: dict, questions: dict[str, list[str]]) -> dict[str, dict[str, float]]:
        """questions: {question_key: [option, ...]} → {question_key: {option: probability}}"""
        ...
