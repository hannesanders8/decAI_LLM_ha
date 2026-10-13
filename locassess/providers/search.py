"""Search interface (SPEC §6.2). Adapters arrive in milestone M3."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol


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
