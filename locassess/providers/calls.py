"""Single wrapper for every external call: retry with backoff, redaction, call_log row."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from locassess.db.repo import CallRecord, insert_call_log

REDACTED = "***"
_SECRET_KEY_RE = re.compile(r"(api[_-]?key|authorization|access[_-]?token|secret|password)", re.I)


class RateLimited(Exception):
    """Raised by an adapter when the provider signals a rate limit; triggers a retry."""


@dataclass
class CallOutcome:
    response: Any
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    model_version: str | None = None


@lru_cache(maxsize=1)
def _env_secrets() -> tuple[str, ...]:
    from locassess.config import Secrets

    return tuple(Secrets().values())


def redact(obj: Any, secrets: tuple[str, ...] | list[str] = ()) -> Any:
    """Drop secret-looking keys and mask every literal secret value, recursively."""
    if isinstance(obj, dict):
        return {
            k: REDACTED if _SECRET_KEY_RE.search(str(k)) else redact(v, secrets) for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [redact(v, secrets) for v in obj]
    if isinstance(obj, str):
        for s in secrets:
            if s:
                obj = obj.replace(s, REDACTED)
        return obj
    return obj


def _to_json(obj: Any, secrets: tuple[str, ...] | list[str]) -> str:
    # Serialise first so non-JSON objects become strings, then redact the plain structure.
    plain = json.loads(json.dumps(obj, default=str, ensure_ascii=False))
    return json.dumps(redact(plain, secrets), ensure_ascii=False, sort_keys=True)


def logged_call(
    conn: sqlite3.Connection,
    *,
    provider: str,
    model: str | None,
    purpose: str,
    request: Any,
    fn: Callable[[], CallOutcome],
    prompt_version: str | None = None,
    model_version: str | None = None,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    secrets: tuple[str, ...] | list[str] | None = None,
) -> tuple[CallOutcome, int]:
    """Run `fn`, retrying on RateLimited with exponential backoff, and write one call_log row.

    Returns (outcome, call_id). On final failure the row has `error` set and the exception
    is re-raised.
    """
    secrets = _env_secrets() if secrets is None else tuple(secrets)
    request_json = _to_json(request, secrets)
    attempt = 0
    while True:
        attempt += 1
        start = time.perf_counter()
        try:
            outcome = fn()
        except RateLimited as exc:
            if attempt < max_attempts:
                sleep(base_delay * 2 ** (attempt - 1))
                continue
            _log_error(conn, provider, model, model_version, purpose, prompt_version,
                       request_json, start, f"rate limited after {attempt} attempts: {exc}", secrets)
            raise
        except Exception as exc:
            _log_error(conn, provider, model, model_version, purpose, prompt_version,
                       request_json, start, f"{type(exc).__name__}: {exc}", secrets)
            raise
        latency_ms = int((time.perf_counter() - start) * 1000)
        call_id = insert_call_log(conn, CallRecord(
            provider=provider, model=model, purpose=purpose, prompt_version=prompt_version,
            model_version=outcome.model_version or model_version,
            request_json=request_json, response_json=_to_json(outcome.response, secrets),
            input_tokens=outcome.input_tokens, output_tokens=outcome.output_tokens,
            cost_usd=outcome.cost_usd, latency_ms=latency_ms,
        ))
        return outcome, call_id


def _log_error(conn, provider, model, model_version, purpose, prompt_version,
               request_json, start, message, secrets) -> None:
    insert_call_log(conn, CallRecord(
        provider=provider, model=model, model_version=model_version, purpose=purpose,
        prompt_version=prompt_version, request_json=request_json,
        latency_ms=int((time.perf_counter() - start) * 1000),
        error=redact(message, secrets),
    ))
