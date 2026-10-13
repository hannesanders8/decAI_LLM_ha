import pytest

from locassess.providers.calls import CallOutcome, RateLimited, logged_call

FAKE_KEY = "sk-fake-1234567890abcdef"


def _rows(conn):
    return [dict(r) for r in conn.execute("SELECT * FROM call_log ORDER BY call_id")]


def test_success_is_logged(conn):
    outcome, call_id = logged_call(
        conn, provider="openai", model="m", purpose="test", prompt_version="v1",
        request={"prompt": "hi"}, secrets=[],
        fn=lambda: CallOutcome({"text": "ok"}, input_tokens=3, output_tokens=1, cost_usd=0.01,
                               model_version="m-2026"),
    )
    assert outcome.response == {"text": "ok"}
    [row] = _rows(conn)
    assert row["call_id"] == call_id
    assert (row["provider"], row["purpose"], row["prompt_version"]) == ("openai", "test", "v1")
    assert (row["input_tokens"], row["output_tokens"], row["cost_usd"]) == (3, 1, 0.01)
    assert row["model_version"] == "m-2026"
    assert row["latency_ms"] >= 0 and row["error"] is None and row["created_at"]


def test_rate_limit_retries_with_backoff(conn):
    attempts, sleeps = [], []

    def fn():
        attempts.append(1)
        if len(attempts) < 3:
            raise RateLimited("429")
        return CallOutcome("ok")

    logged_call(conn, provider="p", model="m", purpose="t", request={}, fn=fn,
                base_delay=0.5, sleep=sleeps.append, secrets=[])
    assert len(attempts) == 3
    assert sleeps == [0.5, 1.0]
    [row] = _rows(conn)
    assert row["error"] is None


def test_rate_limit_exhausted_logs_error(conn):
    def fn():
        raise RateLimited("429")

    with pytest.raises(RateLimited):
        logged_call(conn, provider="p", model="m", purpose="t", request={}, fn=fn,
                    max_attempts=2, sleep=lambda s: None, secrets=[])
    [row] = _rows(conn)
    assert "rate limited after 2 attempts" in row["error"]


def test_other_errors_logged_and_reraised(conn):
    def fn():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        logged_call(conn, provider="p", model="m", purpose="t", request={}, fn=fn, secrets=[])
    [row] = _rows(conn)
    assert row["error"] == "ValueError: boom"
    assert row["response_json"] is None


def test_api_keys_never_logged(conn):
    request = {"headers": {"Authorization": f"Bearer {FAKE_KEY}"}, "api_key": FAKE_KEY,
               "prompt": f"echo {FAKE_KEY}"}
    logged_call(conn, provider="p", model="m", purpose="t", request=request, secrets=[FAKE_KEY],
                fn=lambda: CallOutcome({"echo": FAKE_KEY}))

    def fail():
        raise RuntimeError(f"invalid key {FAKE_KEY}")

    with pytest.raises(RuntimeError):
        logged_call(conn, provider="p", model="m", purpose="t", request=request,
                    secrets=[FAKE_KEY], fn=fail)
    for row in _rows(conn):
        assert all(FAKE_KEY not in str(v) for v in row.values())
