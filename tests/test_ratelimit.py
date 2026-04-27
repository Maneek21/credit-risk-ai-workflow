"""Tests for workflow.ratelimit."""
from __future__ import annotations

import time
from typing import List
from unittest.mock import MagicMock

import pytest

from workflow.ratelimit import (
    CircuitBreaker,
    CostTracker,
    LLMUnavailableError,
    RateLimiter,
    TokenBucket,
    with_retry,
)


# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------


def test_token_bucket_initial_capacity_consumable() -> None:
    bucket = TokenBucket(capacity=5, refill_per_sec=1.0)
    for _ in range(5):
        assert bucket.consume(1.0, block=False) is True


def test_token_bucket_non_blocking_returns_false_over_budget() -> None:
    bucket = TokenBucket(capacity=2, refill_per_sec=0.001)
    assert bucket.consume(2.0, block=False) is True
    # Bucket now empty; non-blocking call must return False, not sleep.
    t0 = time.monotonic()
    result = bucket.consume(1.0, block=False)
    elapsed = time.monotonic() - t0
    assert result is False
    assert elapsed < 0.05  # truly non-blocking


def test_token_bucket_blocking_sleeps_until_refill() -> None:
    # Refill 100 tokens / second → 10 ms per token.
    bucket = TokenBucket(capacity=1, refill_per_sec=100.0)
    assert bucket.consume(1.0, block=False) is True
    t0 = time.monotonic()
    assert bucket.consume(1.0, block=True) is True
    elapsed = time.monotonic() - t0
    # Should have waited roughly 10 ms; allow generous slack for CI jitter.
    assert elapsed >= 0.005
    assert elapsed < 1.0


def test_token_bucket_refill_caps_at_capacity() -> None:
    bucket = TokenBucket(capacity=3, refill_per_sec=1000.0)
    # Drain.
    assert bucket.consume(3.0, block=False) is True
    time.sleep(0.05)  # plenty of time to "overfill"
    # Can take exactly capacity, no more.
    assert bucket.consume(3.0, block=False) is True
    assert bucket.consume(0.5, block=False) is False


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


def test_circuit_breaker_starts_closed() -> None:
    cb = CircuitBreaker(failure_threshold=3, reset_timeout_sec=10.0)
    assert cb.is_open() is False
    assert cb.state == CircuitBreaker.CLOSED


def test_circuit_breaker_trips_after_threshold_failures() -> None:
    cb = CircuitBreaker(failure_threshold=3, reset_timeout_sec=10.0)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open() is False  # 2 < 3
    cb.record_failure()
    assert cb.is_open() is True
    assert cb.state == CircuitBreaker.OPEN


def test_circuit_breaker_success_resets_failure_counter() -> None:
    cb = CircuitBreaker(failure_threshold=3, reset_timeout_sec=10.0)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    cb.record_failure()
    # Only 2 consecutive failures after the reset → still closed.
    assert cb.is_open() is False


def test_circuit_breaker_half_opens_after_timeout() -> None:
    cb = CircuitBreaker(failure_threshold=2, reset_timeout_sec=0.05)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open() is True
    time.sleep(0.08)
    # is_open() lazily transitions OPEN → HALF_OPEN, so it now reads False.
    assert cb.is_open() is False
    assert cb.state == CircuitBreaker.HALF_OPEN


def test_circuit_breaker_half_open_success_closes() -> None:
    cb = CircuitBreaker(failure_threshold=2, reset_timeout_sec=0.05)
    cb.record_failure()
    cb.record_failure()
    time.sleep(0.08)
    assert cb.state == CircuitBreaker.HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitBreaker.CLOSED


def test_circuit_breaker_half_open_failure_reopens() -> None:
    cb = CircuitBreaker(failure_threshold=2, reset_timeout_sec=0.05)
    cb.record_failure()
    cb.record_failure()
    time.sleep(0.08)
    assert cb.state == CircuitBreaker.HALF_OPEN
    cb.record_failure()
    assert cb.is_open() is True


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


def _prices() -> dict:
    # USD per 1M tokens.
    return {
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "claude-opus-4-7": {"input": 15.0, "output": 75.0},
    }


def test_cost_tracker_records_correctly() -> None:
    ct = CostTracker(prices=_prices(), hard_cap_usd=100.0)
    # 1M in @ $2.5 + 1M out @ $10.0 = $12.50
    cost = ct.record("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(12.5)
    assert ct.total_spend == pytest.approx(12.5)


def test_cost_tracker_running_total_accumulates() -> None:
    ct = CostTracker(prices=_prices(), hard_cap_usd=100.0)
    ct.record("gpt-4o", 500_000, 500_000)  # $1.25 + $5.0 = $6.25
    ct.record("gpt-4o", 500_000, 500_000)
    assert ct.total_spend == pytest.approx(12.5)


def test_cost_tracker_raises_after_hard_cap() -> None:
    ct = CostTracker(prices=_prices(), hard_cap_usd=10.0)
    # Cost: 0.5M*$2.5 + 0.5M*$10 = $1.25 + $5 = $6.25 — under cap.
    ct.record("gpt-4o", 500_000, 500_000)
    # Next call: another $6.25 → total $12.5 > $10 cap → raise.
    with pytest.raises(LLMUnavailableError):
        ct.record("gpt-4o", 500_000, 500_000)


def test_cost_tracker_alert_fires_exactly_once() -> None:
    callback = MagicMock()
    ct = CostTracker(
        prices=_prices(),
        hard_cap_usd=100.0,
        alert_at_usd=5.0,
        alert_callback=callback,
    )
    # First call: $1.25 — below alert.
    ct.record("gpt-4o", 500_000, 0)
    assert callback.call_count == 0
    # Second call pushes total to $6.25 → crosses 5.0 → alert fires.
    ct.record("gpt-4o", 500_000, 500_000)
    assert callback.call_count == 1
    args = callback.call_args.args
    # callback(cost, total).
    assert args[0] == pytest.approx(6.25)
    assert args[1] == pytest.approx(7.5)
    # Third call: total now well past 5.0, but alert must NOT re-fire.
    ct.record("gpt-4o", 100_000, 0)
    assert callback.call_count == 1


def test_cost_tracker_unknown_model_raises_keyerror() -> None:
    ct = CostTracker(prices=_prices(), hard_cap_usd=10.0)
    with pytest.raises(KeyError):
        ct.record("nonexistent-model", 100, 100)


# ---------------------------------------------------------------------------
# with_retry
# ---------------------------------------------------------------------------


def test_with_retry_succeeds_after_two_429s() -> None:
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("HTTP 429: rate limit exceeded")
        return "ok"

    out = with_retry(flaky, max_retries=3, base_delay=0.001)
    assert out == "ok"
    assert calls["n"] == 3


def test_with_retry_exhausts_and_raises() -> None:
    calls = {"n": 0}

    def always_429() -> str:
        calls["n"] += 1
        raise RuntimeError("HTTP 429 too many requests")

    with pytest.raises(LLMUnavailableError):
        with_retry(always_429, max_retries=2, base_delay=0.001)
    assert calls["n"] == 3  # initial + 2 retries


def test_with_retry_propagates_non_transient() -> None:
    calls = {"n": 0}

    def bad_request() -> str:
        calls["n"] += 1
        raise ValueError("HTTP 400 bad request — your prompt is malformed")

    with pytest.raises(ValueError):
        with_retry(bad_request, max_retries=3, base_delay=0.001)
    # Non-transient error → no retries.
    assert calls["n"] == 1


def test_with_retry_propagates_llm_unavailable() -> None:
    """LLMUnavailableError is terminal — never retried even if it
    contains a 429-looking string."""
    calls: List[int] = []

    def boom() -> str:
        calls.append(1)
        raise LLMUnavailableError("429 rate cap")

    with pytest.raises(LLMUnavailableError):
        with_retry(boom, max_retries=3, base_delay=0.001)
    assert len(calls) == 1


def test_with_retry_recognizes_timeout_and_503() -> None:
    seq = iter(
        [
            RuntimeError("connection timeout"),
            RuntimeError("503 service unavailable"),
            "ok",
        ]
    )

    def flaky() -> str:
        v = next(seq)
        if isinstance(v, BaseException):
            raise v
        return v

    assert with_retry(flaky, max_retries=3, base_delay=0.001) == "ok"


# ---------------------------------------------------------------------------
# RateLimiter integration
# ---------------------------------------------------------------------------


def test_rate_limiter_acquire_and_record_success() -> None:
    rl = RateLimiter(
        rpm=600,  # 10/sec — fast enough for tests
        tpm=600_000,
        prices=_prices(),
        hard_cap_usd=10.0,
    )
    rl.acquire(model="gpt-4o", expected_input_tokens=100)
    cost = rl.record_call("gpt-4o", 100, 50, success=True)
    assert cost is not None
    assert cost > 0
    assert rl.cost is not None
    assert rl.cost.total_spend == pytest.approx(cost)


def test_rate_limiter_open_circuit_blocks_acquire() -> None:
    rl = RateLimiter(
        rpm=600,
        tpm=600_000,
        prices=_prices(),
        hard_cap_usd=10.0,
        failure_threshold=2,
    )
    rl.record_call("gpt-4o", 0, 0, success=False)
    rl.record_call("gpt-4o", 0, 0, success=False)
    with pytest.raises(LLMUnavailableError):
        rl.acquire(model="gpt-4o", expected_input_tokens=10)


def test_rate_limiter_no_prices_skips_cost() -> None:
    rl = RateLimiter(rpm=600, tpm=600_000, prices=None)
    rl.acquire(model="gpt-4o", expected_input_tokens=10)
    assert rl.record_call("gpt-4o", 10, 10, success=True) is None
