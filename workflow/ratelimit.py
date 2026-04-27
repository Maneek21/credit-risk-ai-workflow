"""Rate limiting, circuit breaking, and cost tracking for LLM calls.

This module provides production-grade safety primitives for wrapping
external LLM API calls. The pieces compose together via :class:`RateLimiter`,
but each is independently usable and unit-tested.

Components:
    * :class:`TokenBucket` — classic token-bucket rate limiter.
    * :class:`CircuitBreaker` — three-state breaker (CLOSED/OPEN/HALF_OPEN).
    * :class:`CostTracker` — running USD spend with alert + hard cap.
    * :class:`RateLimiter` — thin facade composing all three plus retry.
    * :class:`LLMUnavailableError` — terminal "do not retry" exception.

Stdlib-only (``threading``, ``time``, ``random``, ``re``).
"""

from __future__ import annotations

import random
import re
import threading
import time
from typing import Any, Callable, Dict, Optional


class LLMUnavailableError(Exception):
    """Raised when the LLM call must not proceed.

    This is *terminal* — circuit breaker open, hard cost cap exceeded,
    or retries exhausted. Callers should surface to the user, not retry.
    """


class TokenBucket:
    """Thread-safe token-bucket rate limiter.

    Tokens accrue continuously at ``refill_per_sec`` up to ``capacity``.
    Each :meth:`consume` call withdraws tokens; if insufficient, the call
    either blocks (sleeping until enough have refilled) or returns False.

    Attributes:
        capacity: Maximum tokens the bucket can hold.
        refill_per_sec: Tokens added per second.
    """

    def __init__(self, capacity: float, refill_per_sec: float) -> None:
        """Initialize a full bucket.

        Args:
            capacity: Maximum bucket size (also the starting token count).
            refill_per_sec: Token accrual rate in tokens/second.
        """
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if refill_per_sec <= 0:
            raise ValueError("refill_per_sec must be > 0")
        self.capacity = float(capacity)
        self.refill_per_sec = float(refill_per_sec)
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        """Top up tokens based on elapsed wall-clock time. Caller holds lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(
                self.capacity, self._tokens + elapsed * self.refill_per_sec
            )
            self._last_refill = now

    def consume(self, tokens: float = 1.0, block: bool = True) -> bool:
        """Withdraw ``tokens`` from the bucket.

        Args:
            tokens: Amount to consume. May exceed capacity only when
                ``block=True``; the call will sleep until enough tokens
                accrue (clamped — request larger than capacity will be
                served once the bucket fills, but never atomically).
            block: If True, sleep until tokens are available and return True.
                If False, return False immediately when over budget.

        Returns:
            True on successful consumption, False only when ``block=False``
            and the bucket is short.
        """
        if tokens <= 0:
            return True
        # Clamp asks-larger-than-capacity to the capacity, otherwise we'd
        # sleep forever waiting for impossible state.
        request = min(tokens, self.capacity)
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= request:
                    self._tokens -= request
                    return True
                if not block:
                    return False
                deficit = request - self._tokens
                wait = deficit / self.refill_per_sec
            # Sleep outside the lock so other threads can refill / observe.
            time.sleep(wait)


class CircuitBreaker:
    """Three-state circuit breaker for an unreliable downstream.

    State machine::

        CLOSED  --(N consecutive failures)-->  OPEN
        OPEN    --(reset_timeout elapses)  -->  HALF_OPEN
        HALF_OPEN --(success)              -->  CLOSED
        HALF_OPEN --(failure)              -->  OPEN

    Attributes:
        failure_threshold: Consecutive failures that trip the breaker.
        reset_timeout_sec: Seconds OPEN before a HALF_OPEN trial is allowed.
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_sec: float = 60.0,
    ) -> None:
        """Initialize a CLOSED breaker.

        Args:
            failure_threshold: Consecutive failures to OPEN.
            reset_timeout_sec: Cooldown before HALF_OPEN trial.
        """
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be > 0")
        if reset_timeout_sec <= 0:
            raise ValueError("reset_timeout_sec must be > 0")
        self.failure_threshold = int(failure_threshold)
        self.reset_timeout_sec = float(reset_timeout_sec)
        self._state = self.CLOSED
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Current breaker state, lazily transitioning OPEN -> HALF_OPEN."""
        with self._lock:
            self._maybe_half_open_locked()
            return self._state

    def _maybe_half_open_locked(self) -> None:
        """If OPEN past reset window, lazily promote to HALF_OPEN."""
        if (
            self._state == self.OPEN
            and self._opened_at is not None
            and (time.monotonic() - self._opened_at) >= self.reset_timeout_sec
        ):
            self._state = self.HALF_OPEN

    def is_open(self) -> bool:
        """Return True iff the breaker is currently blocking calls.

        HALF_OPEN counts as *not* open — a trial call is permitted.
        """
        with self._lock:
            self._maybe_half_open_locked()
            return self._state == self.OPEN

    def record_success(self) -> None:
        """Record a successful downstream call.

        Resets failure counter; promotes HALF_OPEN to CLOSED.
        """
        with self._lock:
            self._consecutive_failures = 0
            if self._state in (self.HALF_OPEN, self.OPEN):
                self._state = self.CLOSED
                self._opened_at = None

    def record_failure(self, exception: Optional[BaseException] = None) -> None:
        """Record a failed downstream call.

        Args:
            exception: Optional exception for caller-side logging. Not
                inspected here — the breaker is exception-agnostic.
        """
        del exception  # unused; kept for API symmetry
        with self._lock:
            if self._state == self.HALF_OPEN:
                # Trial failed → straight back to OPEN.
                self._state = self.OPEN
                self._opened_at = time.monotonic()
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._state = self.OPEN
                self._opened_at = time.monotonic()


class CostTracker:
    """Running USD spend tracker with alert threshold and hard cap.

    Prices are USD per 1M tokens, separated by input vs. output, e.g.::

        prices = {
            "gpt-4o":            {"input": 2.5,  "output": 10.0},
            "claude-opus-4-7":   {"input": 15.0, "output": 75.0},
        }

    Attributes:
        prices: Per-model price book (USD per 1M tokens, input/output).
        hard_cap_usd: Total spend that, once exceeded, raises
            :class:`LLMUnavailableError` on every subsequent record.
        alert_at_usd: Optional warning threshold; ``alert_callback`` fires
            exactly once when total first crosses this.
    """

    def __init__(
        self,
        prices: Dict[str, Dict[str, float]],
        hard_cap_usd: float,
        alert_at_usd: Optional[float] = None,
        alert_callback: Optional[Callable[[float, float], None]] = None,
    ) -> None:
        """Initialize a zero-spend tracker.

        Args:
            prices: ``{model: {"input": $/1M, "output": $/1M}}``.
            hard_cap_usd: Spend ceiling; further calls raise.
            alert_at_usd: Optional soft alert threshold.
            alert_callback: ``callback(call_cost, running_total)`` invoked
                once when ``running_total`` first crosses ``alert_at_usd``.
        """
        self.prices = dict(prices) if prices else {}
        self.hard_cap_usd = float(hard_cap_usd)
        self.alert_at_usd = (
            float(alert_at_usd) if alert_at_usd is not None else None
        )
        self.alert_callback = alert_callback
        self._total = 0.0
        self._alerted = False
        self._lock = threading.Lock()

    @property
    def total_spend(self) -> float:
        """Running USD total of all recorded calls."""
        with self._lock:
            return self._total

    def _price_for(self, model: str) -> Dict[str, float]:
        """Return the price entry for ``model`` or raise KeyError."""
        if model not in self.prices:
            raise KeyError(f"no price configured for model: {model!r}")
        return self.prices[model]

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record a single call's token usage and return its USD cost.

        Args:
            model: Price-book key.
            input_tokens: Prompt + system tokens billed at input rate.
            output_tokens: Completion tokens billed at output rate.

        Returns:
            USD cost of this call.

        Raises:
            LLMUnavailableError: If the *new* running total strictly exceeds
                ``hard_cap_usd``. The cost is still added before raising so
                spend ledgers stay consistent.
            KeyError: If ``model`` has no price entry.
        """
        price = self._price_for(model)
        cost = (
            (input_tokens / 1_000_000.0) * price["input"]
            + (output_tokens / 1_000_000.0) * price["output"]
        )
        # Determine whether this call crosses the alert threshold and/or
        # blows the hard cap. We do the bookkeeping under the lock, then
        # fire the side-effects (callback, raise) outside it.
        fire_alert = False
        new_total: float
        with self._lock:
            self._total += cost
            new_total = self._total
            if (
                self.alert_at_usd is not None
                and not self._alerted
                and new_total >= self.alert_at_usd
            ):
                self._alerted = True
                fire_alert = True
            over_cap = new_total > self.hard_cap_usd

        if fire_alert and self.alert_callback is not None:
            try:
                self.alert_callback(cost, new_total)
            except Exception:  # noqa: BLE001 - alert is best-effort
                pass

        if over_cap:
            raise LLMUnavailableError(
                f"cost cap exceeded: ${new_total:.4f} > ${self.hard_cap_usd:.2f}"
            )
        return cost


# Substrings (case-insensitive) we treat as transient + retryable.
_RETRYABLE_PATTERNS = re.compile(
    r"(429|rate[ _-]?limit|rate|503|504|timeout|temporarily|unavailable)",
    re.IGNORECASE,
)


def _is_transient(exc: BaseException) -> bool:
    """Heuristic: does this exception look like a transient API blip?"""
    if isinstance(exc, LLMUnavailableError):
        return False
    return bool(_RETRYABLE_PATTERNS.search(str(exc)))


def with_retry(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Call ``fn`` with exponential backoff on transient failures.

    Transient = exception ``str(e)`` contains one of: ``429``, ``rate``,
    ``503``, ``504``, ``timeout``, ``temporarily``, ``unavailable``.
    Non-transient exceptions propagate immediately.

    Args:
        fn: Callable to invoke.
        *args: Positional args forwarded to ``fn``.
        max_retries: Total attempts beyond the first. ``max_retries=3``
            means up to 4 total calls.
        base_delay: Seconds for the first backoff. Subsequent waits are
            ``base_delay * 2**attempt + uniform(0, base_delay)`` jitter.
        **kwargs: Keyword args forwarded to ``fn``.

    Returns:
        Whatever ``fn`` returns on its first success.

    Raises:
        LLMUnavailableError: If all retries are exhausted on transient
            errors. The original exception is chained via ``__cause__``.
        Exception: Any non-transient exception from ``fn`` propagates
            unchanged on the first failure.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except LLMUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - we re-raise selectively
            if not _is_transient(exc):
                raise
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
            time.sleep(delay)
    raise LLMUnavailableError(
        f"transient failures exhausted after {max_retries + 1} attempts: {last_exc}"
    ) from last_exc


class RateLimiter:
    """Composes a token bucket, circuit breaker, and cost tracker.

    Workflow per call::

        rl.acquire(model, expected_input_tokens)   # may block / raise
        try:
            response = your_llm_sdk_call(...)
        except Exception as e:
            rl.record_call(model, in_tok, 0, success=False)
            raise
        rl.record_call(model, in_tok, out_tok, success=True)

    Two token buckets gate the call:
        * RPM bucket — 1 token per call.
        * TPM bucket — ``expected_input_tokens`` per call (estimated upfront).

    Attributes:
        rpm_bucket: Requests-per-minute bucket (capacity = ``rpm``).
        tpm_bucket: Tokens-per-minute bucket (capacity = ``tpm``).
        circuit: Three-state breaker.
        cost: Running USD tracker.
    """

    def __init__(
        self,
        rpm: int = 60,
        tpm: int = 90_000,
        prices: Optional[Dict[str, Dict[str, float]]] = None,
        hard_cap_usd: float = 30.0,
        failure_threshold: int = 5,
    ) -> None:
        """Initialize composed rate limiter.

        Args:
            rpm: Requests per minute.
            tpm: Tokens per minute.
            prices: Price book; see :class:`CostTracker`. None disables
                cost tracking entirely (useful for tests).
            hard_cap_usd: Spend cap forwarded to :class:`CostTracker`.
            failure_threshold: Forwarded to :class:`CircuitBreaker`.
        """
        # Refill one period's worth of capacity over 60 seconds.
        self.rpm_bucket = TokenBucket(capacity=rpm, refill_per_sec=rpm / 60.0)
        self.tpm_bucket = TokenBucket(capacity=tpm, refill_per_sec=tpm / 60.0)
        self.circuit = CircuitBreaker(failure_threshold=failure_threshold)
        self.cost: Optional[CostTracker] = (
            CostTracker(prices=prices, hard_cap_usd=hard_cap_usd)
            if prices is not None
            else None
        )

    def acquire(self, model: str, expected_input_tokens: int) -> None:
        """Block until budget is available; raise if the breaker is open.

        Args:
            model: Model key (informational; circuit is per-limiter, not
                per-model).
            expected_input_tokens: Conservative upfront estimate of the
                prompt size, withdrawn from the TPM bucket.

        Raises:
            LLMUnavailableError: Circuit breaker is OPEN.
        """
        del model  # accepted for API symmetry / future per-model breakers
        if self.circuit.is_open():
            raise LLMUnavailableError(
                "circuit breaker is OPEN; refusing call"
            )
        self.rpm_bucket.consume(1.0, block=True)
        if expected_input_tokens > 0:
            self.tpm_bucket.consume(float(expected_input_tokens), block=True)

    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        success: bool,
    ) -> Optional[float]:
        """Record outcome of a completed call.

        Args:
            model: Price-book key.
            input_tokens: Actual input token count from API response.
            output_tokens: Actual output token count from API response.
            success: True if the call returned a usable response.

        Returns:
            USD cost of this call, or None if no :class:`CostTracker` is
            configured.

        Raises:
            LLMUnavailableError: If recording the cost pushes the running
                total past the hard cap.
        """
        if success:
            self.circuit.record_success()
        else:
            self.circuit.record_failure()
        if self.cost is None:
            return None
        return self.cost.record(model, input_tokens, output_tokens)

    def with_retry(
        self,
        fn: Callable[..., Any],
        *args: Any,
        max_retries: int = 3,
        base_delay: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        """Instance shortcut to module-level :func:`with_retry`."""
        return with_retry(
            fn,
            *args,
            max_retries=max_retries,
            base_delay=base_delay,
            **kwargs,
        )


__all__ = [
    "LLMUnavailableError",
    "TokenBucket",
    "CircuitBreaker",
    "CostTracker",
    "RateLimiter",
    "with_retry",
]
