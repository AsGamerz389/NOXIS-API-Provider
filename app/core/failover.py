"""
Circuit breaker + retry policy.

State machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED

- CLOSED: normal operation. Failures increment a counter.
- OPEN: too many consecutive failures; provider is skipped entirely until
  the cooldown elapses.
- HALF_OPEN: cooldown elapsed; a bounded number of trial requests are
  allowed through. Success -> CLOSED. Failure -> OPEN again (fresh cooldown).

Only transient failures (429/500/502/503/504, timeouts, connection errors)
count against the breaker. Invalid requests, auth failures, unsupported
models, and permission errors are never retried and never trip the breaker.
"""
from __future__ import annotations

import random
import time

from app.providers.base import CircuitState, Provider, ProviderHealth


class CircuitBreaker:
    def __init__(self, failure_threshold: int, cooldown_seconds: float, half_open_max_calls: int):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls

    def allow_request(self, provider: Provider) -> bool:
        if provider.circuit_state == CircuitState.CLOSED:
            return True
        if provider.circuit_state == CircuitState.OPEN:
            if time.time() >= provider.cooldown_until:
                provider.circuit_state = CircuitState.HALF_OPEN
                provider.half_open_calls_in_flight = 0
                return self.allow_request(provider)
            return False
        if provider.circuit_state == CircuitState.HALF_OPEN:
            if provider.half_open_calls_in_flight < self.half_open_max_calls:
                provider.half_open_calls_in_flight += 1
                return True
            return False
        return False

    def record_success(self, provider: Provider) -> None:
        provider.consecutive_failures = 0
        provider.circuit_state = CircuitState.CLOSED
        provider.half_open_calls_in_flight = 0
        if provider.health in (ProviderHealth.RATE_LIMITED, ProviderHealth.COOLDOWN, ProviderHealth.CIRCUIT_OPEN):
            provider.health = ProviderHealth.HEALTHY

    def record_failure(self, provider: Provider, transient: bool) -> None:
        if not transient:
            return  # do not trip the breaker on terminal/non-retryable errors
        provider.consecutive_failures += 1
        if provider.circuit_state == CircuitState.HALF_OPEN:
            self._trip(provider)
            return
        if provider.consecutive_failures >= self.failure_threshold:
            self._trip(provider)

    def _trip(self, provider: Provider) -> None:
        provider.circuit_state = CircuitState.OPEN
        provider.cooldown_until = time.time() + self.cooldown_seconds
        provider.health = ProviderHealth.CIRCUIT_OPEN


def backoff_delay(attempt: int, base: float, cap: float, retry_after: float | None = None) -> float:
    """Exponential backoff with full jitter, honoring an explicit
    Retry-After when the provider supplied one."""
    if retry_after is not None:
        return min(cap, max(0.0, retry_after)) + random.uniform(0, 0.25)
    expo = min(cap, base * (2 ** attempt))
    return random.uniform(0, expo)
