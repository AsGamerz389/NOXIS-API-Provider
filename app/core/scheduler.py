"""
Scheduler: ties router + circuit breaker + rate limiter together to
execute a chat completion (streaming or not) with bounded retries and
failover across providers.

Rules enforced here:
- Only transient failures (429/500/502/503/504, timeouts, connection
  errors) are retried; invalid requests, auth failures, unsupported
  models, and permission errors propagate immediately.
- Exponential backoff with jitter between attempts; Retry-After is
  honored when present.
- Bounded total retries across all providers combined (MAX_RETRIES),
  preventing retry storms.
- Once a streaming response has begun emitting content to the client,
  NOXIS never switches providers mid-stream — a failure after the first
  chunk ends the stream with an error event, it does not silently retry
  on a different backend.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from app.core.failover import CircuitBreaker, backoff_delay
from app.core.rate_limiter import RateLimiter
from app.core.router import Router, RoutingDecision
from app.models.schemas import ChatCompletionRequest
from app.providers.base import ProviderAPIError, ProviderResult
from app.utils.errors import ServiceUnavailableError
from app.utils.logging import log_event

logger = logging.getLogger("noxis.scheduler")


class Scheduler:
    def __init__(self, router: Router, breaker: CircuitBreaker, rate_limiter: RateLimiter,
                 max_retries: int, backoff_base: float, backoff_cap: float):
        self.router = router
        self.breaker = breaker
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap

    def _eligible_decisions(self, request: ChatCompletionRequest) -> list[RoutingDecision]:
        return [d for d in self.router.rank(request) if self.breaker.allow_request(d.provider)]

    async def execute(self, request: ChatCompletionRequest, request_id: str) -> tuple[ProviderResult, str]:
        """Non-streaming path. Returns (result, provider_name)."""
        decisions = self._eligible_decisions(request)
        if not decisions:
            raise ServiceUnavailableError()

        attempts = 0
        last_error: Exception | None = None

        for decision in decisions:
            if attempts >= self.max_retries:
                break
            provider = decision.provider
            call_model = request.model_copy(update={"model": decision.model})
            self.rate_limiter.acquire(provider.name)
            start = time.perf_counter()
            try:
                result = await provider.chat_completion(call_model)
                self.breaker.record_success(provider)
                self.router.observe_latency(provider.name, (time.perf_counter() - start) * 1000)
                log_event(logger, logging.INFO, "provider_success", request_id=request_id,
                          provider=provider.name, model=decision.model, latency_ms=result.latency_ms)
                return result, provider.name
            except ProviderAPIError as exc:
                attempts += 1
                last_error = exc
                self.rate_limiter.record_failure(provider.name)
                if exc.status_code == 429:
                    self.rate_limiter.record_429(provider.name, exc.retry_after)
                self.breaker.record_failure(provider, transient=exc.retryable)
                log_event(logger, logging.WARNING, "provider_failure", request_id=request_id,
                          provider=provider.name, status=exc.status_code, retryable=exc.retryable,
                          attempt=attempts)
                if not exc.retryable:
                    raise
                delay = backoff_delay(attempts, self.backoff_base, self.backoff_cap, exc.retry_after)
                await asyncio.sleep(delay)
                continue
            finally:
                self.rate_limiter.release(provider.name)

        if last_error is not None:
            raise ServiceUnavailableError(
                f"All eligible providers failed after {attempts} attempt(s). Last error: {last_error}"
            )
        raise ServiceUnavailableError()

    async def execute_stream(self, request: ChatCompletionRequest, request_id: str) -> AsyncIterator[tuple[dict[str, Any], str]]:
        """Streaming path. Yields (chunk, provider_name). Once the first
        chunk from a provider has been yielded, failures do not fail over
        to a different provider — the stream simply ends with an error the
        caller surfaces as a terminal SSE event."""
        decisions = self._eligible_decisions(request)
        if not decisions:
            raise ServiceUnavailableError()

        attempts = 0
        last_error: Exception | None = None

        for decision in decisions:
            if attempts >= self.max_retries:
                break
            provider = decision.provider
            call_model = request.model_copy(update={"model": decision.model})
            self.rate_limiter.acquire(provider.name)
            started_emitting = False
            start = time.perf_counter()
            try:
                async for chunk in provider.stream_completion(call_model):
                    started_emitting = True
                    yield chunk, provider.name
                self.breaker.record_success(provider)
                self.router.observe_latency(provider.name, (time.perf_counter() - start) * 1000)
                return
            except ProviderAPIError as exc:
                self.rate_limiter.record_failure(provider.name)
                if exc.status_code == 429:
                    self.rate_limiter.record_429(provider.name, exc.retry_after)
                self.breaker.record_failure(provider, transient=exc.retryable)
                if started_emitting or not exc.retryable:
                    # Never switch providers mid-stream.
                    log_event(logger, logging.ERROR, "stream_failed_mid_output", request_id=request_id,
                              provider=provider.name, status=exc.status_code)
                    raise
                attempts += 1
                last_error = exc
                log_event(logger, logging.WARNING, "stream_provider_failure_pre_output", request_id=request_id,
                          provider=provider.name, status=exc.status_code, attempt=attempts)
                delay = backoff_delay(attempts, self.backoff_base, self.backoff_cap, exc.retry_after)
                await asyncio.sleep(delay)
                continue
            finally:
                self.rate_limiter.release(provider.name)

        if last_error is not None:
            raise ServiceUnavailableError(
                f"All eligible providers failed before producing output. Last error: {last_error}"
            )
        raise ServiceUnavailableError()
