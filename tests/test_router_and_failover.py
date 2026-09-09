from __future__ import annotations

import time

import httpx
import pytest

from app.core.failover import CircuitBreaker
from app.core.rate_limiter import GatewayRateLimiter, RateLimiter
from app.core.router import Router
from app.models.schemas import ChatCompletionRequest, ChatMessage
from app.providers.base import (
    CircuitState,
    ModelSpec,
    ProviderConfig,
    ProviderHealth,
    ProviderType,
    RateLimitConfig,
)
from app.providers.openai_compatible import OpenAICompatibleProvider
from app.providers.registry import ProviderRegistry
from app.config import Settings


def build_provider(name, priority=50, healthy=True, models=None, ptype=ProviderType.KEYLESS):
    cfg = ProviderConfig(
        name=name,
        provider_type=ptype,
        base_url=f"https://{name}.test/v1",
        priority=priority,
        models=[ModelSpec(name=m) for m in (models or ["noxis-test-model"])],
        requires_api_key=(ptype == ProviderType.API_KEY),
        api_key="k" if ptype == ProviderType.API_KEY else None,
        rate_limit=RateLimitConfig(requests_per_minute=60, max_concurrency=5),
    )
    p = OpenAICompatibleProvider(cfg, httpx.AsyncClient())
    p.health = ProviderHealth.HEALTHY if healthy else ProviderHealth.UNAVAILABLE
    return p


class FakeRegistry:
    def __init__(self, providers):
        self._providers = {p.name: p for p in providers}
        self.disabled_overrides = set()

    def enabled(self):
        return [p for p in self._providers.values() if p.name not in self.disabled_overrides]

    def all(self):
        return list(self._providers.values())

    def is_admin_disabled(self, name):
        return name in self.disabled_overrides

    def get(self, name):
        return self._providers.get(name)


def test_router_excludes_unhealthy_providers():
    healthy = build_provider("healthy_one", healthy=True)
    unhealthy = build_provider("unhealthy_one", healthy=False)
    registry = FakeRegistry([healthy, unhealthy])
    router = Router(registry, RateLimiter())
    req = ChatCompletionRequest(model="noxis-test-model", messages=[ChatMessage(role="user", content="hi")])
    decisions = router.rank(req)
    names = [d.provider.name for d in decisions]
    assert "healthy_one" in names
    assert "unhealthy_one" not in names


def test_router_prefers_lower_priority_number():
    fast = build_provider("fast", priority=1)
    slow = build_provider("slow", priority=99)
    registry = FakeRegistry([fast, slow])
    router = Router(registry, RateLimiter())
    req = ChatCompletionRequest(model="noxis-test-model", messages=[ChatMessage(role="user", content="hi")])
    decisions = router.rank(req)
    assert decisions[0].provider.name == "fast"


def test_router_excludes_rate_limited_provider():
    p = build_provider("limited")
    registry = FakeRegistry([p])
    rl = RateLimiter()
    for _ in range(60):
        rl.acquire("limited")
        rl.release("limited")
    router = Router(registry, rl)
    req = ChatCompletionRequest(model="noxis-test-model", messages=[ChatMessage(role="user", content="hi")])
    decisions = router.rank(req)
    assert decisions == []


def test_noxis_free_prefers_keyless():
    keyless = build_provider("free_one", ptype=ProviderType.KEYLESS)
    paid = build_provider("paid_one", ptype=ProviderType.API_KEY)
    from app.models.registry import candidate_providers_for_model
    registry = FakeRegistry([keyless, paid])
    candidates = candidate_providers_for_model(registry, "noxis-free")
    names = [p.name for p in candidates]
    assert "free_one" in names
    assert "paid_one" not in names


def test_context_length_hard_filter_excludes_oversized_request():
    p = build_provider("small_ctx", models=["tiny-model"])
    p.config.models[0].context_length = 10  # tokens
    registry = FakeRegistry([p])
    router = Router(registry, RateLimiter())
    long_content = "word " * 1000
    req = ChatCompletionRequest(model="tiny-model", messages=[ChatMessage(role="user", content=long_content)])
    decisions = router.rank(req)
    assert decisions == []


def test_circuit_breaker_opens_after_threshold_and_half_opens_after_cooldown():
    p = build_provider("flaky")
    breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=0.05, half_open_max_calls=1)
    assert breaker.allow_request(p) is True
    for _ in range(3):
        breaker.record_failure(p, transient=True)
    assert p.circuit_state == CircuitState.OPEN
    assert breaker.allow_request(p) is False
    time.sleep(0.06)
    assert breaker.allow_request(p) is True
    assert p.circuit_state == CircuitState.HALF_OPEN
    breaker.record_success(p)
    assert p.circuit_state == CircuitState.CLOSED
    assert p.consecutive_failures == 0


def test_circuit_breaker_ignores_non_transient_failures():
    p = build_provider("stable")
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=5, half_open_max_calls=1)
    breaker.record_failure(p, transient=False)
    breaker.record_failure(p, transient=False)
    assert p.circuit_state == CircuitState.CLOSED


def test_gateway_rate_limiter_blocks_after_limit():
    limiter = GatewayRateLimiter(requests_per_minute=2)
    assert limiter.allow("key1") is True
    assert limiter.allow("key1") is True
    assert limiter.allow("key1") is False
    assert limiter.allow("key2") is True  # separate bucket per key


def test_provider_never_sends_bearer_none_for_missing_key_config():
    cfg = ProviderConfig(
        name="broken", provider_type=ProviderType.API_KEY, base_url="https://x.test/v1",
        api_key=None, requires_api_key=True,
    )
    p = OpenAICompatibleProvider(cfg, httpx.AsyncClient())
    headers = p._auth_headers()
    assert "Authorization" not in headers
