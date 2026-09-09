"""
Intelligent router.

Selects an ordered list of candidate (provider) choices for a request,
best first, using weighted scoring rather than round-robin.

SCORING ALGORITHM
------------------
For each enabled provider that is health-eligible and supports the
requested model, compute:

    score = ( W_HEALTH   * health_score
            + W_PRIORITY * priority_score
            + W_LATENCY  * latency_score
            + W_RELIABILITY * reliability_score
            + W_QUOTA    * quota_headroom_score
            + W_CAPABILITY * capability_score )

All sub-scores are normalized to [0, 1]; higher is better. Weights sum to
1.0. The provider list is sorted descending by score; the router tries
candidates in that order, falling through to the next on transient
failure (see core/failover.py).

Sub-scores:
- health_score: 1.0 if HEALTHY, 0.4 if UNKNOWN (untested), 0.0 otherwise
  (RATE_LIMITED / COOLDOWN / UNAVAILABLE / CIRCUIT_OPEN are excluded
  entirely before scoring, not merely down-weighted).
- priority_score: derived from the administrator-assigned `priority`
  (lower number = more preferred); normalized against the observed
  min/max priority among candidates.
- latency_score: derived from a rolling average observed latency
  (lower is better); providers with no observations yet get a neutral
  0.5 so they get a fair chance to be tried.
- reliability_score: 1 - (recent_failure_rate), from the rate limiter's
  observed totals.
- quota_headroom_score: fraction of the requests-per-minute /
  requests-per-day budget still remaining (1.0 if unlimited).
- capability_score: 1.0 if the provider fully supports requested
  capabilities (streaming/tools/vision as needed by the request), 0.0 if
  it's missing a *required* capability (excluded before scoring in that
  case), else a partial value.

Context length is enforced as a hard filter: a provider whose model's
context_length is smaller than a conservative estimate of the request's
token footprint is excluded.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.rate_limiter import RateLimiter
from app.models.registry import candidate_providers_for_model, is_alias
from app.models.schemas import ChatCompletionRequest
from app.providers.base import Provider, ProviderHealth
from app.providers.registry import ProviderRegistry

_ELIGIBLE_HEALTH = {ProviderHealth.HEALTHY, ProviderHealth.UNKNOWN}

W_HEALTH = 0.30
W_PRIORITY = 0.15
W_LATENCY = 0.15
W_RELIABILITY = 0.20
W_QUOTA = 0.15
W_CAPABILITY = 0.05


@dataclass
class RoutingDecision:
    provider: Provider
    model: str  # concrete model name to send upstream
    score: float


def _estimate_tokens(request: ChatCompletionRequest) -> int:
    chars = sum(len(m.content) for m in request.messages if isinstance(m.content, str))
    return max(1, chars // 4) + (request.max_tokens or 0)


def _health_score(provider: Provider) -> float:
    if provider.health == ProviderHealth.HEALTHY:
        return 1.0
    if provider.health == ProviderHealth.UNKNOWN:
        return 0.4
    return 0.0


def _priority_score(provider: Provider, candidates: list[Provider]) -> float:
    priorities = [p.config.priority for p in candidates]
    lo, hi = min(priorities), max(priorities)
    if hi == lo:
        return 1.0
    return 1.0 - (provider.config.priority - lo) / (hi - lo)


def _latency_score(provider: Provider, latency_stats: dict[str, float]) -> float:
    avg = latency_stats.get(provider.name)
    if avg is None:
        return 0.5
    # 100ms -> ~1.0, 5000ms -> ~0.0, smooth falloff
    return max(0.0, min(1.0, 1.0 - (avg - 100) / 4900))


def _reliability_score(provider: Provider, rate_limiter: RateLimiter) -> float:
    snap = rate_limiter.snapshot(provider.name)
    total = snap["total_requests"]
    if total == 0:
        return 0.7
    failure_rate = snap["total_failures"] / total
    return max(0.0, 1.0 - failure_rate)


def _quota_score(provider: Provider, rate_limiter: RateLimiter) -> float:
    limits = provider.config.rate_limit
    snap = rate_limiter.snapshot(provider.name)
    scores = []
    if limits.requests_per_minute:
        scores.append(max(0.0, 1.0 - snap["requests_last_minute"] / limits.requests_per_minute))
    if limits.requests_per_day:
        scores.append(max(0.0, 1.0 - snap["requests_last_day"] / limits.requests_per_day))
    if not scores:
        return 1.0
    return sum(scores) / len(scores)


def _capability_score(provider: Provider, request: ChatCompletionRequest) -> float | None:
    if request.stream and not provider.config.supports_streaming:
        return None
    if request.tools and not provider.config.supports_tools:
        return None
    return 1.0


class Router:
    def __init__(self, registry: ProviderRegistry, rate_limiter: RateLimiter):
        self.registry = registry
        self.rate_limiter = rate_limiter
        self._latency_stats: dict[str, float] = {}

    def observe_latency(self, provider_name: str, latency_ms: float) -> None:
        prev = self._latency_stats.get(provider_name)
        self._latency_stats[provider_name] = latency_ms if prev is None else (0.7 * prev + 0.3 * latency_ms)

    def _resolve_model_for_provider(self, provider: Provider, requested_model: str) -> str | None:
        """Returns the concrete model name to send upstream, or None if
        this provider can't safely serve the request (e.g. an alias like
        `noxis-auto` with no configured model list to resolve against —
        sending the literal alias string upstream as `model` would just
        produce a confusing 400 from the provider)."""
        if is_alias(requested_model):
            if provider.config.models:
                return provider.config.models[0].name
            return None  # nothing safe to resolve to; exclude this provider
        return requested_model

    def rank(self, request: ChatCompletionRequest) -> list[RoutingDecision]:
        candidates = candidate_providers_for_model(self.registry, request.model)

        # Hard filters: circuit/health eligibility, quota exhaustion,
        # context length, required capabilities.
        est_tokens = _estimate_tokens(request)
        eligible: list[Provider] = []
        for p in candidates:
            if p.health not in _ELIGIBLE_HEALTH:
                continue
            if self.registry.is_admin_disabled(p.name):
                continue
            limits = p.config.rate_limit
            if not self.rate_limiter.can_serve(
                p.name, limits.requests_per_minute, limits.requests_per_day, limits.max_concurrency
            ):
                continue
            model_name = self._resolve_model_for_provider(p, request.model)
            if model_name is None:
                continue  # can't safely resolve this alias against this provider
            spec = p.model_spec(model_name)
            if spec and spec.context_length and est_tokens > spec.context_length:
                continue
            if _capability_score(p, request) is None:
                continue
            eligible.append(p)

        if not eligible:
            return []

        decisions: list[RoutingDecision] = []
        for p in eligible:
            cap = _capability_score(p, request) or 0.0
            score = (
                W_HEALTH * _health_score(p)
                + W_PRIORITY * _priority_score(p, eligible)
                + W_LATENCY * _latency_score(p, self._latency_stats)
                + W_RELIABILITY * _reliability_score(p, self.rate_limiter)
                + W_QUOTA * _quota_score(p, self.rate_limiter)
                + W_CAPABILITY * cap
            )
            model_name = self._resolve_model_for_provider(p, request.model)
            assert model_name is not None  # guaranteed by the filter above
            decisions.append(RoutingDecision(provider=p, model=model_name, score=score))

        decisions.sort(key=lambda d: d.score, reverse=True)
        return decisions
