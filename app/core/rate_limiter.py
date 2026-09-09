"""
Per-provider and per-gateway-key rate limit tracking.

Keyless does not mean unlimited: NOXIS tracks requests/minute,
requests/day, concurrency, and observed 429s + Retry-After per provider,
and reduces a provider's routing weight (or removes it entirely) when it
is near/over its budget. This is bookkeeping only — it never attempts to
evade a provider's real limit.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class ProviderUsageWindow:
    minute_hits: deque = field(default_factory=deque)
    day_hits: deque = field(default_factory=deque)
    concurrent: int = 0
    total_requests: int = 0
    total_failures: int = 0
    total_429s: int = 0
    rate_limited_until: float = 0.0

    def record_request(self) -> None:
        now = time.time()
        self.minute_hits.append(now)
        self.day_hits.append(now)
        self.total_requests += 1
        self._trim(now)

    def record_429(self, retry_after: float | None) -> None:
        self.total_429s += 1
        cooldown = retry_after if retry_after and retry_after > 0 else 15.0
        self.rate_limited_until = max(self.rate_limited_until, time.time() + cooldown)

    def record_failure(self) -> None:
        self.total_failures += 1

    def _trim(self, now: float) -> None:
        while self.minute_hits and now - self.minute_hits[0] > 60:
            self.minute_hits.popleft()
        while self.day_hits and now - self.day_hits[0] > 86400:
            self.day_hits.popleft()

    def requests_last_minute(self) -> int:
        self._trim(time.time())
        return len(self.minute_hits)

    def requests_last_day(self) -> int:
        self._trim(time.time())
        return len(self.day_hits)

    def is_rate_limited(self) -> bool:
        return time.time() < self.rate_limited_until

    def seconds_until_available(self) -> float:
        return max(0.0, self.rate_limited_until - time.time())


class RateLimiter:
    """In-memory rate limit tracker, one window per provider.

    For multi-instance deployments, swap the backing store for Redis
    (the interface below is intentionally storage-agnostic at the call
    site); a single-process in-memory tracker is sufficient for most
    self-hosted NOXIS deployments.
    """

    def __init__(self) -> None:
        self._windows: dict[str, ProviderUsageWindow] = {}

    def window(self, provider_name: str) -> ProviderUsageWindow:
        if provider_name not in self._windows:
            self._windows[provider_name] = ProviderUsageWindow()
        return self._windows[provider_name]

    def can_serve(self, provider_name: str, rpm_limit: int | None, rpd_limit: int | None, max_concurrency: int) -> bool:
        w = self.window(provider_name)
        if w.is_rate_limited():
            return False
        if w.concurrent >= max_concurrency:
            return False
        if rpm_limit is not None and w.requests_last_minute() >= rpm_limit:
            return False
        if rpd_limit is not None and w.requests_last_day() >= rpd_limit:
            return False
        return True

    def acquire(self, provider_name: str) -> None:
        w = self.window(provider_name)
        w.concurrent += 1
        w.record_request()

    def release(self, provider_name: str) -> None:
        w = self.window(provider_name)
        w.concurrent = max(0, w.concurrent - 1)

    def record_429(self, provider_name: str, retry_after: float | None) -> None:
        self.window(provider_name).record_429(retry_after)

    def record_failure(self, provider_name: str) -> None:
        self.window(provider_name).record_failure()

    def snapshot(self, provider_name: str) -> dict:
        w = self.window(provider_name)
        return {
            "requests_last_minute": w.requests_last_minute(),
            "requests_last_day": w.requests_last_day(),
            "concurrent": w.concurrent,
            "total_requests": w.total_requests,
            "total_failures": w.total_failures,
            "total_429s": w.total_429s,
            "rate_limited": w.is_rate_limited(),
            "seconds_until_available": w.seconds_until_available(),
        }


# Simple gateway-level (per client key) request rate limiting, separate
# from provider tracking, to keep NOXIS from becoming an unrestricted
# public relay.
class GatewayRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.rpm = requests_per_minute
        self._hits: dict[str, deque] = {}

    def allow(self, key_id: str) -> bool:
        now = time.time()
        dq = self._hits.setdefault(key_id, deque())
        while dq and now - dq[0] > 60:
            dq.popleft()
        if len(dq) >= self.rpm:
            return False
        dq.append(now)
        return True
