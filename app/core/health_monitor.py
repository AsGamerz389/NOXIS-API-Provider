"""Background health monitoring.

Periodically calls each provider's health_check(). Also participates in
the RATE_LIMITED/COOLDOWN state transitions driven by the rate limiter,
so the dashboard and /v1/status reflect current reality without waiting
for a live request to fail first.

UNKNOWN -> CHECKING -> HEALTHY -> RATE_LIMITED -> COOLDOWN -> HEALTHY
"""
from __future__ import annotations

import asyncio
import logging
import time

from app.core.rate_limiter import RateLimiter
from app.providers.base import ProviderHealth
from app.providers.registry import ProviderRegistry
from app.utils.logging import log_event

logger = logging.getLogger("noxis.health")


class HealthMonitor:
    def __init__(self, registry: ProviderRegistry, rate_limiter: RateLimiter, interval_seconds: float = 60.0):
        self.registry = registry
        self.rate_limiter = rate_limiter
        self.interval_seconds = interval_seconds
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def _check_all_once(self) -> None:
        for provider in self.registry.all():
            window = self.rate_limiter.window(provider.name)
            if window.is_rate_limited():
                provider.health = ProviderHealth.RATE_LIMITED
                continue
            provider.health = ProviderHealth.CHECKING
            try:
                ok = await provider.health_check()
            except Exception as exc:
                ok = False
                log_event(logger, logging.WARNING, "health_check_error", provider=provider.name, error=str(exc))
            provider.last_checked = time.time()
            provider.health = ProviderHealth.HEALTHY if ok else ProviderHealth.UNAVAILABLE

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._check_all_once()
            except Exception as exc:
                log_event(logger, logging.ERROR, "health_loop_error", error=str(exc))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                pass

    def start(self) -> None:
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
            self._task = None

    async def run_once(self) -> None:
        await self._check_all_once()
