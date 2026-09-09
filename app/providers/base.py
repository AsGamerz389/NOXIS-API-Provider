"""
Universal provider interface.

Every provider — whether it requires an administrator-supplied API key
(API_KEY), is genuinely public and unauthenticated (KEYLESS), or something
custom — implements this same interface so the router can treat them
uniformly.
"""
from __future__ import annotations

import enum
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.models.schemas import ChatCompletionRequest


class ProviderType(str, enum.Enum):
    KEYLESS = "keyless"
    API_KEY = "api_key"
    OAUTH = "oauth"
    CUSTOM = "custom"


class ProviderHealth(str, enum.Enum):
    UNKNOWN = "unknown"
    CHECKING = "checking"
    HEALTHY = "healthy"
    RATE_LIMITED = "rate_limited"
    COOLDOWN = "cooldown"
    UNAVAILABLE = "unavailable"
    CIRCUIT_OPEN = "circuit_open"


class CircuitState(str, enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ModelSpec:
    name: str
    context_length: int = 8192
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_vision: bool = False


@dataclass
class RateLimitConfig:
    requests_per_minute: int | None = None
    requests_per_day: int | None = None
    max_concurrency: int = 10


@dataclass
class ProviderConfig:
    name: str
    provider_type: ProviderType
    base_url: str
    api_key: str | None = None
    enabled: bool = True
    priority: int = 50  # lower = preferred, all else equal
    models: list[ModelSpec] = field(default_factory=list)
    requires_api_key: bool = False
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_vision: bool = False
    timeout_seconds: float = 60.0
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    chat_path: str = "/chat/completions"
    models_path: str = "/models"
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class ProviderResult:
    """Normalized non-streaming result from a provider call."""
    raw: dict[str, Any]
    latency_ms: float
    provider_name: str


class ProviderAPIError(Exception):
    """Raised for upstream errors. Carries enough info for the router to
    decide whether the failure is transient (retryable) or terminal."""

    def __init__(self, status_code: int, message: str, retryable: bool, retry_after: float | None = None):
        self.status_code = status_code
        self.message = message
        self.retryable = retryable
        self.retry_after = retry_after
        super().__init__(message)


TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class Provider:
    """Base class for all NOXIS providers."""

    def __init__(self, config: ProviderConfig, client: httpx.AsyncClient):
        self.config = config
        self.client = client
        self.name = config.name
        self.provider_type = config.provider_type
        self.requires_api_key = config.requires_api_key

        # Mutable runtime health/circuit state, owned by HealthMonitor /
        # CircuitBreaker but kept on the provider instance for convenience.
        self.health: ProviderHealth = ProviderHealth.UNKNOWN
        self.circuit_state: CircuitState = CircuitState.CLOSED
        self.consecutive_failures: int = 0
        self.cooldown_until: float = 0.0
        self.last_checked: float = 0.0
        self.half_open_calls_in_flight: int = 0

    # -- capability helpers -------------------------------------------------
    def supports_model(self, model: str) -> bool:
        if not self.config.models:
            return True  # unrestricted / dynamic provider
        return any(m.name == model for m in self.config.models)

    def model_spec(self, model: str) -> ModelSpec | None:
        for m in self.config.models:
            if m.name == model:
                return m
        return None

    def _auth_headers(self) -> dict[str, str]:
        """Never send `Authorization: Bearer None` or a fake credential."""
        headers = dict(self.config.extra_headers)
        if self.config.requires_api_key and self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    # -- interface to implement/override ------------------------------------
    async def chat_completion(self, request: ChatCompletionRequest) -> ProviderResult:
        raise NotImplementedError

    async def stream_completion(self, request: ChatCompletionRequest) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError
        yield {}  # pragma: no cover

    async def list_models(self) -> list[str]:
        raise NotImplementedError

    async def health_check(self) -> bool:
        raise NotImplementedError

    async def get_limits(self) -> RateLimitConfig:
        return self.config.rate_limit
