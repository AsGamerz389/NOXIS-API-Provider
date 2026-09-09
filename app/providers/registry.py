"""
Provider registry.

Builds the set of active providers from (1) built-in defaults driven by
environment variables, and (2) an optional providers.yaml for extra
OpenAI-compatible providers. Administrators can enable/disable providers
without touching code — either via env vars or the Admin API (which flips
an in-memory/DB flag, not the config file).

Keyless providers are verified at startup: NOXIS checks reachability and
whether the endpoint actually appears to require authentication, and marks
providers UNAVAILABLE rather than silently trusting the spec sheet.
"""
from __future__ import annotations

import logging
from pathlib import Path

import httpx
import yaml

from app.config import Settings
from app.providers.base import (
    ModelSpec,
    Provider,
    ProviderConfig,
    ProviderHealth,
    ProviderType,
    RateLimitConfig,
)
from app.providers.openai_compatible import OpenAICompatibleProvider
from app.utils.logging import log_event

logger = logging.getLogger("noxis.registry")

# Verified, documented as of this build. Administrators can override /
# extend via providers.yaml. Marked "not fully verified" providers start
# disabled by default even if NAME_ENABLED=true is unset, requiring
# explicit opt-in, since keyless endpoints can change without notice.
_BUILTIN_API_KEY_PROVIDERS: list[dict] = [
    dict(name="groq", base_url="https://api.groq.com/openai/v1", priority=10,
         models=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]),
    dict(name="cerebras", base_url="https://api.cerebras.ai/v1", priority=10,
         models=["llama3.1-8b", "llama-3.3-70b"]),
    dict(name="openrouter", base_url="https://openrouter.ai/api/v1", priority=20, models=[]),
    dict(name="mistral", base_url="https://api.mistral.ai/v1", priority=20,
         models=["mistral-small-latest", "mistral-large-latest"]),
    dict(name="gemini", base_url="https://generativelanguage.googleapis.com/v1beta/openai", priority=20,
         models=["gemini-1.5-flash", "gemini-1.5-pro"]),
    dict(name="deepseek", base_url="https://api.deepseek.com/v1", priority=20,
         models=["deepseek-chat", "deepseek-reasoner"]),
    dict(name="nvidia", base_url="https://integrate.api.nvidia.com/v1", priority=30, models=[]),
    dict(name="cohere", base_url="https://api.cohere.ai/compatibility/v1", priority=30,
         models=["command-r-plus", "command-r"]),
    dict(name="github_models", base_url="https://models.inference.ai.azure.com", priority=30, models=[]),
    dict(name="huggingface", base_url="https://api-inference.huggingface.co/v1", priority=40, models=[]),
]

_BUILTIN_KEYLESS_PROVIDERS: list[dict] = [
    dict(name="pollinations", priority=50, models=["openai", "openai-large", "mistral", "llama"]),
    dict(name="llm7", priority=55, models=[]),
    dict(name="keylessai", priority=60, models=[]),
    dict(name="airforce", priority=70, models=[]),
    dict(name="ovh_ai", priority=70, models=[]),
]


class ProviderRegistry:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client
        self.providers: dict[str, Provider] = {}
        self.disabled_overrides: set[str] = set()  # set via Admin API at runtime

    # -- construction ---------------------------------------------------
    def _build_api_key_providers(self) -> None:
        s = self.settings
        cred_map = {
            "groq": (s.GROQ_ENABLED, s.GROQ_API_KEY),
            "cerebras": (s.CEREBRAS_ENABLED, s.CEREBRAS_API_KEY),
            "openrouter": (s.OPENROUTER_ENABLED, s.OPENROUTER_API_KEY),
            "mistral": (s.MISTRAL_ENABLED, s.MISTRAL_API_KEY),
            "gemini": (s.GEMINI_ENABLED, s.GEMINI_API_KEY),
            "deepseek": (s.DEEPSEEK_ENABLED, s.DEEPSEEK_API_KEY),
            "nvidia": (s.NVIDIA_ENABLED, s.NVIDIA_API_KEY),
            "cohere": (s.COHERE_ENABLED, s.COHERE_API_KEY),
            "github_models": (s.GITHUB_MODELS_ENABLED, s.GITHUB_TOKEN),
            "huggingface": (s.HUGGINGFACE_ENABLED, s.HF_TOKEN),
        }
        for spec in _BUILTIN_API_KEY_PROVIDERS:
            name = spec["name"]
            enabled, api_key = cred_map[name]
            if not enabled:
                continue
            if not api_key:
                log_event(logger, logging.WARNING, "provider_missing_key",
                          provider=name, reason="enabled but no API key configured; skipping")
                continue
            cfg = ProviderConfig(
                name=name,
                provider_type=ProviderType.API_KEY,
                base_url=spec["base_url"],
                api_key=api_key,
                enabled=True,
                priority=spec["priority"],
                models=[ModelSpec(name=m) for m in spec["models"]],
                requires_api_key=True,
                supports_streaming=True,
                rate_limit=RateLimitConfig(requests_per_minute=60, max_concurrency=20),
            )
            self.providers[name] = OpenAICompatibleProvider(cfg, self.client)

    def _build_keyless_providers(self) -> None:
        s = self.settings
        enabled_map = {
            "pollinations": (s.POLLINATIONS_ENABLED, s.POLLINATIONS_BASE_URL),
            "llm7": (s.LLM7_ENABLED, s.LLM7_BASE_URL),
            "keylessai": (s.KEYLESSAI_ENABLED, s.KEYLESSAI_BASE_URL),
            "airforce": (s.AIRFORCE_ENABLED, s.AIRFORCE_BASE_URL),
            "ovh_ai": (s.OVH_AI_ENABLED, s.OVH_AI_BASE_URL),
        }
        for spec in _BUILTIN_KEYLESS_PROVIDERS:
            name = spec["name"]
            enabled, base_url = enabled_map[name]
            if not enabled:
                continue
            cfg = ProviderConfig(
                name=name,
                provider_type=ProviderType.KEYLESS,
                base_url=base_url,
                api_key=None,
                enabled=True,
                priority=spec["priority"],
                models=[ModelSpec(name=m) for m in spec["models"]],
                requires_api_key=False,
                supports_streaming=True,
                rate_limit=RateLimitConfig(requests_per_minute=20, max_concurrency=5),
            )
            self.providers[name] = OpenAICompatibleProvider(cfg, self.client)

    def _build_from_yaml(self) -> None:
        path = Path(self.settings.PROVIDERS_CONFIG_PATH)
        if not path.exists():
            return
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:
            log_event(logger, logging.ERROR, "providers_yaml_parse_failed", error=str(exc))
            return
        for entry in data.get("providers", []):
            name = entry["name"]
            if name in self.providers:
                continue  # built-ins take precedence
            ptype = ProviderType(entry.get("type", "custom"))
            api_key_env = entry.get("api_key_env")
            api_key = None
            if api_key_env:
                import os
                api_key = os.environ.get(api_key_env)
            requires_key = entry.get("requires_api_key", ptype == ProviderType.API_KEY)
            if requires_key and not api_key:
                log_event(logger, logging.WARNING, "yaml_provider_missing_key", provider=name)
                continue
            cfg = ProviderConfig(
                name=name,
                provider_type=ptype,
                base_url=entry["base_url"],
                api_key=api_key,
                enabled=entry.get("enabled", True),
                priority=entry.get("priority", 50),
                models=[ModelSpec(name=m) if isinstance(m, str) else ModelSpec(**m)
                        for m in entry.get("models", [])],
                requires_api_key=requires_key,
                supports_streaming=entry.get("supports_streaming", True),
                supports_tools=entry.get("supports_tools", False),
                supports_vision=entry.get("supports_vision", False),
            )
            if not cfg.enabled:
                continue
            self.providers[name] = OpenAICompatibleProvider(cfg, self.client)

    def build(self) -> None:
        self._build_api_key_providers()
        self._build_keyless_providers()
        self._build_from_yaml()
        log_event(logger, logging.INFO, "providers_built",
                  count=len(self.providers), names=list(self.providers.keys()))

    # -- runtime verification --------------------------------------------
    async def verify_keyless_providers(self) -> None:
        """
        For each keyless provider: verify reachability, and heuristically
        confirm no authentication is actually required. Never attempts to
        bypass auth — if a provider now demands credentials, it is marked
        UNAVAILABLE, not worked around.
        """
        for provider in list(self.providers.values()):
            if provider.provider_type != ProviderType.KEYLESS:
                continue
            provider.health = ProviderHealth.CHECKING
            try:
                reachable = await provider.health_check()
            except Exception as exc:
                reachable = False
                log_event(logger, logging.WARNING, "keyless_verify_error",
                          provider=provider.name, error=str(exc))
            if reachable:
                provider.health = ProviderHealth.HEALTHY
                log_event(logger, logging.INFO, "keyless_provider_verified", provider=provider.name)
            else:
                provider.health = ProviderHealth.UNAVAILABLE
                log_event(logger, logging.WARNING, "keyless_provider_unavailable", provider=provider.name)

    # -- accessors ---------------------------------------------------------
    def get(self, name: str) -> Provider | None:
        return self.providers.get(name)

    def all(self) -> list[Provider]:
        return list(self.providers.values())

    def enabled(self) -> list[Provider]:
        return [p for p in self.providers.values()
                if p.config.enabled and p.name not in self.disabled_overrides]

    def set_enabled(self, name: str, enabled: bool) -> bool:
        if name not in self.providers:
            return False
        if enabled:
            self.disabled_overrides.discard(name)
        else:
            self.disabled_overrides.add(name)
        return True

    def is_admin_disabled(self, name: str) -> bool:
        return name in self.disabled_overrides
