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
    # Pollinations exposes a public OpenAI-compatible endpoint without an
    # Authorization header. Keep the base URL at the service root; the
    # adapter supplies /openai and /models.
    dict(name="pollinations", priority=10, base_url="https://text.pollinations.ai",
         chat_path="/openai", models_path="/models", models=[]),
    # KeylessAI is a public OpenAI-compatible proxy. Its own service performs
    # provider failover; NOXIS still treats it as a normal keyless upstream.
    dict(name="keylessai", priority=20, base_url="https://keylessai.thryx.workers.dev/v1",
         chat_path="/chat/completions", models_path="/models", models=[]),
    # These are optional integrations. They stay disabled unless explicitly
    # enabled and configured, because an endpoint claiming to be free/keyless
    # can change its authentication policy without notice.
    dict(name="llm7", priority=30, base_url="https://api.llm7.io/v1",
         chat_path="/chat/completions", models_path="/models", models=[]),
    dict(name="devtoolbox", priority=40, base_url="",
         chat_path="/chat/completions", models_path="/models", models=[]),
    dict(name="inferenceport", priority=50, base_url="",
         chat_path="/chat/completions", models_path="/models", models=[]),
]


class ProviderRegistry:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client
        self.providers: dict[str, Provider] = {}
        self.disabled_overrides: set[str] = set()  # set via Admin API at runtime

    # -- construction ---------------------------------------------------
    def _build_api_key_providers(self) -> None:
        # NOXIS free deployment: credentialed providers are intentionally
        # excluded from the runtime registry. They cannot become fallback
        # providers through environment variables or YAML.
        return

    def _build_keyless_providers(self) -> None:
        s = self.settings
        enabled_map = {
            "pollinations": (s.POLLINATIONS_ENABLED, s.POLLINATIONS_BASE_URL),
            "keylessai": (s.KEYLESSAI_ENABLED, s.KEYLESSAI_BASE_URL),
            "llm7": (s.LLM7_ENABLED, s.LLM7_BASE_URL),
            "devtoolbox": (s.DEVTOOLBOX_ENABLED, s.DEVTOOLBOX_BASE_URL),
            "inferenceport": (s.INFERENCEPORT_ENABLED, s.INFERENCEPORT_BASE_URL),
        }
        for spec in _BUILTIN_KEYLESS_PROVIDERS:
            name = spec["name"]
            enabled, base_url = enabled_map[name]
            if not enabled or not base_url:
                continue
            cfg = ProviderConfig(
                name=name,
                provider_type=ProviderType.KEYLESS,
                base_url=base_url,
                api_key=None,
                enabled=True,
                priority=spec["priority"],
                models=[ModelSpec(name=m) for m in spec["models"]],
                chat_path=spec.get("chat_path", "/chat/completions"),
                models_path=spec.get("models_path", "/models"),
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
            # This build is intentionally keyless-only. YAML extensions may
            # add keyless providers, but must never introduce credentialed
            # providers into the gateway.
            if ptype != ProviderType.KEYLESS:
                log_event(logger, logging.INFO, "yaml_provider_skipped",
                          provider=name, reason="keyless_only_mode")
                continue
            api_key_env = None
            api_key = None
            if api_key_env:
                import os
                api_key = os.environ.get(api_key_env)
            requires_key = False
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
    async def discover_keyless_models(self) -> None:
        """Discover public model lists and attach them to keyless providers.

        Discovery is best-effort. A provider with an explicit model list keeps
        that list; dynamic providers get models only when their public /models
        endpoint returns them. No credentials are added during discovery.
        """
        for provider in list(self.providers.values()):
            if provider.provider_type != ProviderType.KEYLESS or provider.health == ProviderHealth.UNAVAILABLE:
                continue
            try:
                models = await provider.list_models()
                if models:
                    provider.config.models = [ModelSpec(name=m) for m in dict.fromkeys(models) if m]
                    log_event(logger, logging.INFO, "keyless_models_discovered",
                              provider=provider.name, count=len(provider.config.models))
            except Exception as exc:
                log_event(logger, logging.WARNING, "keyless_model_discovery_failed",
                          provider=provider.name, error=str(exc))

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
