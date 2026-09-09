"""
Model registry: exposes GET /v1/models and resolves NOXIS aliases
(noxis-auto, noxis-fast, noxis-smart, noxis-free) to a candidate provider
set at routing time. Aliases are *not* resolved to a single fixed model;
the router picks the best (provider, model) pair per-request.
"""
from __future__ import annotations

from app.models.schemas import ModelInfo, ModelListResponse
from app.providers.base import ProviderType
from app.providers.registry import ProviderRegistry

NOXIS_ALIASES = {"noxis-auto", "noxis-fast", "noxis-smart", "noxis-free"}


def is_alias(model: str) -> bool:
    return model in NOXIS_ALIASES


def list_models(registry: ProviderRegistry) -> ModelListResponse:
    seen: dict[str, list[str]] = {}
    for provider in registry.enabled():
        for m in provider.config.models:
            seen.setdefault(m.name, []).append(provider.name)
    data = [ModelInfo(id=name, noxis_providers=providers) for name, providers in sorted(seen.items())]
    for alias in sorted(NOXIS_ALIASES):
        data.append(ModelInfo(id=alias, owned_by="noxis-router", noxis_providers=["auto"]))
    return ModelListResponse(data=data)


def candidate_providers_for_model(registry: ProviderRegistry, model: str):
    """Return providers that could plausibly serve `model`, honoring alias
    semantics. Final health/quota/priority filtering happens in the router."""
    providers = registry.enabled()

    if model == "noxis-free":
        # Prefer providers that do not require an administrator-supplied
        # paid credential.
        keyless = [p for p in providers if p.provider_type == ProviderType.KEYLESS]
        return keyless or providers

    if model == "noxis-auto":
        return providers

    if model == "noxis-fast":
        return sorted(providers, key=lambda p: p.config.priority)

    if model == "noxis-smart":
        # "Smart" = prefer larger/paid frontier providers where available,
        # falling back to everything else.
        api_key_first = [p for p in providers if p.provider_type == ProviderType.API_KEY]
        return api_key_first or providers

    # Concrete model name: only providers that declare (or dynamically
    # support) it.
    return [p for p in providers if p.supports_model(model)]
