from __future__ import annotations

from fastapi import APIRouter, Request, Response

from app.core.metrics import render_latest

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    state = request.app.state
    providers = state.registry.all()
    healthy = [p.name for p in providers if p.health.value == "healthy"]
    return {
        "status": "ok" if healthy else "degraded",
        "providers_total": len(providers),
        "providers_healthy": len(healthy),
    }


@router.get("/v1/providers")
async def providers_list(request: Request):
    state = request.app.state
    out = []
    for p in state.registry.all():
        out.append({
            "name": p.name,
            "type": p.provider_type.value,
            "auth": "required" if p.requires_api_key else "none",
            "enabled": p.config.enabled and not state.registry.is_admin_disabled(p.name),
            "health": p.health.value,
            "circuit_state": p.circuit_state.value,
            "priority": p.config.priority,
            "models": [m.name for m in p.config.models],
            "supports_streaming": p.config.supports_streaming,
            "supports_tools": p.config.supports_tools,
            "supports_vision": p.config.supports_vision,
            "rate_limit": state.rate_limiter.snapshot(p.name),
        })
    return {"providers": out}


@router.get("/v1/status")
async def status(request: Request):
    state = request.app.state
    providers = state.registry.all()
    by_health: dict[str, int] = {}
    for p in providers:
        by_health[p.health.value] = by_health.get(p.health.value, 0) + 1
    usage_summary = await state.usage.summary(limit=200)
    return {
        "app": state.settings.APP_NAME,
        "version": state.settings.APP_VERSION,
        "environment": state.settings.ENVIRONMENT,
        "providers_by_health": by_health,
        "usage_summary": usage_summary,
        "cache_enabled": state.settings.CACHE_ENABLED,
        "note": "NOXIS aggregates legitimate provider capacity. It does not bypass quotas or create unlimited access.",
    }


@router.get("/metrics")
async def metrics():
    body, content_type = render_latest()
    return Response(content=body, media_type=content_type)
