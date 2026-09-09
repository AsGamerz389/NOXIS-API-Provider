from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import authenticate
from app.utils.errors import NotFoundError, PermissionDeniedError

router = APIRouter()


async def require_master(request: Request, key_id: str = Depends(authenticate)) -> str:
    """Admin endpoints require the NOXIS_MASTER_KEY specifically, not just
    any gateway key, so client-issued keys can't manage providers."""
    state = request.app.state
    from app.core.security import extract_bearer_token, hash_key
    token = extract_bearer_token(request)
    if not state.settings.NOXIS_MASTER_KEY or not token or hash_key(token) != hash_key(state.settings.NOXIS_MASTER_KEY):
        raise PermissionDeniedError("Admin endpoints require the NOXIS master key.")
    return key_id


@router.get("/admin/providers")
async def admin_providers(request: Request, key_id: str = Depends(require_master)):
    state = request.app.state
    return {
        "providers": [
            {
                "name": p.name,
                "type": p.provider_type.value,
                "enabled": p.config.enabled and not state.registry.is_admin_disabled(p.name),
                "health": p.health.value,
                "circuit_state": p.circuit_state.value,
                "priority": p.config.priority,
                "has_credential": bool(p.config.api_key),
            }
            for p in state.registry.all()
        ]
    }


@router.post("/admin/providers/{name}/enable")
async def admin_enable(name: str, request: Request, key_id: str = Depends(require_master)):
    state = request.app.state
    if not state.registry.set_enabled(name, True):
        raise NotFoundError(f"Unknown provider '{name}'.")
    await state.usage.audit(key_id, "provider_enable", {"provider": name})
    return {"provider": name, "enabled": True}


@router.post("/admin/providers/{name}/disable")
async def admin_disable(name: str, request: Request, key_id: str = Depends(require_master)):
    state = request.app.state
    if not state.registry.set_enabled(name, False):
        raise NotFoundError(f"Unknown provider '{name}'.")
    await state.usage.audit(key_id, "provider_disable", {"provider": name})
    return {"provider": name, "enabled": False}


@router.get("/admin/usage")
async def admin_usage(request: Request, key_id: str = Depends(require_master)):
    return await request.app.state.usage.summary(limit=1000)


@router.get("/admin/health")
async def admin_health(request: Request, key_id: str = Depends(require_master)):
    state = request.app.state
    await state.health_monitor.run_once()
    return {
        p.name: {"health": p.health.value, "circuit_state": p.circuit_state.value}
        for p in state.registry.all()
    }


@router.get("/admin/config")
async def admin_config(request: Request, key_id: str = Depends(require_master)):
    """Returns non-secret configuration only. Never returns API keys,
    the master key, or gateway keys."""
    s = request.app.state.settings
    return {
        "app_name": s.APP_NAME,
        "version": s.APP_VERSION,
        "environment": s.ENVIRONMENT,
        "cache_enabled": s.CACHE_ENABLED,
        "cache_ttl": s.CACHE_TTL,
        "max_retries": s.MAX_RETRIES,
        "circuit_failure_threshold": s.CIRCUIT_FAILURE_THRESHOLD,
        "circuit_cooldown_seconds": s.CIRCUIT_COOLDOWN_SECONDS,
        "gateway_rate_limit_per_min": s.GATEWAY_RATE_LIMIT_PER_MIN,
        "allow_unauthenticated": s.ALLOW_UNAUTHENTICATED,
        "providers_configured": len(request.app.state.registry.all()),
    }
