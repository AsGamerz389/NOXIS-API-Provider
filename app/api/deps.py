from __future__ import annotations

from fastapi import Request

from app.core.security import require_auth
from app.utils.errors import RateLimitError


def get_state(request: Request):
    return request.app.state


async def authenticate(request: Request) -> str:
    state = request.app.state
    key_id = require_auth(request, state.authenticator, state.settings.ALLOW_UNAUTHENTICATED)
    if not state.gateway_limiter.allow(key_id):
        raise RateLimitError("NOXIS gateway rate limit exceeded for this key.")
    return key_id
