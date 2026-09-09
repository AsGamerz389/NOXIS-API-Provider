"""
Security utilities: gateway API-key authentication, key hashing,
basic SSRF protection for any admin-configurable URLs, and security
headers middleware.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import socket
from urllib.parse import urlparse

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import Settings
from app.utils.errors import AuthenticationError, InvalidRequestError

_PRIVATE_NETS = [
    ipaddress.ip_network(n)
    for n in (
        "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
        "169.254.0.0/16", "::1/128", "fc00::/7", "fe80::/10", "0.0.0.0/8",
    )
]


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)


class KeyAuthenticator:
    """Validates gateway keys by comparing SHA-256 hashes, not raw
    strings, and never logs the raw key."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._valid_hashes = {hash_key(k) for k in settings.gateway_keys}

    def refresh(self, settings: Settings) -> None:
        self.settings = settings
        self._valid_hashes = {hash_key(k) for k in settings.gateway_keys}

    def is_valid(self, raw_key: str) -> bool:
        candidate = hash_key(raw_key)
        return any(constant_time_eq(candidate, h) for h in self._valid_hashes)

    def key_id(self, raw_key: str) -> str:
        return hash_key(raw_key)[:12]


def extract_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization")
    if not auth or not auth.lower().startswith("bearer "):
        return None
    return auth[7:].strip()


def require_auth(request: Request, authenticator: KeyAuthenticator, allow_unauthenticated: bool) -> str:
    token = extract_bearer_token(request)
    if token is None:
        if allow_unauthenticated:
            return "anonymous"
        raise AuthenticationError("Missing Authorization: Bearer <NOXIS key> header.")
    if not authenticator.is_valid(token):
        raise AuthenticationError("Invalid NOXIS API key.")
    return authenticator.key_id(token)


def validate_outbound_url(url: str) -> None:
    """Guard against SSRF for any admin-configurable outbound URL (e.g. a
    custom provider base_url added at runtime). Resolves the host and
    rejects private/loopback/link-local ranges. Does not apply to
    built-in providers, which are fixed, publicly documented endpoints."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise InvalidRequestError(f"Unsupported URL scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise InvalidRequestError("URL is missing a host.")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise InvalidRequestError(f"Could not resolve host {host!r}: {exc}") from exc
    for info in infos:
        addr = info[4][0]
        ip = ipaddress.ip_address(addr)
        if any(ip in net for net in _PRIVATE_NETS):
            raise InvalidRequestError(f"Refusing to configure a provider pointing at a private address ({addr}).")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers.setdefault("Cache-Control", "no-store")
        return response
