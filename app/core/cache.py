"""
Optional response cache. Never caches streaming responses by default.
Cache key includes model, messages, and all generation parameters that
affect output (temperature, top_p, tools, response_format, etc), so two
requests only collide when they are functionally identical.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any


def build_cache_key(payload: dict[str, Any]) -> str:
    relevant = {
        "model": payload.get("model"),
        "messages": payload.get("messages"),
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "max_tokens": payload.get("max_tokens"),
        "stop": payload.get("stop"),
        "presence_penalty": payload.get("presence_penalty"),
        "frequency_penalty": payload.get("frequency_penalty"),
        "tools": payload.get("tools"),
        "tool_choice": payload.get("tool_choice"),
        "response_format": payload.get("response_format"),
        "seed": payload.get("seed"),
    }
    blob = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class InMemoryCache:
    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}

    async def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if not entry:
            return None
        expires_at, value = entry
        if time.time() > expires_at:
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time() + self.ttl, value)

    async def clear(self) -> None:
        self._store.clear()


class RedisCache:
    def __init__(self, redis_client, ttl_seconds: int):
        self.redis = redis_client
        self.ttl = ttl_seconds

    async def get(self, key: str) -> Any | None:
        raw = await self.redis.get(f"noxis:cache:{key}")
        if raw is None:
            return None
        return json.loads(raw)

    async def set(self, key: str, value: Any) -> None:
        await self.redis.set(f"noxis:cache:{key}", json.dumps(value, default=str), ex=self.ttl)

    async def clear(self) -> None:
        pass  # scoped clear intentionally omitted for a shared Redis instance
