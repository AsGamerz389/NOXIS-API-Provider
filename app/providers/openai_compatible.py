"""
Generic adapter for any OpenAI-compatible upstream — used for both
API_KEY providers (Groq, Cerebras, OpenRouter, Mistral, DeepSeek, NVIDIA,
Cohere-compat, GitHub Models) and KEYLESS providers (Pollinations, LLM7,
KeylessAI, etc). The only behavioral difference is whether an Authorization
header is attached, which `Provider._auth_headers()` already handles.
"""
from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.models.schemas import ChatCompletionRequest
from app.providers.base import (
    TRANSIENT_STATUS_CODES,
    Provider,
    ProviderAPIError,
    ProviderResult,
)


def _request_payload(request: ChatCompletionRequest, stream: bool) -> dict[str, Any]:
    payload = request.model_dump(exclude_none=True, exclude={"stream"})
    payload["stream"] = stream
    return payload


class OpenAICompatibleProvider(Provider):
    async def _post(self, path: str, payload: dict[str, Any], stream: bool) -> httpx.Response:
        url = self.config.base_url.rstrip("/") + path
        headers = {"Content-Type": "application/json", **self._auth_headers()}
        req = self.client.build_request("POST", url, json=payload, headers=headers)
        response = await self.client.send(req, stream=stream)
        return response

    @staticmethod
    def _raise_for_status(resp: httpx.Response, body_text: str | None = None) -> None:
        if resp.status_code < 400:
            return
        retryable = resp.status_code in TRANSIENT_STATUS_CODES
        retry_after = None
        ra_header = resp.headers.get("retry-after")
        if ra_header:
            try:
                retry_after = float(ra_header)
            except ValueError:
                retry_after = None
        message = body_text or f"Upstream returned HTTP {resp.status_code}"
        raise ProviderAPIError(resp.status_code, message[:500], retryable, retry_after)

    async def chat_completion(self, request: ChatCompletionRequest) -> ProviderResult:
        payload = _request_payload(request, stream=False)
        start = time.perf_counter()
        try:
            resp = await self._post(self.config.chat_path, payload, stream=False)
        except httpx.TimeoutException as exc:
            raise ProviderAPIError(504, f"Timeout contacting {self.name}: {exc}", retryable=True) from exc
        except httpx.ConnectError as exc:
            raise ProviderAPIError(502, f"Connection error to {self.name}: {exc}", retryable=True) from exc
        latency_ms = (time.perf_counter() - start) * 1000
        if resp.status_code >= 400:
            text = resp.text
            self._raise_for_status(resp, text)
        data = resp.json()
        return ProviderResult(raw=data, latency_ms=latency_ms, provider_name=self.name)

    async def stream_completion(self, request: ChatCompletionRequest) -> AsyncIterator[dict[str, Any]]:
        payload = _request_payload(request, stream=True)
        url = self.config.base_url.rstrip("/") + self.config.chat_path
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream", **self._auth_headers()}
        try:
            async with self.client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    self._raise_for_status(resp, body.decode(errors="replace"))
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        return
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
        except httpx.TimeoutException as exc:
            raise ProviderAPIError(504, f"Timeout streaming from {self.name}: {exc}", retryable=True) from exc
        except httpx.ConnectError as exc:
            raise ProviderAPIError(502, f"Connection error to {self.name}: {exc}", retryable=True) from exc

    async def list_models(self) -> list[str]:
        if self.config.models:
            return [m.name for m in self.config.models]
        url = self.config.base_url.rstrip("/") + self.config.models_path
        try:
            resp = await self.client.get(url, headers=self._auth_headers(), timeout=10.0)
            if resp.status_code >= 400:
                return []
            data = resp.json()
            items = data.get("data", data if isinstance(data, list) else [])
            return [it.get("id") if isinstance(it, dict) else str(it) for it in items]
        except Exception:
            return []

    async def health_check(self) -> bool:
        url = self.config.base_url.rstrip("/") + self.config.models_path
        try:
            resp = await self.client.get(url, headers=self._auth_headers(), timeout=8.0)
            # Some keyless providers 404 /models but are still reachable for
            # chat completions; treat any non-5xx, non-connection-error
            # response as "reachable".
            return resp.status_code < 500
        except Exception:
            return False
