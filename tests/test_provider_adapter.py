from __future__ import annotations

import httpx
import pytest
import respx

from app.models.schemas import ChatCompletionRequest, ChatMessage
from app.providers.base import ProviderAPIError, ProviderConfig, ProviderType, RateLimitConfig
from app.providers.openai_compatible import OpenAICompatibleProvider


def make_request(model="test-model", stream=False):
    return ChatCompletionRequest(model=model, messages=[ChatMessage(role="user", content="hi")], stream=stream)


def make_provider(name="keyless_test", requires_api_key=False, api_key=None):
    cfg = ProviderConfig(
        name=name,
        provider_type=ProviderType.KEYLESS if not requires_api_key else ProviderType.API_KEY,
        base_url="https://example-provider.test/v1",
        api_key=api_key,
        requires_api_key=requires_api_key,
        rate_limit=RateLimitConfig(requests_per_minute=100, max_concurrency=10),
    )
    client = httpx.AsyncClient()
    return OpenAICompatibleProvider(cfg, client)


@pytest.mark.asyncio
@respx.mock
async def test_keyless_provider_initializes_without_api_key():
    provider = make_provider(requires_api_key=False, api_key=None)
    assert provider.requires_api_key is False
    assert provider.config.api_key is None


@pytest.mark.asyncio
@respx.mock
async def test_no_authorization_header_sent_for_keyless():
    route = respx.post("https://example-provider.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "cmpl-1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
    )
    provider = make_provider(requires_api_key=False)
    result = await provider.chat_completion(make_request())
    assert route.called
    sent_headers = route.calls[0].request.headers
    assert "authorization" not in {k.lower() for k in sent_headers.keys()}
    assert result.raw["choices"][0]["message"]["content"] == "hello"


@pytest.mark.asyncio
@respx.mock
async def test_api_key_provider_sends_bearer_header():
    route = respx.post("https://example-provider.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "cmpl-2", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        })
    )
    provider = make_provider(requires_api_key=True, api_key="sk-real-key")
    await provider.chat_completion(make_request())
    sent = route.calls[0].request.headers.get("authorization")
    assert sent == "Bearer sk-real-key"


@pytest.mark.asyncio
@respx.mock
async def test_never_sends_bearer_none():
    # Misconfigured: requires_api_key True but no key set -> registry
    # should have skipped this provider entirely, but defensively verify
    # the adapter itself never emits "Bearer None".
    route = respx.post("https://example-provider.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"id": "x", "choices": []})
    )
    provider = make_provider(requires_api_key=True, api_key=None)
    await provider.chat_completion(make_request())
    sent = route.calls[0].request.headers.get("authorization")
    assert sent is None


@pytest.mark.asyncio
@respx.mock
async def test_429_raises_retryable_provider_error_with_retry_after():
    respx.post("https://example-provider.test/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": "rate limited"}, headers={"Retry-After": "12"})
    )
    provider = make_provider()
    with pytest.raises(ProviderAPIError) as exc_info:
        await provider.chat_completion(make_request())
    assert exc_info.value.status_code == 429
    assert exc_info.value.retryable is True
    assert exc_info.value.retry_after == 12.0


@pytest.mark.asyncio
@respx.mock
async def test_400_raises_non_retryable_provider_error():
    respx.post("https://example-provider.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": "bad request"})
    )
    provider = make_provider()
    with pytest.raises(ProviderAPIError) as exc_info:
        await provider.chat_completion(make_request())
    assert exc_info.value.status_code == 400
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
@respx.mock
async def test_500_is_retryable():
    respx.post("https://example-provider.test/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    provider = make_provider()
    with pytest.raises(ProviderAPIError) as exc_info:
        await provider.chat_completion(make_request())
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
@respx.mock
async def test_streaming_yields_normalized_chunks():
    sse_body = (
        b'data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        b'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
        b'data: [DONE]\n\n'
    )
    respx.post("https://example-provider.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})
    )
    provider = make_provider()
    chunks = [c async for c in provider.stream_completion(make_request(stream=True))]
    assert len(chunks) == 2
    assert chunks[1]["choices"][0]["delta"]["content"] == "hi"


@pytest.mark.asyncio
@respx.mock
async def test_model_discovery_via_models_endpoint():
    respx.get("https://example-provider.test/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "model-a"}, {"id": "model-b"}]})
    )
    provider = make_provider()
    models = await provider.list_models()
    assert models == ["model-a", "model-b"]


@pytest.mark.asyncio
@respx.mock
async def test_health_check_reports_reachability():
    respx.get("https://example-provider.test/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))
    provider = make_provider()
    assert await provider.health_check() is True
