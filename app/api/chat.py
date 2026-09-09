from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from app.api.deps import authenticate
from app.core.cache import build_cache_key
from app.core.metrics import (
    noxis_active_requests,
    noxis_provider_failures_total,
    noxis_provider_latency,
    noxis_provider_requests_total,
    noxis_request_errors_total,
    noxis_request_latency,
    noxis_requests_total,
    noxis_tokens_total,
)
from app.models.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    DeltaMessage,
    Usage,
)
from app.providers.base import ProviderAPIError
from app.utils.errors import (
    InvalidRequestError,
    NoxisError,
    PayloadTooLargeError,
)
from app.utils.ids import gen_chat_id, gen_request_id
from app.utils.logging import log_event

logger = logging.getLogger("noxis.chat")
router = APIRouter()


def _normalize_chat_response(raw: dict, request_id: str, model: str, provider_name: str) -> ChatCompletionResponse:
    choices = []
    for i, c in enumerate(raw.get("choices", [])):
        msg = c.get("message", {})
        choices.append(
            ChatCompletionChoice(
                index=c.get("index", i),
                message=ChatMessage(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content"),
                    tool_calls=msg.get("tool_calls"),
                ),
                finish_reason=c.get("finish_reason"),
            )
        )
    usage_raw = raw.get("usage") or {}
    usage = Usage(
        prompt_tokens=usage_raw.get("prompt_tokens", 0),
        completion_tokens=usage_raw.get("completion_tokens", 0),
        total_tokens=usage_raw.get("total_tokens", 0),
    )
    return ChatCompletionResponse(
        id=raw.get("id") or gen_chat_id(),
        created=raw.get("created", int(time.time())),
        model=raw.get("model", model),
        choices=choices or [ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=""),
                                                   finish_reason="stop")],
        usage=usage,
        noxis_provider=provider_name,
    )


@router.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request, key_id: str = Depends(authenticate)):
    state = request.app.state
    settings = state.settings
    request_id = gen_request_id()
    noxis_active_requests.inc()
    start = time.perf_counter()

    body_size = len(json.dumps(payload.model_dump(exclude_none=True)))
    if body_size > settings.MAX_REQUEST_BYTES:
        raise PayloadTooLargeError()
    if payload.max_tokens and payload.max_tokens > settings.MAX_TOKENS_CEILING:
        raise InvalidRequestError(f"max_tokens exceeds the configured ceiling of {settings.MAX_TOKENS_CEILING}.",
                                   param="max_tokens")

    try:
        if payload.stream:
            # Concurrency is bounded for the lifetime of the stream itself
            # (acquired/released inside the generator), not just the time
            # it takes to construct the response.
            return await _stream_response(payload, request, request_id, key_id)
        async with state.concurrency_semaphore:
            return await _sync_response(payload, request, request_id, key_id)
    finally:
        noxis_active_requests.dec()
        noxis_request_latency.labels(endpoint="chat_completions").observe(time.perf_counter() - start)


async def _sync_response(payload: ChatCompletionRequest, request: Request, request_id: str, key_id: str):
    state = request.app.state
    cache_key = None

    if state.settings.CACHE_ENABLED:
        cache_key = build_cache_key(payload.model_dump(exclude_none=True))
        cached = await state.cache.get(cache_key)
        if cached is not None:
            log_event(logger, logging.INFO, "cache_hit", request_id=request_id)
            noxis_requests_total.labels(endpoint="chat_completions", status="success").inc()
            return ChatCompletionResponse(**cached)

    try:
        result, provider_name = await state.scheduler.execute(payload, request_id)
    except ProviderAPIError as exc:
        noxis_request_errors_total.labels(endpoint="chat_completions", error_code=str(exc.status_code)).inc()
        raise
    except NoxisError as exc:
        noxis_request_errors_total.labels(endpoint="chat_completions", error_code=exc.err_code).inc()
        await state.usage.record(
            request_id=request_id, gateway_key_id=key_id, provider="none", model="none",
            requested_model=payload.model, status="error", error_code=exc.err_code, streamed=False,
        )
        raise

    response = _normalize_chat_response(result.raw, request_id, payload.model, provider_name)
    noxis_provider_requests_total.labels(provider=provider_name, model=response.model).inc()
    noxis_provider_latency.labels(provider=provider_name).observe(result.latency_ms / 1000)
    noxis_tokens_total.labels(provider=provider_name, kind="prompt").inc(response.usage.prompt_tokens)
    noxis_tokens_total.labels(provider=provider_name, kind="completion").inc(response.usage.completion_tokens)
    noxis_requests_total.labels(endpoint="chat_completions", status="success").inc()

    await state.usage.record(
        request_id=request_id, gateway_key_id=key_id, provider=provider_name, model=response.model,
        requested_model=payload.model, latency_ms=result.latency_ms, status="success",
        prompt_tokens=response.usage.prompt_tokens, completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens, streamed=False,
    )

    if cache_key and state.settings.CACHE_ENABLED:
        await state.cache.set(cache_key, response.model_dump())

    return response


async def _stream_response(payload: ChatCompletionRequest, request: Request, request_id: str, key_id: str):
    state = request.app.state
    chat_id = gen_chat_id()

    async def event_gen():
        provider_used = "unknown"
        await state.concurrency_semaphore.acquire()
        try:
            async for raw_chunk, provider_name in state.scheduler.execute_stream(payload, request_id):
                provider_used = provider_name
                chunk = _normalize_chunk(raw_chunk, chat_id, payload.model, provider_name)
                noxis_provider_requests_total.labels(provider=provider_name, model=payload.model).inc()
                # Plain "data: {...}" lines only (no "event:" field) to match
                # the wire format OpenAI-compatible SSE clients expect.
                yield {"data": chunk.model_dump_json()}
            yield {"data": "[DONE]"}
            noxis_requests_total.labels(endpoint="chat_completions_stream", status="success").inc()
            await state.usage.record(
                request_id=request_id, gateway_key_id=key_id, provider=provider_used, model=payload.model,
                requested_model=payload.model, status="success", streamed=True,
            )
        except NoxisError as exc:
            noxis_request_errors_total.labels(endpoint="chat_completions_stream", error_code=exc.err_code).inc()
            error_payload = exc.detail if isinstance(exc.detail, dict) else {
                "error": {"message": str(exc.detail), "type": "service_unavailable", "code": "NOXIS_STREAM_FAILED"}
            }
            yield {"data": json.dumps(error_payload)}
            await state.usage.record(
                request_id=request_id, gateway_key_id=key_id, provider=provider_used, model=payload.model,
                requested_model=payload.model, status="error",
                error_code=error_payload["error"]["code"], streamed=True,
            )
        except ProviderAPIError as exc:
            noxis_provider_failures_total.labels(provider=provider_used, status_code=str(exc.status_code)).inc()
            yield {"data": json.dumps({
                "error": {"message": exc.message, "type": "service_unavailable", "code": "NOXIS_STREAM_FAILED"}
            })}
        finally:
            state.concurrency_semaphore.release()

    return EventSourceResponse(event_gen())


def _normalize_chunk(raw: dict, chat_id: str, requested_model: str, provider_name: str) -> ChatCompletionChunk:
    choices = []
    for i, c in enumerate(raw.get("choices", [])):
        delta = c.get("delta", {})
        choices.append(
            ChatCompletionChunkChoice(
                index=c.get("index", i),
                delta=DeltaMessage(
                    role=delta.get("role"),
                    content=delta.get("content"),
                    tool_calls=delta.get("tool_calls"),
                ),
                finish_reason=c.get("finish_reason"),
            )
        )
    return ChatCompletionChunk(
        id=raw.get("id", chat_id),
        model=raw.get("model", requested_model),
        choices=choices,
        noxis_provider=provider_name,
    )
