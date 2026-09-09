from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.api.deps import authenticate
from app.providers.base import ProviderAPIError
from app.utils.errors import ServiceUnavailableError
from app.utils.ids import gen_id


class EmbeddingsRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str | None = None


router = APIRouter()


@router.post("/v1/embeddings")
async def create_embeddings(payload: EmbeddingsRequest, request: Request, key_id: str = Depends(authenticate)):
    """Routes to any enabled provider that declares the requested
    embedding model. NOXIS does not fabricate embedding support for
    providers that only expose chat completions."""
    state = request.app.state
    candidates = [p for p in state.registry.enabled() if p.supports_model(payload.model)]
    if not candidates:
        raise ServiceUnavailableError(f"No enabled provider supports embedding model '{payload.model}'.")

    last_error: Exception | None = None
    for provider in sorted(candidates, key=lambda p: p.config.priority):
        url = provider.config.base_url.rstrip("/") + "/embeddings"
        headers = {"Content-Type": "application/json", **provider._auth_headers()}
        try:
            resp = await provider.client.post(url, json=payload.model_dump(exclude_none=True), headers=headers)
            if resp.status_code >= 400:
                last_error = ProviderAPIError(resp.status_code, resp.text[:300], resp.status_code in (429, 500, 502, 503, 504))
                continue
            data = resp.json()
            data.setdefault("id", gen_id("embd"))
            data["noxis_provider"] = provider.name
            return data
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    raise ServiceUnavailableError(f"All providers failed for embeddings. Last error: {last_error}")
