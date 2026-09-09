from __future__ import annotations

from fastapi import APIRouter, Depends, Request, HTTPException

from app.api.deps import authenticate
from app.models.registry import list_models

router = APIRouter()


@router.get("/v1/models")
async def get_models(request: Request, key_id: str = Depends(authenticate)):
    response = list_models(request.app.state.registry)
    if not response.data:
        raise HTTPException(status_code=503, detail={
            "message": "No verified keyless models are currently available.",
            "type": "provider_unavailable",
            "code": "NOXIS_NO_KEYLESS_MODELS",
        })
    return response
