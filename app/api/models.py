from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import authenticate
from app.models.registry import list_models

router = APIRouter()


@router.get("/v1/models")
async def get_models(request: Request, key_id: str = Depends(authenticate)):
    return list_models(request.app.state.registry)
