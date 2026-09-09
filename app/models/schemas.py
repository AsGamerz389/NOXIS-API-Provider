"""OpenAI-compatible request/response schemas."""
from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class FunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDef(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: dict[str, Any]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    n: int | None = Field(default=1, ge=1, le=8)
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    tools: list[ToolDef] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: ResponseFormat | None = None
    seed: int | None = None

    @field_validator("messages")
    @classmethod
    def non_empty_messages(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if not v:
            raise ValueError("messages must not be empty")
        return v


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)
    system_fingerprint: str | None = None
    # NOXIS extension: which upstream provider actually served the request.
    noxis_provider: str | None = None


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]
    noxis_provider: str | None = None


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "noxis"
    noxis_providers: list[str] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class ErrorBody(BaseModel):
    message: str
    type: str
    code: str
    param: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
