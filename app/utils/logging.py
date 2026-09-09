"""Structured JSON logging with automatic secret redaction.

Secrets (API keys, Authorization headers, gateway keys) must never reach
log output. `redact()` is applied defensively at every log call site that
touches request/response data.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any

_REDACT_PATTERNS = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9\-_\.=]+", re.IGNORECASE),
    re.compile(r"(sk-[A-Za-z0-9]{10,})"),
    re.compile(r"(api[_-]?key\"?\s*[:=]\s*\"?)[A-Za-z0-9\-_\.=]{8,}", re.IGNORECASE),
]
_SENSITIVE_KEYS = {
    "authorization", "api_key", "apikey", "x-api-key", "noxis_master_key",
    "password", "secret", "token", "key",
}


def redact(value: Any) -> Any:
    if isinstance(value, str):
        redacted = value
        for pat in _REDACT_PATTERNS:
            redacted = pat.sub(lambda m: (m.group(1) + "***REDACTED***") if m.groups() else "***REDACTED***", redacted)
        return redacted
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if str(k).lower() in _SENSITIVE_KEYS:
                out[k] = "***REDACTED***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(redact(extra))
        if record.exc_info:
            # Never include raw stack traces in production logs/output.
            payload["error"] = redact(str(record.exc_info[1]))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)


def log_event(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    logger.log(level, message, extra={"extra_fields": fields})
