from __future__ import annotations

import secrets
import time


def gen_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(12)}"


def gen_request_id() -> str:
    return f"req_{int(time.time() * 1000):x}{secrets.token_hex(4)}"


def gen_chat_id() -> str:
    return gen_id("chatcmpl")
