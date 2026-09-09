"""Consistent OpenAI-style error handling. Never leaks stack traces."""
from __future__ import annotations

from fastapi import HTTPException


class NoxisError(HTTPException):
    """Base error that renders as an OpenAI-compatible error body."""

    def __init__(self, status_code: int, message: str, type_: str, code: str, param: str | None = None):
        self.err_type = type_
        self.err_code = code
        self.param = param
        super().__init__(
            status_code=status_code,
            detail={
                "error": {
                    "message": message,
                    "type": type_,
                    "code": code,
                    "param": param,
                }
            },
        )


class InvalidRequestError(NoxisError):
    def __init__(self, message: str, param: str | None = None, code: str = "NOXIS_INVALID_REQUEST"):
        super().__init__(400, message, "invalid_request_error", code, param)


class AuthenticationError(NoxisError):
    def __init__(self, message: str = "Invalid or missing API key."):
        super().__init__(401, message, "authentication_error", "NOXIS_AUTH_FAILED")


class PermissionDeniedError(NoxisError):
    def __init__(self, message: str = "You do not have permission to perform this action."):
        super().__init__(403, message, "permission_error", "NOXIS_PERMISSION_DENIED")


class NotFoundError(NoxisError):
    def __init__(self, message: str):
        super().__init__(404, message, "invalid_request_error", "NOXIS_NOT_FOUND")


class PayloadTooLargeError(NoxisError):
    def __init__(self, message: str = "Request payload exceeds the configured size limit."):
        super().__init__(413, message, "invalid_request_error", "NOXIS_PAYLOAD_TOO_LARGE")


class RateLimitError(NoxisError):
    def __init__(self, message: str = "Rate limit exceeded."):
        super().__init__(429, message, "rate_limit_error", "NOXIS_RATE_LIMITED")


class ServiceUnavailableError(NoxisError):
    def __init__(self, message: str = "No healthy provider is currently available."):
        super().__init__(503, message, "service_unavailable", "NOXIS_ALL_PROVIDERS_FAILED")


class UpstreamTimeoutError(NoxisError):
    def __init__(self, message: str = "Upstream provider timed out."):
        super().__init__(504, message, "timeout_error", "NOXIS_UPSTREAM_TIMEOUT")


class InternalError(NoxisError):
    def __init__(self, message: str = "An internal error occurred."):
        super().__init__(500, message, "internal_error", "NOXIS_INTERNAL_ERROR")
