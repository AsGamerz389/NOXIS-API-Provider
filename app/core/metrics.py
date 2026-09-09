from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

noxis_requests_total = Counter(
    "noxis_requests_total", "Total requests received by NOXIS", ["endpoint", "status"]
)
noxis_request_errors_total = Counter(
    "noxis_request_errors_total", "Total request errors", ["endpoint", "error_code"]
)
noxis_provider_requests_total = Counter(
    "noxis_provider_requests_total", "Total requests sent to a provider", ["provider", "model"]
)
noxis_provider_failures_total = Counter(
    "noxis_provider_failures_total", "Total provider failures", ["provider", "status_code"]
)
noxis_request_latency = Histogram(
    "noxis_request_latency_seconds", "End-to-end request latency", ["endpoint"]
)
noxis_provider_latency = Histogram(
    "noxis_provider_latency_seconds", "Upstream provider latency", ["provider"]
)
noxis_tokens_total = Counter(
    "noxis_tokens_total", "Total tokens processed", ["provider", "kind"]  # kind=prompt|completion
)
noxis_active_requests = Gauge(
    "noxis_active_requests", "Currently in-flight requests"
)
noxis_provider_health = Gauge(
    "noxis_provider_health", "Provider health (1=healthy, 0=not)", ["provider"]
)
noxis_circuit_open = Gauge(
    "noxis_circuit_open", "Circuit breaker open (1=open)", ["provider"]
)


def render_latest() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
