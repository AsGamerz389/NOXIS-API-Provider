from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import admin, chat, embeddings, health, models
from app.config import get_settings
from app.core.cache import InMemoryCache, RedisCache
from app.core.failover import CircuitBreaker
from app.core.health_monitor import HealthMonitor
from app.core.rate_limiter import GatewayRateLimiter, RateLimiter
from app.core.router import Router
from app.core.scheduler import Scheduler
from app.core.security import KeyAuthenticator, SecurityHeadersMiddleware
from app.core.usage import UsageTracker
from app.database.database import Database
from app.providers.registry import ProviderRegistry
from app.utils.errors import NoxisError
from app.utils.ids import gen_request_id
from app.utils.logging import configure_logging, log_event

logger = logging.getLogger("noxis.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL, settings.LOG_JSON)
    log_event(logger, logging.INFO, "starting_noxis", version=settings.APP_VERSION, env=settings.ENVIRONMENT)

    limits = httpx.Limits(
        max_connections=settings.MAX_CONNECTIONS,
        max_keepalive_connections=settings.MAX_KEEPALIVE_CONNECTIONS,
    )
    timeout = httpx.Timeout(
        connect=settings.CONNECT_TIMEOUT, read=settings.READ_TIMEOUT,
        write=settings.WRITE_TIMEOUT, pool=settings.TOTAL_TIMEOUT,
    )
    client = httpx.AsyncClient(limits=limits, timeout=timeout)

    registry = ProviderRegistry(settings, client)
    registry.build()
    await registry.verify_keyless_providers()
    await registry.discover_keyless_models()

    rate_limiter = RateLimiter()
    breaker = CircuitBreaker(
        failure_threshold=settings.CIRCUIT_FAILURE_THRESHOLD,
        cooldown_seconds=settings.CIRCUIT_COOLDOWN_SECONDS,
        half_open_max_calls=settings.CIRCUIT_HALF_OPEN_MAX_CALLS,
    )
    router_ = Router(registry, rate_limiter)
    scheduler = Scheduler(
        router_, breaker, rate_limiter,
        max_retries=settings.MAX_RETRIES,
        backoff_base=settings.BACKOFF_BASE_SECONDS,
        backoff_cap=settings.BACKOFF_MAX_SECONDS,
    )
    health_monitor = HealthMonitor(registry, rate_limiter)
    health_monitor.start()

    db = Database(settings.DATABASE_URL)
    await db.init()
    usage = UsageTracker(db.session_factory)

    authenticator = KeyAuthenticator(settings)
    gateway_limiter = GatewayRateLimiter(settings.GATEWAY_RATE_LIMIT_PER_MIN)

    if settings.CACHE_ENABLED and settings.CACHE_BACKEND == "redis" and settings.REDIS_URL:
        import redis.asyncio as aioredis
        redis_client = aioredis.from_url(settings.REDIS_URL)
        cache = RedisCache(redis_client, settings.CACHE_TTL)
    else:
        cache = InMemoryCache(settings.CACHE_TTL)

    app.state.settings = settings
    app.state.http_client = client
    app.state.registry = registry
    app.state.rate_limiter = rate_limiter
    app.state.breaker = breaker
    app.state.router = router_
    app.state.scheduler = scheduler
    app.state.health_monitor = health_monitor
    app.state.db = db
    app.state.usage = usage
    app.state.authenticator = authenticator
    app.state.gateway_limiter = gateway_limiter
    app.state.cache = cache
    app.state.concurrency_semaphore = asyncio.Semaphore(settings.GATEWAY_MAX_CONCURRENCY)

    log_event(logger, logging.INFO, "noxis_ready",
              providers=[p.name for p in registry.all()], enabled=[p.name for p in registry.enabled()])

    yield

    await health_monitor.stop()
    await client.aclose()
    await db.dispose()


app = FastAPI(
    title="NOXIS Unified AI Gateway",
    version=get_settings().APP_VERSION,
    lifespan=lifespan,
)

_settings_for_cors = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings_for_cors.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or gen_request_id()
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log_event(logger, logging.ERROR, "unhandled_request_error", request_id=request_id, path=str(request.url.path))
        raise
    response.headers["x-request-id"] = request_id
    log_event(
        logger, logging.INFO, "request_completed",
        request_id=request_id, path=str(request.url.path), method=request.method,
        status=response.status_code, latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )
    return response


@app.exception_handler(NoxisError)
async def noxis_error_handler(request: Request, exc: NoxisError):
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    # Never leak stack traces to the client.
    log_event(logger, logging.ERROR, "unhandled_exception", path=str(request.url.path), error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "An internal error occurred.", "type": "internal_error",
                            "code": "NOXIS_INTERNAL_ERROR"}},
    )


@app.get("/")
async def root():
    return {"name": "NOXIS", "status": "online", "version": get_settings().APP_VERSION}


app.include_router(chat.router)
app.include_router(models.router)
app.include_router(embeddings.router)
app.include_router(health.router)
app.include_router(admin.router, prefix="")

try:
    app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")
except RuntimeError:
    pass  # dashboard directory not present (e.g. in some test environments)
