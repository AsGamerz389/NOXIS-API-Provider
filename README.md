# NOXIS — Unified AI Gateway

NOXIS is a self-hosted, OpenAI-compatible AI gateway that unifies legitimate
AI providers — both administrator-supplied API-key providers and genuinely
public, unauthenticated ("keyless") endpoints — behind one API, with
intelligent routing, failover, rate-limit awareness, and observability.

> **NOXIS aggregates legitimate provider capacity. It does not bypass
> quotas or create unlimited access.**
>
> It never implements CAPTCHA/authentication bypass, credential theft,
> account farming, token extraction, quota evasion, IP rotation,
> private-session scraping, or paid-usage circumvention. Keyless
> providers are only used when they are genuinely public and
> policy-compliant, and their limits are tracked and respected like any
> other provider's.

---

## 1. Architecture Overview

```
                              Client (OpenAI SDK / curl)
                                        |
                                   Authentication
                                        |
                              Gateway rate limiting
                                        |
                        ┌───────────────┴────────────────┐
                        │        Chat / Models API        │
                        └───────────────┬────────────────┘
                                        |
                                 Intelligent Router  ← weighted scoring
                                        |
                     ┌──────────────────┼──────────────────┐
                     |                  |                  |
              Rate Limiter      Circuit Breaker      Health Monitor
                     |                  |                  |
                     └──────────────────┼──────────────────┘
                                        |
                                    Scheduler         ← retries + backoff
                                        |
                     ┌──────────────────┼──────────────────┐
                     |                  |                  |
              KEYLESS providers   API_KEY providers   CUSTOM providers
              (Pollinations, ...) (Groq, Mistral, ...) (via providers.yaml)
```

Every provider — keyless or credentialed — implements the same `Provider`
interface, so the router, rate limiter, circuit breaker, and health
monitor treat them uniformly. The only structural difference is whether
an `Authorization` header is attached.

## 2. Directory Tree

```
noxis/
├── app/
│   ├── main.py                  FastAPI app, lifespan wiring, middleware
│   ├── config.py                Env-driven settings (no hard-coded secrets)
│   ├── api/
│   │   ├── chat.py              POST /v1/chat/completions (+ streaming)
│   │   ├── models.py            GET /v1/models
│   │   ├── embeddings.py        POST /v1/embeddings
│   │   ├── health.py            /health, /v1/providers, /v1/status, /metrics
│   │   ├── admin.py             /admin/* (master-key protected)
│   │   └── deps.py              Shared auth dependency
│   ├── core/
│   │   ├── router.py            Weighted-scoring intelligent router
│   │   ├── scheduler.py         Retry/failover orchestration, streaming
│   │   ├── rate_limiter.py      Per-provider + per-key rate tracking
│   │   ├── failover.py          Circuit breaker + backoff
│   │   ├── health_monitor.py    Background health checks
│   │   ├── cache.py             In-memory / Redis response cache
│   │   ├── security.py          Auth, key hashing, SSRF guard, headers
│   │   ├── usage.py             Usage/audit persistence
│   │   └── metrics.py           Prometheus metric definitions
│   ├── providers/
│   │   ├── base.py              Provider interface + dataclasses
│   │   ├── registry.py          Builds providers from env + providers.yaml
│   │   ├── openai_compatible.py Generic adapter (keyless + API-key)
│   │   └── adapters/            Slot for bespoke, non-OpenAI-shaped providers
│   ├── models/
│   │   ├── schemas.py           OpenAI-compatible Pydantic schemas
│   │   └── registry.py          Model listing + alias resolution
│   ├── database/
│   │   ├── database.py          Async SQLAlchemy engine/session
│   │   └── models.py            UsageRecord, AuditLogRecord (no message content)
│   └── utils/
│       ├── logging.py           Structured JSON logs + secret redaction
│       ├── ids.py                ID generation
│       └── errors.py            OpenAI-style error taxonomy
├── tests/                       pytest + respx mocked-provider tests
├── dashboard/index.html         Lightweight live dashboard (static, no build step)
├── providers.yaml               Optional extra OpenAI-compatible providers
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── pytest.ini
├── LICENSE
└── README.md
```

## 3. Dependencies

See `requirements.txt`: FastAPI, Uvicorn, httpx, Pydantic v2, SQLAlchemy
(async, SQLite via `aiosqlite` or Postgres via `asyncpg`), `redis` (optional),
`prometheus-client`, `PyYAML`, `sse-starlette` (SSE streaming), `pytest` +
`pytest-asyncio` + `respx` for testing.

## 4. Configuration Design

All configuration is environment-driven (`pydantic-settings`, loaded from
`.env`). **Nothing is hard-coded.** Provider credentials are optional; a
provider is only active if:

- it's an `API_KEY` provider **and** its `*_ENABLED=true` **and** its API
  key env var is set (otherwise NOXIS logs a warning and skips it — it
  never runs with a missing/fake credential), or
- it's a `KEYLESS` provider **and** its `*_ENABLED=true` **and** it passes
  startup verification (see §9).

**NOXIS starts with only one provider configured by default**
(`POLLINATIONS_ENABLED=true`), so a fresh clone with an empty `.env` still
boots and serves requests.

Administrators can also extend NOXIS with arbitrary OpenAI-compatible
endpoints via `providers.yaml`, without touching code:

```yaml
providers:
  - name: my_provider
    type: api_key          # keyless | api_key | oauth | custom
    base_url: https://example.com/v1
    api_key_env: MY_PROVIDER_API_KEY
    enabled: true
    priority: 25
    models:
      - name: my-model
        context_length: 32768
```

## 5. Provider Abstraction

```python
class Provider:
    name: str
    provider_type: str        # keyless | api_key | oauth | custom
    requires_api_key: bool

    async def chat_completion(self, request): ...
    async def stream_completion(self, request): ...
    async def list_models(self): ...
    async def health_check(self): ...
    async def get_limits(self): ...
```

`OpenAICompatibleProvider` implements this for any OpenAI-shaped upstream
(which covers every built-in provider). It:

- never sends `Authorization: Bearer None` or a fabricated credential —
  the header is omitted entirely when no key is configured;
- classifies upstream errors into retryable (429/500/502/503/504,
  timeouts, connection errors) vs terminal (everything else);
- honors `Retry-After` when present;
- proxies SSE streams and normalizes chunks into OpenAI's shape.

## 6. Routing Design

`GET /v1/models` lists concrete models plus the NOXIS aliases:
`noxis-auto`, `noxis-fast`, `noxis-smart`, `noxis-free`.

- **noxis-auto** — considers every enabled, compatible provider.
- **noxis-fast** — sorted by administrator priority (lowest number first).
- **noxis-smart** — prefers credentialed (`API_KEY`) providers when available.
- **noxis-free** — prefers providers that can serve the request **without**
  an administrator-supplied paid credential (keyless providers). This
  never means "unlimited" — see §9.

### Scoring algorithm

For each health-eligible, quota-eligible, capability-eligible, and
context-length-eligible candidate:

```
score = 0.30 * health_score
      + 0.15 * priority_score
      + 0.15 * latency_score
      + 0.20 * reliability_score
      + 0.15 * quota_headroom_score
      + 0.05 * capability_score
```

All sub-scores are normalized to `[0, 1]`; the full derivation lives as a
docstring in `app/core/router.py`. Candidates are excluded outright (not
merely down-weighted) if they are rate-limited, circuit-open, missing a
required capability, or their model's context length can't fit the
request. A provider with no configured model list (several built-ins
default to dynamic/empty model lists — e.g. `openrouter`, `nvidia`,
`cohere`, `github_models`, `huggingface`) is also excluded from **alias**
routing specifically, since NOXIS has no safe upstream model name to
resolve `noxis-auto` etc. to for it — such a provider still works fine
when addressed by its exact model name directly. List its models
explicitly (built-in credential env vars only enable the provider; add
model names via `providers.yaml` if you want it eligible for aliases) to
opt it into alias-based routing. The remaining candidates are sorted by
score, and the scheduler tries them in order with bounded retries and
exponential backoff + jitter.

**Streaming never switches providers mid-response**: once a provider has
emitted its first chunk, a subsequent failure ends the stream with an
error event rather than silently retrying elsewhere.

---

## 7. Installation

```bash
git clone <this-repo> noxis && cd noxis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# generate a master key
python -c "import secrets; print(secrets.token_urlsafe(32))"
# paste it into NOXIS_MASTER_KEY in .env
```

## 8. Startup

```bash
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/` → `{"name": "NOXIS", "status": "online", ...}`,
and `http://localhost:8000/dashboard/` for the live dashboard.

## 9. Keyless Provider Layer

Keyless providers are first-class, not an afterthought. At startup NOXIS:

1. Checks each enabled keyless provider's reachability.
2. Heuristically confirms it doesn't now require authentication.
3. Attempts model discovery via `GET /models` (falls back to
   manually configured models if unsupported).
4. Marks unreachable/unverifiable providers `UNAVAILABLE` — it never
   attempts to work around a provider that now requires auth.

**Keyless does not mean unlimited.** Requests/minute, requests/day,
concurrency, 429s, and `Retry-After` are tracked per provider exactly like
credentialed providers, and a keyless provider that's rate-limited or
repeatedly failing is skipped by the router and given a cooldown before
being retried — never bypassed via IP/proxy/identity rotation or CAPTCHA
circumvention, which NOXIS does not implement under any configuration.

Enable additional keyless providers in `.env`:

```
LLM7_ENABLED=true
KEYLESSAI_ENABLED=true
```

Because keyless endpoints are public, NOXIS itself defaults to requiring
authentication (`ALLOW_UNAUTHENTICATED=false`) so your NOXIS instance
doesn't become an unrestricted public relay for someone else's free tier.

## 10. Docker

```bash
docker build -t noxis .
docker run --env-file .env -p 8000:8000 noxis
```

Or with Compose (includes optional Redis/Postgres):

```bash
docker compose up --build
# with Postgres too:
docker compose --profile postgres up --build
```

## 11. Render

`render.yaml` is included — connect the repo in the Render dashboard, or:

```bash
render blueprint launch
```

It provisions a Docker web service, auto-generates `NOXIS_MASTER_KEY`, and
leaves provider credentials as `sync: false` (set them in the Render
dashboard, never in the repo).

## 12. Python Client Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YOUR-NOXIS-KEY",
)

response = client.chat.completions.create(
    model="noxis-auto",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

## 13. JavaScript Client Usage

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "YOUR-NOXIS-KEY",
});

const response = await client.chat.completions.create({
  model: "noxis-free",
  messages: [{ role: "user", content: "Hello" }],
});
console.log(response.choices[0].message.content);
```

## 14. Streaming

```python
stream = client.chat.completions.create(
    model="noxis-auto",
    messages=[{"role": "user", "content": "Write a haiku about routers."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

Wire format is standard SSE:

```
data: {"id": "...", "object": "chat.completion.chunk", ...}

data: [DONE]
```

## 15. Adding a Provider

**API-key provider (built-in):** set `NAME_API_KEY` and `NAME_ENABLED=true`
in `.env`.

**Any other OpenAI-compatible provider:** add an entry to `providers.yaml`
— no code changes required.

**A provider with a non-OpenAI wire format:** subclass `Provider` in
`app/providers/adapters/`, implement `chat_completion` /
`stream_completion` / `list_models` / `health_check`, and register it in
`app/providers/registry.py`.

## 16. Security

- Auth: `Authorization: Bearer <key>`. Keys are compared by SHA-256 hash,
  never stored or logged raw.
- Admin endpoints (`/admin/*`) require the `NOXIS_MASTER_KEY` specifically.
- SSRF guard (`app/core/security.py`) resolves and rejects
  private/loopback/link-local hosts for any admin-configurable provider URL.
- Structured JSON logs redact `Authorization` headers, `sk-...`-shaped
  tokens, and any field named `api_key`/`token`/`secret`/`password`.
- `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy` security
  headers on every response.
- Configurable CORS (`CORS_ALLOW_ORIGINS`).
- Every response carries `x-request-id`.
- Gateway-level per-key rate limiting and a bounded global concurrency
  semaphore, independent of per-provider limits.
- Message content is **not** persisted in the usage table by default —
  only metadata (provider, model, latency, tokens, status).

## 17. Testing

```bash
pytest
```

Covers (via `respx`-mocked HTTP, no real network calls): successful
requests, timeouts, 429/500/503 handling + `Retry-After`, authentication,
invalid models/requests, disabled providers, total-failure → 503,
streaming normalization, rate limiting, circuit breaker state transitions
(`CLOSED → OPEN → HALF_OPEN → CLOSED`), weighted routing decisions,
keyless-specific behavior (no `Authorization` header ever sent, never
`Bearer None`), and admin-endpoint auth boundaries.

## 18. Troubleshooting

- **`503 NOXIS_ALL_PROVIDERS_FAILED`**: no enabled provider is currently
  healthy/within quota for the requested model. Check `GET /v1/providers`
  and `GET /v1/status`.
- **A keyless provider shows `UNAVAILABLE`**: it failed the startup
  reachability check, or now appears to require authentication that NOXIS
  won't bypass. Check its current documentation and, if it now requires a
  key, configure `PROVIDER_API_KEY` and flip it to an `API_KEY` entry in
  `providers.yaml`.
- **429s from a keyless provider**: expected under load — NOXIS reads
  `Retry-After`, cools that provider down, and routes elsewhere. This is
  the provider's real limit, not a NOXIS bug.
- **Streaming client hangs**: confirm the provider actually supports
  `stream=true`; if not, the router should have skipped it — check
  `supports_streaming` in `/v1/providers`.

## 19. Extension Points

- New providers: `providers.yaml` (zero code) or `app/providers/adapters/`
  (custom wire formats).
- New routing strategies: adjust weights or add sub-scores in
  `app/core/router.py` — the scoring function is isolated and unit-tested.
- New storage backends: swap `DATABASE_URL` for Postgres; swap the cache
  backend to Redis via `CACHE_BACKEND=redis` + `REDIS_URL`.
