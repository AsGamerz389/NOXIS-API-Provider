"""
NOXIS configuration.

All secrets are loaded from environment variables (.env). Nothing here is
hard-coded. Provider credentials are optional; providers without a
configured key are either disabled (for API_KEY-type providers) or run as
KEYLESS (only if the provider is explicitly declared keyless).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Core ---
    APP_NAME: str = "NOXIS"
    APP_VERSION: str = "1.0.0"
    PORT: int = Field(default=8000)
    ENVIRONMENT: Literal["development", "production"] = "development"
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True

    # --- Auth ---
    NOXIS_MASTER_KEY: str | None = None
    # Comma-separated list of additional gateway API keys (issued to clients)
    NOXIS_GATEWAY_KEYS: str = ""
    ALLOW_UNAUTHENTICATED: bool = False

    # --- Database ---
    DATABASE_URL: str = "sqlite+aiosqlite:///./noxis.db"

    # --- Redis (optional; used for distributed rate limiting / cache) ---
    REDIS_URL: str | None = None

    # --- Cache ---
    CACHE_ENABLED: bool = False
    CACHE_TTL: int = 300
    CACHE_BACKEND: Literal["memory", "redis"] = "memory"

    # --- HTTP client ---
    CONNECT_TIMEOUT: float = 5.0
    READ_TIMEOUT: float = 60.0
    WRITE_TIMEOUT: float = 10.0
    TOTAL_TIMEOUT: float = 90.0
    MAX_CONNECTIONS: int = 200
    MAX_KEEPALIVE_CONNECTIONS: int = 50

    # --- Request limits ---
    MAX_REQUEST_BYTES: int = 2_000_000
    MAX_TOKENS_CEILING: int = 32000
    GATEWAY_RATE_LIMIT_PER_MIN: int = 120
    GATEWAY_MAX_CONCURRENCY: int = 100

    # --- Failover ---
    MAX_RETRIES: int = 3
    BACKOFF_BASE_SECONDS: float = 0.5
    BACKOFF_MAX_SECONDS: float = 8.0
    CIRCUIT_FAILURE_THRESHOLD: int = 5
    CIRCUIT_COOLDOWN_SECONDS: float = 30.0
    CIRCUIT_HALF_OPEN_MAX_CALLS: int = 1

    # --- CORS ---
    CORS_ALLOW_ORIGINS: str = "*"

    # --- Provider configuration file ---
    PROVIDERS_CONFIG_PATH: str = "providers.yaml"

    # === Provider credentials (all optional; presence enables the provider) ===
    GROQ_API_KEY: str | None = None
    GROQ_ENABLED: bool = False

    CEREBRAS_API_KEY: str | None = None
    CEREBRAS_ENABLED: bool = False

    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_ENABLED: bool = False

    MISTRAL_API_KEY: str | None = None
    MISTRAL_ENABLED: bool = False

    GEMINI_API_KEY: str | None = None
    GEMINI_ENABLED: bool = False

    DEEPSEEK_API_KEY: str | None = None
    DEEPSEEK_ENABLED: bool = False

    NVIDIA_API_KEY: str | None = None
    NVIDIA_ENABLED: bool = False

    COHERE_API_KEY: str | None = None
    COHERE_ENABLED: bool = False

    GITHUB_TOKEN: str | None = None
    GITHUB_MODELS_ENABLED: bool = False

    HF_TOKEN: str | None = None
    HUGGINGFACE_ENABLED: bool = False

    # === Keyless providers ===
    POLLINATIONS_ENABLED: bool = True
    POLLINATIONS_BASE_URL: str = "https://text.pollinations.ai"

    KEYLESSAI_ENABLED: bool = True
    KEYLESSAI_BASE_URL: str = "https://keylessai.thryx.workers.dev/v1"

    # LLM7 currently documents token-based access for its higher-limit API.
    # Keep it opt-in so NOXIS never assumes token-required access is keyless.
    LLM7_ENABLED: bool = False
    LLM7_BASE_URL: str = "https://api.llm7.io/v1"

    DEVTOOLBOX_ENABLED: bool = False
    DEVTOOLBOX_BASE_URL: str = ""

    INFERENCEPORT_ENABLED: bool = False
    INFERENCEPORT_BASE_URL: str = ""

    @property
    def gateway_keys(self) -> list[str]:
        keys = [k.strip() for k in self.NOXIS_GATEWAY_KEYS.split(",") if k.strip()]
        if self.NOXIS_MASTER_KEY:
            keys.append(self.NOXIS_MASTER_KEY)
        return keys

    @property
    def cors_origins(self) -> list[str]:
        if self.CORS_ALLOW_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ALLOW_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
