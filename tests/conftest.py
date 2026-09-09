from __future__ import annotations

import os

import pytest
import pytest_asyncio

os.environ.setdefault("NOXIS_MASTER_KEY", "test-master-key")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("POLLINATIONS_ENABLED", "false")  # tests build their own providers
os.environ.setdefault("LOG_JSON", "false")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test-master-key"}
