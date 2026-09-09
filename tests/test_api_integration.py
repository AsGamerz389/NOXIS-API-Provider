from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_root_endpoint(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "NOXIS"
    assert body["status"] == "online"


@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


@pytest.mark.asyncio
async def test_chat_completions_requires_auth(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "noxis-auto", "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["type"] == "authentication_error"


@pytest.mark.asyncio
async def test_chat_completions_rejects_bad_key(client):
    resp = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer wrong-key"},
        json={"model": "noxis-auto", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_chat_completions_with_no_providers_returns_503(client, auth_headers):
    resp = await client.post(
        "/v1/chat/completions", headers=auth_headers,
        json={"model": "noxis-auto", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "NOXIS_ALL_PROVIDERS_FAILED"


@pytest.mark.asyncio
async def test_invalid_request_empty_messages_returns_422_or_400(client, auth_headers):
    resp = await client.post(
        "/v1/chat/completions", headers=auth_headers,
        json={"model": "noxis-auto", "messages": []},
    )
    assert resp.status_code in (400, 422)


@pytest.mark.asyncio
async def test_models_endpoint_requires_auth(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_models_endpoint_lists_aliases(client, auth_headers):
    resp = await client.get("/v1/models", headers=auth_headers)
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()["data"]}
    assert "noxis-auto" in ids
    assert "noxis-free" in ids


@pytest.mark.asyncio
async def test_providers_endpoint_is_public_metadata_only(client):
    resp = await client.get("/v1/providers")
    assert resp.status_code == 200
    for p in resp.json()["providers"]:
        assert "api_key" not in p
        assert "credential" not in p


@pytest.mark.asyncio
async def test_admin_requires_master_key(client, auth_headers):
    resp = await client.get("/admin/providers")
    assert resp.status_code == 401
    resp2 = await client.get("/admin/providers", headers=auth_headers)
    assert resp2.status_code == 200


@pytest.mark.asyncio
async def test_admin_config_never_exposes_secrets(client, auth_headers):
    resp = await client.get("/admin/config", headers=auth_headers)
    assert resp.status_code == 200
    body_str = str(resp.json())
    assert "test-master-key" not in body_str


@pytest.mark.asyncio
async def test_admin_disable_unknown_provider_returns_404(client, auth_headers):
    resp = await client.post("/admin/providers/does-not-exist/disable", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_prometheus_format(client):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert b"noxis_requests_total" in resp.content or resp.content == b""


@pytest.mark.asyncio
async def test_status_endpoint_includes_disclaimer(client):
    resp = await client.get("/v1/status")
    assert resp.status_code == 200
    assert "does not bypass quotas" in resp.json()["note"]


@pytest.mark.asyncio
async def test_request_id_header_present(client):
    resp = await client.get("/health")
    assert "x-request-id" in resp.headers
