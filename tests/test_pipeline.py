"""Tests for pipeline endpoints."""
import pytest
from httpx import AsyncClient

from tests.conftest import AUTH_HEADERS


async def _create_room(client: AsyncClient, name: str = "Pipeline Room") -> int:
    resp = await client.post("/api/rooms", json={"name": name}, headers=AUTH_HEADERS)
    assert resp.status_code == 201
    return resp.json()["id"]


@pytest.mark.asyncio
async def test_process_all_empty_room(client: AsyncClient):
    """process-all on a room with no documents should return gracefully."""
    room_id = await _create_room(client)
    resp = await client.post(
        f"/api/rooms/{room_id}/pipeline/process-all",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["documents_processed"] == 0
    assert data["documents_failed"] == 0
    assert "No documents" in data["message"]


@pytest.mark.asyncio
async def test_build_knowledge_empty_room(client: AsyncClient):
    """Building knowledge on an empty room should succeed with zeroes."""
    room_id = await _create_room(client)
    resp = await client.post(
        f"/api/rooms/{room_id}/concepts/build",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["project_id"] == room_id
    assert isinstance(data["processing_time_seconds"], (int, float))


@pytest.mark.asyncio
async def test_pipeline_status_idle(client: AsyncClient):
    """Pipeline status for a room with no running pipeline should be idle."""
    room_id = await _create_room(client)
    resp = await client.get(
        f"/api/rooms/{room_id}/pipeline/status",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["phase"] == "idle"
