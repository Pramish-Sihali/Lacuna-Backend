"""Tests for concept map and build endpoints."""
import pytest
from httpx import AsyncClient

from tests.conftest import AUTH_HEADERS


async def _create_room(client: AsyncClient, name: str = "Concept Room") -> int:
    resp = await client.post("/api/rooms", json={"name": name}, headers=AUTH_HEADERS)
    assert resp.status_code == 201
    return resp.json()["id"]


@pytest.mark.asyncio
async def test_concept_map_structure(client: AsyncClient):
    """The concept map response should have the expected ReactFlow shape."""
    room_id = await _create_room(client)
    resp = await client.get(f"/api/rooms/{room_id}/concepts/map", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()

    # Top-level keys
    assert "nodes" in data
    assert "edges" in data
    assert "gaps" in data
    assert "metadata" in data

    # Metadata shape
    meta = data["metadata"]
    assert "total_concepts" in meta
    assert "total_relationships" in meta
    assert "total_gaps" in meta
    assert "num_clusters" in meta
    assert "has_clustering" in meta


@pytest.mark.asyncio
async def test_concept_map_isolation(client: AsyncClient):
    """Two rooms should have independent concept maps."""
    room1 = await _create_room(client, "Map Room 1")
    room2 = await _create_room(client, "Map Room 2")

    map1 = await client.get(f"/api/rooms/{room1}/concepts/map", headers=AUTH_HEADERS)
    map2 = await client.get(f"/api/rooms/{room2}/concepts/map", headers=AUTH_HEADERS)

    assert map1.status_code == 200
    assert map2.status_code == 200

    # Both should be empty and independent
    assert map1.json()["metadata"]["total_concepts"] == 0
    assert map2.json()["metadata"]["total_concepts"] == 0


@pytest.mark.asyncio
async def test_legacy_concept_map_endpoint(client: AsyncClient):
    """The legacy (unscoped) concept map endpoint should still work."""
    resp = await client.get("/api/concepts/map")
    # Should return 200 even if there are no concepts (returns empty map)
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
