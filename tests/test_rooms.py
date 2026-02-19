"""Tests for room CRUD and isolation.

Verifies that rooms are properly scoped to users and that data from one room
doesn't leak into another.
"""
import pytest
from httpx import AsyncClient

from tests.conftest import AUTH_HEADERS, AUTH_HEADERS_USER2


@pytest.mark.asyncio
async def test_create_room(client: AsyncClient):
    resp = await client.post(
        "/api/rooms",
        json={"name": "My Research Room", "description": "Test room", "color_index": 2},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "My Research Room"
    assert data["description"] == "Test room"
    assert data["color_index"] == 2
    assert data["paper_count"] == 0
    assert isinstance(data["id"], int)


@pytest.mark.asyncio
async def test_list_rooms_returns_only_own_rooms(client: AsyncClient):
    """Each user should only see their own rooms."""
    # User 1 creates two rooms
    await client.post("/api/rooms", json={"name": "Room A"}, headers=AUTH_HEADERS)
    await client.post("/api/rooms", json={"name": "Room B"}, headers=AUTH_HEADERS)

    # User 2 creates one room
    await client.post("/api/rooms", json={"name": "Room C"}, headers=AUTH_HEADERS_USER2)

    # User 1 should see 2 rooms
    resp = await client.get("/api/rooms", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    rooms = resp.json()
    names = {r["name"] for r in rooms}
    assert "Room A" in names
    assert "Room B" in names
    assert "Room C" not in names

    # User 2 should see 1 room
    resp2 = await client.get("/api/rooms", headers=AUTH_HEADERS_USER2)
    assert resp2.status_code == 200
    rooms2 = resp2.json()
    names2 = {r["name"] for r in rooms2}
    assert "Room C" in names2
    assert "Room A" not in names2


@pytest.mark.asyncio
async def test_get_room_detail(client: AsyncClient):
    resp = await client.post(
        "/api/rooms",
        json={"name": "Detail Room"},
        headers=AUTH_HEADERS,
    )
    room_id = resp.json()["id"]

    resp = await client.get(f"/api/rooms/{room_id}", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Detail Room"
    assert data["paper_count"] == 0


@pytest.mark.asyncio
async def test_delete_room(client: AsyncClient):
    resp = await client.post(
        "/api/rooms",
        json={"name": "Delete Me"},
        headers=AUTH_HEADERS,
    )
    room_id = resp.json()["id"]

    del_resp = await client.delete(f"/api/rooms/{room_id}", headers=AUTH_HEADERS)
    assert del_resp.status_code == 204

    # Should be gone now
    get_resp = await client.get(f"/api/rooms/{room_id}", headers=AUTH_HEADERS)
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_room_concept_map_empty(client: AsyncClient):
    """A fresh room's concept map should return zero nodes/edges/gaps."""
    resp = await client.post(
        "/api/rooms",
        json={"name": "Empty Map Room"},
        headers=AUTH_HEADERS,
    )
    room_id = resp.json()["id"]

    map_resp = await client.get(f"/api/rooms/{room_id}/concepts/map", headers=AUTH_HEADERS)
    assert map_resp.status_code == 200
    data = map_resp.json()
    assert data["nodes"] == []
    assert data["edges"] == []
    assert data["gaps"] == []
    assert data["metadata"]["total_concepts"] == 0


@pytest.mark.asyncio
async def test_room_brain_status_empty(client: AsyncClient):
    """A fresh room's brain status should show zeroes."""
    resp = await client.post(
        "/api/rooms",
        json={"name": "Brain Status Room"},
        headers=AUTH_HEADERS,
    )
    room_id = resp.json()["id"]

    status_resp = await client.get(f"/api/rooms/{room_id}/brain/status", headers=AUTH_HEADERS)
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["doc_count"] == 0
    assert data["concept_count"] == 0
    assert data["has_brain"] is False
