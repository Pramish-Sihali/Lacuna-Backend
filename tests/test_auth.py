"""Tests for authentication boundaries.

Verifies that room-scoped endpoints require X-User-Id and that
users cannot access other users' rooms.
"""
import pytest
from httpx import AsyncClient

from tests.conftest import AUTH_HEADERS, AUTH_HEADERS_USER2


@pytest.mark.asyncio
async def test_rooms_requires_auth_header(client: AsyncClient):
    """GET /api/rooms without X-User-Id should return 422 (missing required header)."""
    resp = await client.get("/api/rooms")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_room_requires_auth_header(client: AsyncClient):
    """POST /api/rooms without X-User-Id should return 422."""
    resp = await client.post("/api/rooms", json={"name": "Unauthed Room"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_wrong_user_cannot_access_room(client: AsyncClient):
    """User 2 should get 404 when accessing user 1's room."""
    # Create room as user 1
    resp = await client.post(
        "/api/rooms",
        json={"name": "Private Room"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 201
    room_id = resp.json()["id"]

    # User 2 tries to access it
    resp = await client.get(f"/api/rooms/{room_id}", headers=AUTH_HEADERS_USER2)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_wrong_user_cannot_delete_room(client: AsyncClient):
    """User 2 should get 404 when trying to delete user 1's room."""
    resp = await client.post(
        "/api/rooms",
        json={"name": "Another Private Room"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 201
    room_id = resp.json()["id"]

    resp = await client.delete(f"/api/rooms/{room_id}", headers=AUTH_HEADERS_USER2)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_nonexistent_room_returns_404(client: AsyncClient):
    """Accessing a non-existent room ID should return 404."""
    resp = await client.get("/api/rooms/999999", headers=AUTH_HEADERS)
    assert resp.status_code == 404
