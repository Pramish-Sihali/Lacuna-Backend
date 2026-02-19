"""Tests for document upload, list, and delete within rooms."""
import io

import pytest
from httpx import AsyncClient

from tests.conftest import AUTH_HEADERS, AUTH_HEADERS_USER2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_dummy_pdf() -> bytes:
    """Return minimal valid PDF bytes (1 blank page)."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n190\n%%EOF\n"
    )


async def _create_room(client: AsyncClient, name: str = "Doc Room", headers=None) -> int:
    headers = headers or AUTH_HEADERS
    resp = await client.post("/api/rooms", json={"name": name}, headers=headers)
    assert resp.status_code == 201
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_unsupported_type(client: AsyncClient):
    """Uploading a .txt file should fail with 400."""
    room_id = await _create_room(client)
    resp = await client.post(
        f"/api/rooms/{room_id}/documents/upload",
        headers=AUTH_HEADERS,
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_upload_empty_pdf(client: AsyncClient):
    """A minimal PDF with no text should return 422."""
    room_id = await _create_room(client)
    pdf_bytes = _create_dummy_pdf()
    resp = await client.post(
        f"/api/rooms/{room_id}/documents/upload",
        headers=AUTH_HEADERS,
        files={"file": ("empty.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    # Minimal PDF has no text → 422 "no extractable text"
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_list_documents_empty(client: AsyncClient):
    """A fresh room should have no documents."""
    room_id = await _create_room(client)
    resp = await client.get(f"/api/rooms/{room_id}/documents/", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_delete_nonexistent_document(client: AsyncClient):
    """Deleting a document that doesn't exist should return 404."""
    room_id = await _create_room(client)
    resp = await client.delete(
        f"/api/rooms/{room_id}/documents/999999",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cross_room_document_isolation(client: AsyncClient):
    """Documents in one room should not appear in another room's list."""
    room1 = await _create_room(client, "Iso Room 1")
    room2 = await _create_room(client, "Iso Room 2")

    # Both rooms should start empty
    r1 = await client.get(f"/api/rooms/{room1}/documents/", headers=AUTH_HEADERS)
    r2 = await client.get(f"/api/rooms/{room2}/documents/", headers=AUTH_HEADERS)
    assert r1.json() == []
    assert r2.json() == []
