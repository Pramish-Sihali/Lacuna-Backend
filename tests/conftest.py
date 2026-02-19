"""
Shared fixtures for Lacuna backend integration tests.

Uses the test database (lacuna_test_db on port 5433) provisioned by docker-compose.
Each test function gets its own session. Tables are created via init-db.sql
(Docker entrypoint) and also via create_all in a per-test setup to be safe.
Data is cleaned between tests by truncating all tables.
"""
from __future__ import annotations

import os
from typing import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

# Override DATABASE_URL *before* any app module is imported, so that
# settings.DATABASE_URL and the global engine point at the test DB.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://lacuna_user:lacuna_password@localhost:5433/lacuna_test_db",
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL

from app.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402

# Table names in dependency order (children first) for truncation
_ALL_TABLES = [
    "brain_state",
    "relationships",
    "claims",
    "concepts",
    "chunks",
    "documents",
    "projects",
    "users",
]


# ---------------------------------------------------------------------------
# Per-test fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a DB session for each test. After the test, all tables are truncated
    so each test starts with a clean slate.
    """
    engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=NullPool)

    # Ensure tables exist
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    # Truncate all tables after the test
    async with engine.begin() as conn:
        for table in _ALL_TABLES:
            await conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))

    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    httpx AsyncClient wired to the FastAPI app with the DB dependency
    overridden to use the per-test session.
    """

    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AUTH_HEADERS = {
    "X-User-Id": "test-user-1",
    "X-User-Email": "test1@example.com",
    "X-User-Name": "Test User 1",
}

AUTH_HEADERS_USER2 = {
    "X-User-Id": "test-user-2",
    "X-User-Email": "test2@example.com",
    "X-User-Name": "Test User 2",
}
