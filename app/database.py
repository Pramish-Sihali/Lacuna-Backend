"""
Database connection and session management.
Handles PostgreSQL + pgvector setup with SQLAlchemy async engine.
"""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from typing import AsyncGenerator
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,  # Set to True for SQL query logging
    future=True,
    pool_pre_ping=True,
    poolclass=NullPool,  # Use NullPool for better async compatibility
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for ORM models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function to get database session.

    Yields:
        AsyncSession: Database session

    Example:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initialize database tables and pgvector extension.
    Creates all tables defined in Base metadata.
    """
    try:
        async with engine.begin() as conn:
            # Import all models to ensure they're registered
            from app.models import database_models  # noqa: F401

            # Create pgvector extension
            await conn.execute(
                text("CREATE EXTENSION IF NOT EXISTS vector")
            )
            logger.info("pgvector extension created/verified")

            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created/verified")

            # Fix auto-increment sequences for tables that may have
            # manually-inserted rows (e.g. DEFAULT_PROJECT_ID=1).
            # Without this, PostgreSQL's nextval() can return an ID that
            # already exists, causing IntegrityError on INSERT.
            for table in ("projects", "documents", "chunks", "concepts",
                          "claims", "relationships", "brain_state"):
                await conn.execute(text(
                    f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                    f"COALESCE((SELECT MAX(id) FROM {table}), 0) + 1, false)"
                ))
            logger.info("Auto-increment sequences synced")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


async def close_db() -> None:
    """Close database connections gracefully."""
    try:
        await engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")
        raise
