"""
Authentication dependencies for FastAPI routes.

Extracts user identity from the X-User-Id header (set by the Next.js frontend).
Demo mode (no header) falls back to DEFAULT_PROJECT_ID.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.database_models import Project, User

logger = logging.getLogger(__name__)


async def get_current_user_id(
    x_user_id: str = Header(..., alias="X-User-Id"),
) -> str:
    """Extract the authenticated user ID from the request header. Raises 401 if missing."""
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-User-Id header.",
        )
    return x_user_id


async def get_optional_user_id(
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
) -> Optional[str]:
    """Extract user ID if present, return None for demo mode."""
    return x_user_id or None


async def get_or_create_user(
    user_id: str = Depends(get_current_user_id),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_user_name: Optional[str] = Header(None, alias="X-User-Name"),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Ensure the user exists in the local users table. Creates if needed."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        user = User(
            id=user_id,
            email=x_user_email or f"{user_id}@lacuna.local",
            name=x_user_name,
        )
        db.add(user)
        await db.flush()
        logger.info("Created new user: id=%s email=%s", user_id, user.email)

    return user


async def get_authorized_project(
    room_id: int,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> Project:
    """
    Verify that the given room (project) belongs to the current user.
    Returns the Project ORM object or raises 404.
    """
    result = await db.execute(
        select(Project).where(
            Project.id == room_id,
            Project.user_id == user_id,
        )
    )
    project = result.scalar_one_or_none()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Room {room_id} not found.",
        )

    return project
