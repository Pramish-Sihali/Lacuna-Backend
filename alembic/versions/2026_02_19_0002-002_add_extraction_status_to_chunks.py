"""add extraction_status to chunks

Revision ID: 002
Revises: 001
Create Date: 2026-02-19 20:30:00

The chunks table was created before extraction_status was added to the
SQLAlchemy model.  This migration adds the column to the live database.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column(
            "extraction_status",
            sa.String(length=50),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("chunks", "extraction_status")
