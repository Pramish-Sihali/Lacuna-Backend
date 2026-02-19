"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-02-19

All 8 tables as defined in app/models/database_models.py:
users, projects, documents, chunks, concepts, claims, relationships, brain_state.
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None

VECTOR_DIM = 768


def upgrade() -> None:
    # pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── Enum types ────────────────────────────────────────────────────────
    claim_type = sa.Enum("supports", "contradicts", "extends", "complements", name="claimtype")
    claim_type.create(op.get_bind(), checkfirst=True)

    relationship_type = sa.Enum(
        "prerequisite", "builds_on", "contradicts", "complements", "similar", "parent_child",
        name="relationshiptype",
    )
    relationship_type.create(op.get_bind(), checkfirst=True)

    gap_type = sa.Enum("missing_link", "under_explored", "contradictory", "isolated_concept", name="gaptype")
    gap_type.create(op.get_bind(), checkfirst=True)

    # ── users ─────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # ── projects ──────────────────────────────────────────────────────────
    op.create_table(
        "projects",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("user_id", sa.String(255), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True),
        sa.Column("color_index", sa.Integer, default=0),
    )

    # ── documents ─────────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("project_id", sa.Integer, sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("file_path", sa.String(512), nullable=False),
        sa.Column("file_type", sa.String(50), nullable=False),
        sa.Column("content_text", sa.Text, nullable=True),
        sa.Column("metadata_json", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # ── chunks ────────────────────────────────────────────────────────────
    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("document_id", sa.Integer, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("embedding", Vector(VECTOR_DIM), nullable=True),
        sa.Column("metadata_json", sa.JSON, nullable=True),
        sa.Column("extraction_status", sa.String(20), nullable=True),
    )

    # ── concepts ──────────────────────────────────────────────────────────
    op.create_table(
        "concepts",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("project_id", sa.Integer, sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(255), nullable=False, index=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("generality_score", sa.Float, nullable=True),
        sa.Column("coverage_score", sa.Float, nullable=True),
        sa.Column("consensus_score", sa.Float, nullable=True),
        sa.Column("embedding", Vector(VECTOR_DIM), nullable=True),
        sa.Column("is_gap", sa.Boolean, default=False, nullable=False, index=True),
        sa.Column("gap_type", sa.Enum("missing_link", "under_explored", "contradictory", "isolated_concept", name="gaptype", create_type=False), nullable=True),
        sa.Column("parent_concept_id", sa.Integer, sa.ForeignKey("concepts.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("cluster_label", sa.Integer, nullable=True),
        sa.Column("metadata_json", sa.JSON, nullable=True),
    )

    # ── claims ────────────────────────────────────────────────────────────
    op.create_table(
        "claims",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("document_id", sa.Integer, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("concept_id", sa.Integer, sa.ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("claim_text", sa.Text, nullable=False),
        sa.Column("claim_type", sa.Enum("supports", "contradicts", "extends", "complements", name="claimtype", create_type=False), nullable=False),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("embedding", Vector(VECTOR_DIM), nullable=True),
    )

    # ── relationships ─────────────────────────────────────────────────────
    op.create_table(
        "relationships",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("source_concept_id", sa.Integer, sa.ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("target_concept_id", sa.Integer, sa.ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("relationship_type", sa.Enum("prerequisite", "builds_on", "contradicts", "complements", "similar", "parent_child", name="relationshiptype", create_type=False), nullable=False),
        sa.Column("strength", sa.Float, nullable=True),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("evidence_json", sa.JSON, nullable=True),
    )

    # ── brain_state ───────────────────────────────────────────────────────
    op.create_table(
        "brain_state",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("project_id", sa.Integer, sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("last_updated", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("summary_text", sa.Text, nullable=True),
        sa.Column("consensus_json", sa.JSON, nullable=True),
    )


def downgrade() -> None:
    op.drop_table("brain_state")
    op.drop_table("relationships")
    op.drop_table("claims")
    op.drop_table("concepts")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("projects")
    op.drop_table("users")

    op.execute("DROP TYPE IF EXISTS gaptype")
    op.execute("DROP TYPE IF EXISTS relationshiptype")
    op.execute("DROP TYPE IF EXISTS claimtype")
