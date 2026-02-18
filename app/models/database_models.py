"""
SQLAlchemy ORM models for Lacuna database.
Includes pgvector support for embeddings.
"""
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    Float,
    Boolean,
    Enum as SQLEnum,
    JSON,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from datetime import datetime
import enum

from app.database import Base
from app.config import settings


# Enums
class ClaimType(str, enum.Enum):
    """Types of claims that can be extracted from documents."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    COMPLEMENTS = "complements"


class RelationshipType(str, enum.Enum):
    """Types of relationships between concepts."""

    PREREQUISITE = "prerequisite"
    BUILDS_ON = "builds_on"
    CONTRADICTS = "contradicts"
    COMPLEMENTS = "complements"
    SIMILAR = "similar"
    PARENT_CHILD = "parent_child"


class GapType(str, enum.Enum):
    """Types of knowledge gaps that can be detected."""

    MISSING_LINK = "missing_link"
    UNDER_EXPLORED = "under_explored"
    CONTRADICTORY = "contradictory"
    ISOLATED_CONCEPT = "isolated_concept"


# Models
class User(Base):
    """User account (synced from NextAuth)."""

    __tablename__ = "users"

    id = Column(String(255), primary_key=True)  # matches NextAuth UUID
    email = Column(String(255), nullable=False, unique=True, index=True)
    name = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    projects = relationship("Project", back_populates="user", cascade="all, delete-orphan")


class Project(Base):
    """Research project containing documents and concepts (= a Room in the frontend)."""

    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Owner (nullable for demo / legacy data)
    user_id = Column(String(255), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    # Room color index (0-4 mapping to ROOM_COLORS in frontend)
    color_index = Column(Integer, default=0)

    # Relationships
    user = relationship("User", back_populates="projects")
    documents = relationship("Document", back_populates="project", cascade="all, delete-orphan")
    concepts = relationship("Concept", back_populates="project", cascade="all, delete-orphan")
    brain_states = relationship("BrainState", back_populates="project", cascade="all, delete-orphan")


class Document(Base):
    """Uploaded document with parsed content."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, docx, etc.
    content_text = Column(Text, nullable=True)  # Full extracted text
    metadata_json = Column(JSON, nullable=True)  # Author, date, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    project = relationship("Project", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    claims = relationship("Claim", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    """Text chunk with embedding for semantic search."""

    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Position in document
    embedding = Column(Vector(settings.VECTOR_DIMENSION), nullable=True)
    metadata_json = Column(JSON, nullable=True)  # Page number, section, etc.

    # Relationships
    document = relationship("Document", back_populates="chunks")


class Concept(Base):
    """Extracted concept with hierarchy and gap information."""

    __tablename__ = "concepts"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Scores
    generality_score = Column(Float, nullable=True)  # How general/abstract this concept is
    coverage_score = Column(Float, nullable=True)  # How well-covered in documents
    consensus_score = Column(Float, nullable=True)  # Agreement across sources

    # Embedding
    embedding = Column(Vector(settings.VECTOR_DIMENSION), nullable=True)

    # Gap detection
    is_gap = Column(Boolean, default=False, nullable=False, index=True)
    gap_type = Column(SQLEnum(GapType), nullable=True)

    # Hierarchy
    parent_concept_id = Column(Integer, ForeignKey("concepts.id", ondelete="SET NULL"), nullable=True, index=True)
    cluster_label = Column(Integer, nullable=True)  # HDBSCAN cluster assignment

    # Metadata
    metadata_json = Column(JSON, nullable=True)

    # Relationships
    project = relationship("Project", back_populates="concepts")
    parent_concept = relationship("Concept", remote_side=[id], backref="child_concepts")
    claims = relationship("Claim", back_populates="concept", cascade="all, delete-orphan")
    source_relationships = relationship(
        "Relationship",
        foreign_keys="Relationship.source_concept_id",
        back_populates="source_concept",
        cascade="all, delete-orphan"
    )
    target_relationships = relationship(
        "Relationship",
        foreign_keys="Relationship.target_concept_id",
        back_populates="target_concept",
        cascade="all, delete-orphan"
    )


class Claim(Base):
    """Claim extracted from document supporting a concept."""

    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    concept_id = Column(Integer, ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False, index=True)
    claim_text = Column(Text, nullable=False)
    claim_type = Column(SQLEnum(ClaimType), nullable=False)
    confidence = Column(Float, nullable=True)  # LLM confidence in extraction
    embedding = Column(Vector(settings.VECTOR_DIMENSION), nullable=True)

    # Relationships
    document = relationship("Document", back_populates="claims")
    concept = relationship("Concept", back_populates="claims")


class Relationship(Base):
    """Relationship between two concepts."""

    __tablename__ = "relationships"

    id = Column(Integer, primary_key=True, index=True)
    source_concept_id = Column(
        Integer, ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_concept_id = Column(
        Integer, ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relationship_type = Column(SQLEnum(RelationshipType), nullable=False)
    strength = Column(Float, nullable=True)  # 0-1 score
    confidence = Column(Float, nullable=True)  # How confident we are
    evidence_json = Column(JSON, nullable=True)  # Supporting evidence

    # Relationships
    source_concept = relationship(
        "Concept",
        foreign_keys=[source_concept_id],
        back_populates="source_relationships"
    )
    target_concept = relationship(
        "Concept",
        foreign_keys=[target_concept_id],
        back_populates="target_relationships"
    )


class BrainState(Base):
    """Central brain state tracking project-wide consensus and insights."""

    __tablename__ = "brain_state"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    summary_text = Column(Text, nullable=True)  # Overall project summary
    consensus_json = Column(JSON, nullable=True)  # Key consensus findings

    # Relationships
    project = relationship("Project", back_populates="brain_states")
