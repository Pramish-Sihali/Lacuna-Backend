"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# Enums (matching database enums)
class ClaimTypeSchema(str, Enum):
    """Claim types for API responses."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    COMPLEMENTS = "complements"


class RelationshipTypeSchema(str, Enum):
    """Relationship types for API responses."""

    PREREQUISITE = "prerequisite"
    BUILDS_ON = "builds_on"
    CONTRADICTS = "contradicts"
    COMPLEMENTS = "complements"
    SIMILAR = "similar"
    PARENT_CHILD = "parent_child"


class GapTypeSchema(str, Enum):
    """Gap types for API responses."""

    MISSING_LINK = "missing_link"
    UNDER_EXPLORED = "under_explored"
    CONTRADICTORY = "contradictory"
    ISOLATED_CONCEPT = "isolated_concept"


# Project Schemas
class ProjectCreate(BaseModel):
    """Schema for creating a new project."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class ProjectResponse(BaseModel):
    """Schema for project responses."""

    id: int
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Document Schemas
class DocumentUploadResponse(BaseModel):
    """Schema for document upload response."""

    id: int
    filename: str
    file_type: str
    status: str = "uploaded"
    message: str = "Document uploaded successfully"


class DocumentResponse(BaseModel):
    """Schema for document details."""

    id: int
    project_id: int
    filename: str
    file_type: str
    content_text: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: datetime
    chunk_count: Optional[int] = 0

    model_config = ConfigDict(from_attributes=True)


# Concept Schemas
class ConceptResponse(BaseModel):
    """Schema for concept details."""

    id: int
    project_id: int
    name: str
    description: Optional[str] = None
    generality_score: Optional[float] = None
    coverage_score: Optional[float] = None
    consensus_score: Optional[float] = None
    is_gap: bool = False
    gap_type: Optional[GapTypeSchema] = None
    parent_concept_id: Optional[int] = None
    cluster_label: Optional[int] = None
    metadata_json: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class ConceptNode(BaseModel):
    """Schema for concept map node."""

    id: int
    name: str
    description: Optional[str] = None
    generality_score: Optional[float] = None
    coverage_score: Optional[float] = None
    consensus_score: Optional[float] = None
    is_gap: bool = False
    gap_type: Optional[GapTypeSchema] = None
    cluster_label: Optional[int] = None
    children_count: int = 0


class ConceptEdge(BaseModel):
    """Schema for concept map edge."""

    source_id: int
    target_id: int
    relationship_type: RelationshipTypeSchema
    strength: Optional[float] = None
    confidence: Optional[float] = None


class ConceptMapResponse(BaseModel):
    """Schema for complete concept map."""

    nodes: List[ConceptNode]
    edges: List[ConceptEdge]
    total_concepts: int
    total_gaps: int
    clusters: Dict[int, int]  # cluster_label -> count


# Claim Schemas
class ClaimResponse(BaseModel):
    """Schema for claim details."""

    id: int
    document_id: int
    concept_id: int
    claim_text: str
    claim_type: ClaimTypeSchema
    confidence: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


# Relationship Schemas
class RelationshipResponse(BaseModel):
    """Schema for relationship details."""

    id: int
    source_concept_id: int
    target_concept_id: int
    relationship_type: RelationshipTypeSchema
    strength: Optional[float] = None
    confidence: Optional[float] = None
    evidence_json: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


# Brain State Schemas
class BrainStateResponse(BaseModel):
    """Schema for brain state."""

    id: int
    project_id: int
    last_updated: datetime
    summary_text: Optional[str] = None
    consensus_json: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class BrainQueryRequest(BaseModel):
    """Schema for querying the brain."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    include_gaps: bool = True


class BrainQueryResponse(BaseModel):
    """Schema for brain query results."""

    query: str
    answer: str
    relevant_concepts: List[ConceptResponse]
    relevant_documents: List[str]
    confidence: float


# Health Check Schema
class HealthCheckResponse(BaseModel):
    """Schema for health check endpoint."""

    status: str
    database: str
    ollama: str
    timestamp: datetime
    version: str = "0.1.0"


# Processing Status Schemas
class ProcessingStatus(BaseModel):
    """Schema for document processing status."""

    document_id: int
    status: str  # pending, processing, completed, failed
    progress: float  # 0-100
    current_step: str
    error: Optional[str] = None


# Statistics Schemas
class ProjectStatistics(BaseModel):
    """Schema for project statistics."""

    project_id: int
    total_documents: int
    total_concepts: int
    total_gaps: int
    total_relationships: int
    coverage_distribution: Dict[str, int]  # low, medium, high
    gap_type_distribution: Dict[str, int]
