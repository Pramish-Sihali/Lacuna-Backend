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


# Room Schemas (Room = Project with user ownership)
class RoomCreateRequest(BaseModel):
    """Schema for creating a new room (project)."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    color_index: int = Field(0, ge=0, le=4)


class RoomResponse(BaseModel):
    """Schema for room responses."""

    id: int
    name: str
    description: Optional[str] = None
    color_index: int = 0
    paper_count: int = 0
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


# ---------------------------------------------------------------------------
# Clustering Schemas
# ---------------------------------------------------------------------------

class ClusterMemberNode(BaseModel):
    """A concept node within a cluster, supporting a 2-level child tree."""

    id: int
    name: str
    description: Optional[str] = None
    generality_score: Optional[float] = None
    coverage_score: Optional[float] = None
    consensus_score: Optional[float] = None
    is_gap: bool = False
    gap_type: Optional[GapTypeSchema] = None
    is_cluster_head: bool = False
    children: List["ClusterMemberNode"] = []


# Pydantic v2 requires model_rebuild() for self-referential models.
ClusterMemberNode.model_rebuild()


class ClusterGroup(BaseModel):
    """A single cluster with its concept tree (head + children)."""

    cluster_id: int
    label: str                        # name of the cluster-head concept
    size: int
    parent_concept_id: Optional[int] = None
    avg_generality: Optional[float] = None
    cohesion: Optional[float] = None  # average cosine similarity to centroid
    concepts: List[ClusterMemberNode] = []   # root node(s) of the cluster tree


class ConceptMapResponse(BaseModel):
    """Full hierarchical concept map returned by GET /api/concepts/map."""

    clusters: List[ClusterGroup]
    unclustered: List[ConceptNode]      # concepts without a cluster_label
    edges: List[ConceptEdge]            # intra-cluster / unclustered relationships
    bridge_edges: List[ConceptEdge]     # cross-cluster relationships
    total_concepts: int
    total_clustered: int
    total_gaps: int
    num_clusters: int
    metadata: Dict[str, Any] = {}


class ClusterSummary(BaseModel):
    """Brief per-cluster summary returned inside ClusteringResponse."""

    cluster_id: int
    label: str
    size: int
    parent_concept_id: Optional[int] = None
    parent_concept_name: Optional[str] = None
    avg_generality: float


class ClusteringResponse(BaseModel):
    """Response for POST /api/concepts/cluster."""

    project_id: int
    total_concepts: int
    clustered_concepts: int
    noise_reassigned: int
    num_clusters: int
    algorithm_used: str
    clusters: List[ClusterSummary]
    bridge_relationships_found: int
    message: str


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


# ---------------------------------------------------------------------------
# Embedding / Search Schemas
# ---------------------------------------------------------------------------

class EmbedDocumentResponse(BaseModel):
    """Response for POST /api/documents/{document_id}/embed."""

    document_id: int
    embedded_chunks: int
    total_chunks: int
    status: str          # "completed" | "partial" | "no_chunks"
    message: str


class EmbedAllResponse(BaseModel):
    """Response for POST /api/documents/embed-all."""

    documents_processed: int
    total_embedded: int
    total_chunks: int
    message: str


class SimilarChunkResult(BaseModel):
    """A single result from a similarity search."""

    chunk_id: int
    document_id: int
    chunk_index: int
    content: str
    similarity: float
    filename: str
    file_type: str
    metadata_json: Optional[Dict[str, Any]] = None


class SimilarChunksResponse(BaseModel):
    """Response for GET /api/search/similar."""

    query: str
    results: List[SimilarChunkResult]
    total_results: int


# ---------------------------------------------------------------------------
# Extraction Schemas
# ---------------------------------------------------------------------------

class ExtractionResponse(BaseModel):
    """Response for POST /api/documents/{document_id}/extract."""

    document_id: int
    document_title: str
    chunks_processed: int
    chunks_skipped: int
    concepts_extracted: int   # raw count before deduplication
    concepts_saved: int       # net new concepts added to the project
    claims_saved: int
    relationships_found: int
    errors: List[str]
    message: str


class ProcessDocumentResponse(BaseModel):
    """
    Response for POST /api/documents/{document_id}/process.

    Combines results from the embedding phase and the extraction phase.
    """

    document_id: int
    document_title: str
    # --- embedding phase ---
    embedded_chunks: int
    total_chunks: int
    # --- extraction phase ---
    chunks_processed: int
    chunks_skipped: int
    concepts_extracted: int
    concepts_saved: int
    claims_saved: int
    relationships_found: int
    errors: List[str]
    message: str


# ---------------------------------------------------------------------------
# Normalisation Schemas
# ---------------------------------------------------------------------------

class NormalizationResponse(BaseModel):
    """Response for POST /api/concepts/normalize."""

    project_id: int
    total_concepts_before: int
    canonical_concepts_after: int
    merged_count: int
    # One entry per group where ≥ 2 concepts were merged
    groups_merged: List[Dict[str, Any]]
    # variant_name -> canonical_name for every alias produced
    alias_map: Dict[str, str]
    message: str


class GapItem(BaseModel):
    """
    A single detected gap, returned by GET /api/concepts/gaps.

    ``is_synthetic`` is True for new concept nodes created by gap detection
    (expected_topic, bridging) and False for existing concepts that were
    flagged in-place (weak_coverage).
    """

    id: int
    name: str
    description: Optional[str] = None
    gap_type: GapTypeSchema
    gap_subtype: str                        # expected_topic | bridging | weak_coverage
    importance: Optional[str] = None        # critical | important | nice_to_have
    suggestions: List[str] = []
    related_to: List[str] = []
    is_synthetic: bool = False
    coverage_score: Optional[float] = None
    generality_score: Optional[float] = None
    cluster_label: Optional[int] = None


class GapDetectionResponse(BaseModel):
    """Response for POST /api/concepts/detect-gaps."""

    project_id: int
    expected_gaps_count: int
    bridging_gaps_count: int
    weak_coverage_count: int
    total_gaps: int
    suggestions: List[str] = []
    message: str


# ---------------------------------------------------------------------------
# React Flow Map Schemas  (GET /api/concepts/map)
# ---------------------------------------------------------------------------

class ReactFlowPosition(BaseModel):
    """2-D position placeholder; layout is computed client-side."""

    x: float = 0.0
    y: float = 0.0


class ReactFlowNodeData(BaseModel):
    """Payload attached to every React Flow node."""

    label: str
    description: Optional[str] = None
    coverage_score: Optional[float] = None
    consensus_score: Optional[float] = None
    generality_score: Optional[float] = None
    document_count: int = 0
    is_gap: bool = False
    gap_type: Optional[str] = None
    cluster_id: Optional[str] = None   # e.g. "cluster_0"
    parent_id: Optional[str] = None    # e.g. "concept_5"
    children: List[str] = []           # e.g. ["concept_5", "concept_8"]


class ReactFlowNode(BaseModel):
    """A single node in the React Flow graph."""

    id: str                            # "concept_{concept_id}"
    type: str                          # "concept" | "gap"
    data: ReactFlowNodeData
    position: ReactFlowPosition = Field(default_factory=ReactFlowPosition)


class ReactFlowEdgeData(BaseModel):
    """Payload attached to every React Flow edge."""

    strength: Optional[float] = None
    confidence: Optional[float] = None
    label: str = ""


class ReactFlowEdge(BaseModel):
    """A single edge in the React Flow graph."""

    id: str                            # "rel_{relationship_id}"
    source: str                        # "concept_{source_id}"
    target: str                        # "concept_{target_id}"
    type: str                          # relationship type value
    data: ReactFlowEdgeData


class ReactFlowMapMetadata(BaseModel):
    """Top-level metadata for the concept map."""

    total_concepts: int
    total_relationships: int
    total_gaps: int
    num_clusters: int
    brain_last_updated: Optional[datetime] = None
    has_clustering: bool = False


class ReactFlowConceptMapResponse(BaseModel):
    """
    Full concept map in React Flow format.

    Returned by GET /api/concepts/map — the primary endpoint consumed
    by the Next.js frontend.
    """

    nodes: List[ReactFlowNode]
    edges: List[ReactFlowEdge]
    gaps: List[GapItem]
    metadata: ReactFlowMapMetadata


class RelationshipDetectionResponse(BaseModel):
    """Response for POST /api/concepts/relationships."""

    project_id: int
    total_pairs_considered: int
    relationships_found: int
    relationships_saved: int
    by_type_count: Dict[str, int]
    message: str


# ---------------------------------------------------------------------------
# Brain Schemas
# ---------------------------------------------------------------------------

class BrainBuildResponse(BaseModel):
    """Response for POST /api/brain/build and POST /api/brain/rebuild."""

    project_id: int
    concepts_scored: int
    strong_consensus_count: int
    contested_count: int
    contradiction_count: int
    summary_text: str
    message: str


class BrainChatRequest(BaseModel):
    """Request body for POST /api/brain/chat."""

    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)


class BrainChatResponse(BaseModel):
    """Response for POST /api/brain/chat."""

    question: str
    answer: str
    sources: List[str]
    relevant_concepts: List[str]
    confidence: float


class BrainStatusResponse(BaseModel):
    """Response for GET /api/brain/status."""

    project_id: int
    last_updated: Optional[datetime] = None
    doc_count: int
    concept_count: int
    gap_count: int
    relationship_count: int
    avg_consensus: Optional[float] = None
    health_score: float
    summary_text: Optional[str] = None
    has_brain: bool


class ConceptDetailResponse(BaseModel):
    """
    Rich response for GET /api/concepts/{concept_id}.

    Extends ConceptResponse with aliases, source documents, and claims.
    """

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
    # Enriched fields
    aliases: List[str] = []
    source_documents: List[Dict[str, Any]] = []   # [{id, filename, file_type}]
    claims: List[ClaimResponse] = []
    claim_count: int = 0

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Pipeline / Orchestration Schemas
# ---------------------------------------------------------------------------

class ProcessingResultResponse(BaseModel):
    """Response for processing a single document through the full pipeline."""

    document_id: int
    document_title: str
    embedded_chunks: int
    total_chunks: int
    chunks_processed: int
    chunks_skipped: int
    concepts_extracted: int
    concepts_saved: int
    claims_saved: int
    relationships_found: int
    errors: List[str]
    processing_time_seconds: float
    message: str


class KnowledgeBuildResponse(BaseModel):
    """
    Response for POST /api/concepts/build and POST /api/pipeline/full-rebuild.

    Summarises all stages of the knowledge graph rebuild pipeline:
    normalise → cluster → relationships → gaps → consensus.
    """

    project_id: int
    # Normalisation
    concepts_before_normalize: int
    concepts_after_normalize: int
    merged_count: int
    # Clustering
    num_clusters: int
    algorithm_used: str
    clustered_concepts: int
    # Relationships
    relationships_found: int
    relationships_saved: int
    # Gaps
    expected_gaps: int
    bridging_gaps: int
    weak_coverage_gaps: int
    total_gaps: int
    # Brain / consensus
    concepts_scored: int
    strong_consensus_count: int
    contested_count: int
    contradiction_count: int
    brain_summary: str
    processing_time_seconds: float
    message: str


class PipelineProcessAllResponse(BaseModel):
    """Response for POST /api/pipeline/process-all."""

    project_id: int
    documents_processed: int
    documents_failed: int
    total_concepts: int
    total_relationships: int
    total_gaps: int
    processing_time_seconds: float
    errors: List[str] = []
    message: str
