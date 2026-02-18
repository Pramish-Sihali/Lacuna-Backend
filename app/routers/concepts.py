"""
Concept map, normalisation, and analysis endpoints.

Route summary
-------------
POST /extract          — legacy: extract from all docs (background)
POST /normalize        — normalise + deduplicate all project concepts
POST /build            — full knowledge rebuild (normalize→cluster→rels→gaps→consensus)
GET  /                 — list canonical concepts (with coverage scores)
GET  /map              — React Flow concept map (nodes + edges + gaps + metadata)
POST /relationships    — detect + score all concept relationships (synchronous)
GET  /relationships    — list all stored relationships
POST /detect-gaps      — run all 3 gap detection passes (synchronous)
GET  /gaps             — list all detected gaps with suggestions
GET  /{concept_id}     — concept detail (aliases, source docs, claims)
POST /cluster          — HDBSCAN clustering (synchronous, returns ClusteringResponse)
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.database_models import (
    BrainState,
    Claim,
    Concept,
    Document,
    Relationship,
)
from app.models.schemas import (
    ClaimResponse,
    ClusterGroup,
    ClusteringResponse,
    ClusterMemberNode,
    ClusterSummary,
    ConceptDetailResponse,
    ConceptEdge,
    ConceptMapResponse,
    ConceptNode,
    ConceptResponse,
    GapDetectionResponse,
    GapItem,
    GapTypeSchema,
    KnowledgeBuildResponse,
    NormalizationResponse,
    ReactFlowConceptMapResponse,
    ReactFlowEdge,
    ReactFlowEdgeData,
    ReactFlowMapMetadata,
    ReactFlowNode,
    ReactFlowNodeData,
    ReactFlowPosition,
    RelationshipDetectionResponse,
    RelationshipResponse,
)
from app.services.clustering import ConceptClusterer
from app.services.embedding import EmbeddingService
from app.services.gap_detector import GapDetector, GapDetectionResult
from app.services.llm_extractor import LLMExtractor
from app.services.normalizer import ConceptNormalizer
from app.services.pipeline import LacunaPipeline
from app.services.relationships import RelationshipDetector, RelationshipService

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /extract  (legacy background extraction)
# ---------------------------------------------------------------------------

@router.post("/extract", status_code=status.HTTP_202_ACCEPTED)
async def extract_concepts(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Extract concepts from all documents in the background.

    Prefer the per-document ``POST /api/documents/{id}/process`` endpoint
    for new work; this route is kept for backward compatibility.
    """
    background_tasks.add_task(process_concept_extraction, db)
    return {"message": "Concept extraction started in background", "status": "processing"}


async def process_concept_extraction(db: AsyncSession) -> None:
    """Background task: extract + embed + normalise concepts for all documents."""
    try:
        logger.info("Starting concept extraction...")

        result = await db.execute(
            select(Document).where(Document.project_id == settings.DEFAULT_PROJECT_ID)
        )
        documents = result.scalars().all()

        if not documents:
            logger.warning("No documents found for concept extraction")
            return

        llm_extractor = LLMExtractor()
        embedding_service = EmbeddingService()
        normalizer = ConceptNormalizer()

        all_extracted: List[Dict] = []

        for document in documents:
            logger.info("Extracting concepts from %s", document.filename)
            concepts = await llm_extractor.extract_concepts(document.content_text)

            for concept_data in concepts:
                concept_embedding = await embedding_service.generate_embedding(
                    f"{concept_data['name']} {concept_data.get('description', '')}"
                )
                all_extracted.append(
                    {**concept_data, "embedding": concept_embedding, "document_id": document.id}
                )

        logger.info("Normalising %d extracted concepts...", len(all_extracted))
        normalized = await normalizer.normalize_concepts(all_extracted)

        for concept_data in normalized:
            db.add(
                Concept(
                    project_id=settings.DEFAULT_PROJECT_ID,
                    name=concept_data["name"],
                    description=concept_data.get("description"),
                    generality_score=concept_data.get("generality_score", 0.5),
                    embedding=concept_data.get("embedding"),
                    coverage_score=0.5,
                    consensus_score=1.0,
                )
            )

        await db.commit()
        logger.info("Concept extraction completed. Saved %d concepts.", len(normalized))

    except Exception as exc:
        logger.error("Error in concept extraction: %s", exc, exc_info=True)
        await db.rollback()


# ---------------------------------------------------------------------------
# POST /normalize
# ---------------------------------------------------------------------------

@router.post("/normalize", response_model=NormalizationResponse)
async def normalize_concepts(
    db: AsyncSession = Depends(get_db),
) -> NormalizationResponse:
    """
    Normalise all concepts for the default project.

    **Pipeline**

    1. Group concepts by exact normalised name (guaranteed duplicates).
    2. Compute a representative embedding per name-group (average).
    3. Run Agglomerative Clustering (cosine distance, average linkage,
       distance threshold = 1 − similarity_threshold) to find semantically
       equivalent groups across different surface forms.
    4. For each cluster:
       - Pick the canonical name (most frequent → shortest → alphabetical).
       - Merge descriptions.
       - Re-embed (average of all variant embeddings, L2-normalised).
       - Recompute coverage_score.
       - Remap all claims / relationships / parent refs to the canonical ID.
       - Delete duplicate rows.
    5. Update coverage scores for singleton concepts.

    Safe to call multiple times.  Returns a full merge summary.
    """
    normalizer = ConceptNormalizer()
    try:
        result = await normalizer.normalize_project(
            settings.DEFAULT_PROJECT_ID, db
        )
    except Exception as exc:
        logger.exception("normalize_concepts: unexpected error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Normalisation failed: {exc}",
        )

    message = (
        f"Normalised {result.total_concepts_before} concepts into "
        f"{result.canonical_concepts_after} canonical concepts "
        f"({result.merged_count} merged)."
    )
    logger.info(message)

    return NormalizationResponse(
        project_id=result.project_id,
        total_concepts_before=result.total_concepts_before,
        canonical_concepts_after=result.canonical_concepts_after,
        merged_count=result.merged_count,
        groups_merged=result.groups_merged,
        alias_map=result.alias_map,
        message=message,
    )


# ---------------------------------------------------------------------------
# POST /build  — full knowledge rebuild pipeline (synchronous)
# ---------------------------------------------------------------------------

@router.post("/build", response_model=KnowledgeBuildResponse)
async def build_knowledge(
    db: AsyncSession = Depends(get_db),
) -> KnowledgeBuildResponse:
    """
    Run the full knowledge-graph rebuild pipeline for the project.

    **Pipeline** (all stages run synchronously in order):

    1. **Normalise** — deduplicate concepts via exact-name grouping and
       ``AgglomerativeClustering`` (cosine distance); remaps claims and
       relationships to canonical IDs.
    2. **Cluster** — ``HDBSCAN`` (KMeans fallback); builds a 2-level concept
       hierarchy and detects cross-cluster bridge relationships.
    3. **Relationships** — embedding similarity + LLM classification + multi-
       signal strength scoring.
    4. **Gaps** — three passes: expected topics (LLM), bridging gaps (graph),
       weak-coverage concepts (statistical).
    5. **Consensus** — per-concept support/contradict scoring; generates an
       LLM synthesis summary; persists ``BrainState``.

    Safe to call multiple times.  Already-merged/clustered artefacts are
    handled by each service's own idempotency logic.

    Returns a comprehensive summary of every pipeline stage.
    """
    pipeline = LacunaPipeline()
    t0 = time.monotonic()
    try:
        result = await pipeline.rebuild_project_knowledge(
            settings.DEFAULT_PROJECT_ID, db
        )
    except Exception as exc:
        logger.exception("build_knowledge: unexpected error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge build failed: {exc}",
        )

    elapsed = round(time.monotonic() - t0, 2)
    logger.info("build_knowledge: completed in %.2fs", elapsed)

    return KnowledgeBuildResponse(
        project_id=result.project_id,
        concepts_before_normalize=result.concepts_before_normalize,
        concepts_after_normalize=result.concepts_after_normalize,
        merged_count=result.merged_count,
        num_clusters=result.num_clusters,
        algorithm_used=result.algorithm_used,
        clustered_concepts=result.clustered_concepts,
        relationships_found=result.relationships_found,
        relationships_saved=result.relationships_saved,
        expected_gaps=result.expected_gaps,
        bridging_gaps=result.bridging_gaps,
        weak_coverage_gaps=result.weak_coverage_gaps,
        total_gaps=result.total_gaps,
        concepts_scored=result.concepts_scored,
        strong_consensus_count=result.strong_consensus_count,
        contested_count=result.contested_count,
        contradiction_count=result.contradiction_count,
        brain_summary=result.brain_summary,
        processing_time_seconds=elapsed,
        message=result.message,
    )


# ---------------------------------------------------------------------------
# GET /  — list canonical concepts
# ---------------------------------------------------------------------------

@router.get("/", response_model=List[ConceptResponse])
async def list_concepts(
    skip: int = 0,
    limit: int = 100,
    gaps_only: bool = False,
    db: AsyncSession = Depends(get_db),
) -> List[ConceptResponse]:
    """
    List all normalised canonical concepts for the project.

    Results are ordered by ``coverage_score`` descending so the most
    well-evidenced concepts appear first.

    Query parameters
    ----------------
    gaps_only : bool
        When ``true``, return only concepts flagged as knowledge gaps.
    """
    query = (
        select(Concept)
        .where(Concept.project_id == settings.DEFAULT_PROJECT_ID)
    )
    if gaps_only:
        query = query.where(Concept.is_gap == True)  # noqa: E712

    query = (
        query
        .order_by(Concept.coverage_score.desc().nullslast())
        .offset(skip)
        .limit(limit)
    )
    res = await db.execute(query)
    concepts = res.scalars().all()
    return [ConceptResponse.model_validate(c) for c in concepts]


# ---------------------------------------------------------------------------
# GET /map  — React Flow concept map (THE MAIN FRONTEND ENDPOINT)
# ---------------------------------------------------------------------------

@router.get("/map", response_model=ReactFlowConceptMapResponse)
async def get_concept_map(
    db: AsyncSession = Depends(get_db),
) -> ReactFlowConceptMapResponse:
    """
    Return the full concept map in **React Flow** format.

    This is the primary endpoint consumed by the Next.js frontend.

    **Nodes**

    Every concept (including gap nodes) appears as a node:

    - ``id``   — ``"concept_{concept_id}"``
    - ``type`` — ``"concept"`` or ``"gap"``
    - ``data`` — label, scores, document count, cluster/parent/children refs
    - ``position`` — ``{x: 0, y: 0}`` (the frontend computes layout)

    **Edges**

    All stored ``Relationship`` rows are returned as edges:

    - ``id``     — ``"rel_{relationship_id}"``
    - ``source`` / ``target`` — ``"concept_{id}"`` references
    - ``type``   — relationship type string (e.g. ``"builds_on"``)

    **Gaps**

    A parallel list of ``GapItem`` objects (one per gap concept) with
    suggestions and importance rankings — useful for the gap-list panel.

    **Metadata**

    Aggregate counts + brain last-updated timestamp.
    """
    # ---- Load concepts --------------------------------------------------
    concept_res = await db.execute(
        select(Concept).where(Concept.project_id == settings.DEFAULT_PROJECT_ID)
    )
    concepts: List[Concept] = list(concept_res.scalars().all())

    # ---- Load relationships ----------------------------------------------
    # Filtered to source concepts that belong to this project
    concept_ids_in_project = {c.id for c in concepts}
    rel_res = await db.execute(
        select(Relationship).where(
            Relationship.source_concept_id.in_(concept_ids_in_project)
        )
    )
    relationships = list(rel_res.scalars().all())

    # ---- Document count per concept (via claims) ------------------------
    doc_counts: Dict[int, int] = {}
    if concept_ids_in_project:
        dc_res = await db.execute(
            select(
                Claim.concept_id,
                func.count(distinct(Claim.document_id)).label("doc_count"),
            )
            .where(Claim.concept_id.in_(concept_ids_in_project))
            .group_by(Claim.concept_id)
        )
        doc_counts = {row.concept_id: row.doc_count for row in dc_res}

    # ---- Brain state (for last-updated timestamp) -----------------------
    brain_res = await db.execute(
        select(BrainState)
        .where(BrainState.project_id == settings.DEFAULT_PROJECT_ID)
        .order_by(BrainState.last_updated.desc())
        .limit(1)
    )
    brain_state = brain_res.scalar_one_or_none()

    # ---- Children map: parent_id → [child_id, …] -----------------------
    children_map: Dict[int, List[int]] = {}
    for c in concepts:
        if c.parent_concept_id is not None:
            children_map.setdefault(c.parent_concept_id, []).append(c.id)

    # ---- Build nodes + gap items ----------------------------------------
    nodes: List[ReactFlowNode] = []
    gap_items: List[GapItem] = []
    _importance_rank = {"critical": 0, "important": 1, "nice_to_have": 2}

    for c in concepts:
        node_type = "gap" if c.is_gap else "concept"
        cluster_id = (
            f"cluster_{c.cluster_label}" if c.cluster_label is not None else None
        )
        parent_id = (
            f"concept_{c.parent_concept_id}" if c.parent_concept_id else None
        )
        children = [
            f"concept_{cid}" for cid in children_map.get(c.id, [])
        ]

        nodes.append(
            ReactFlowNode(
                id=f"concept_{c.id}",
                type=node_type,
                data=ReactFlowNodeData(
                    label=c.name,
                    description=c.description,
                    coverage_score=c.coverage_score,
                    consensus_score=c.consensus_score,
                    generality_score=c.generality_score,
                    document_count=doc_counts.get(c.id, 0),
                    is_gap=c.is_gap,
                    gap_type=c.gap_type.value if c.gap_type else None,
                    cluster_id=cluster_id,
                    parent_id=parent_id,
                    children=children,
                ),
                position=ReactFlowPosition(x=0.0, y=0.0),
            )
        )

        # Parallel gap list for the gap panel
        if c.is_gap:
            meta = c.metadata_json or {}
            gap_subtype = meta.get("gap_subtype", "weak_coverage")
            gap_type_val = c.gap_type.value if c.gap_type else "missing_link"
            gap_items.append(
                GapItem(
                    id=c.id,
                    name=c.name,
                    description=c.description,
                    gap_type=GapTypeSchema(gap_type_val),
                    gap_subtype=gap_subtype,
                    importance=meta.get("importance"),
                    suggestions=meta.get("suggestions", []),
                    related_to=meta.get("related_to", []),
                    is_synthetic=bool(meta.get("is_synthetic_gap", False)),
                    coverage_score=c.coverage_score,
                    generality_score=c.generality_score,
                    cluster_label=c.cluster_label,
                )
            )

    # Sort gaps: critical first, then by subtype
    gap_items.sort(
        key=lambda g: (
            _importance_rank.get(g.importance or "important", 1),
            g.gap_subtype,
        )
    )

    # ---- Build edges ----------------------------------------------------
    edges: List[ReactFlowEdge] = []
    for r in relationships:
        # Only include edges where both endpoints are in the response
        if (
            r.source_concept_id not in concept_ids_in_project
            or r.target_concept_id not in concept_ids_in_project
        ):
            continue
        rel_type = r.relationship_type.value if r.relationship_type else "similar"
        edges.append(
            ReactFlowEdge(
                id=f"rel_{r.id}",
                source=f"concept_{r.source_concept_id}",
                target=f"concept_{r.target_concept_id}",
                type=rel_type,
                data=ReactFlowEdgeData(
                    strength=r.strength,
                    confidence=r.confidence,
                    label=rel_type,
                ),
            )
        )

    # ---- Metadata -------------------------------------------------------
    cluster_ids = {c.cluster_label for c in concepts if c.cluster_label is not None}
    metadata = ReactFlowMapMetadata(
        total_concepts=len(concepts),
        total_relationships=len(edges),
        total_gaps=len(gap_items),
        num_clusters=len(cluster_ids),
        brain_last_updated=brain_state.last_updated if brain_state else None,
        has_clustering=bool(cluster_ids),
    )

    return ReactFlowConceptMapResponse(
        nodes=nodes,
        edges=edges,
        gaps=gap_items,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# POST /relationships  — detect + score relationships (synchronous)
# ---------------------------------------------------------------------------

@router.post("/relationships", response_model=RelationshipDetectionResponse)
async def detect_relationships(
    db: AsyncSession = Depends(get_db),
) -> RelationshipDetectionResponse:
    """
    Detect and score semantic relationships between all concepts in the project.

    **Pipeline**

    1. Load all concepts that have embeddings.
    2. Build an O(n²) cosine-similarity matrix; keep candidate pairs where
       0.30 ≤ similarity ≤ 0.85 (below → too unrelated; above → should have
       been merged in normalisation).
    3. For large projects, cap at 500 LLM calls by keeping highest-similarity
       pairs first.
    4. For each candidate pair, gather evidence:
       - Claims associated with each concept.
       - Text chunks where **both** concept names appear (co-occurrence).
    5. Send evidence to the LLM to classify relationship type and direction.
    6. Compute multi-signal strength score (embedding sim + co-occurrence +
       shared documents + LLM confidence).
    7. Deduplicate: merge bidirectional "extends+extends" → "complements";
       resolve conflicting types by keeping higher-confidence detection.
    8. Persist new ``Relationship`` rows and commit.

    Safe to call multiple times — already-stored pairs are skipped.

    Relationship type mapping (LLM → database enum):
    - ``supports``    → ``similar``
    - ``extends``     → ``builds_on``
    - ``contradicts`` → ``contradicts``
    - ``complements`` → ``complements``
    """
    detector = RelationshipDetector()
    try:
        result = await detector.detect_relationships(
            settings.DEFAULT_PROJECT_ID, db
        )
    except Exception as exc:
        logger.exception("detect_relationships: unexpected error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Relationship detection failed: {exc}",
        )

    message = (
        f"Analysed {result.total_pairs_considered} candidate pair(s). "
        f"Found {result.relationships_found} relationship(s) "
        f"({result.relationships_saved} new). "
        f"Types: {result.by_type_count}."
    )
    logger.info(message)

    return RelationshipDetectionResponse(
        project_id=result.project_id,
        total_pairs_considered=result.total_pairs_considered,
        relationships_found=result.relationships_found,
        relationships_saved=result.relationships_saved,
        by_type_count=result.by_type_count,
        message=message,
    )


# ---------------------------------------------------------------------------
# GET /relationships  — list all relationships for the project
# ---------------------------------------------------------------------------

@router.get("/relationships", response_model=List[RelationshipResponse])
async def list_relationships(
    skip: int = 0,
    limit: int = 200,
    db: AsyncSession = Depends(get_db),
) -> List[RelationshipResponse]:
    """
    Return all stored relationships for the default project, ordered by
    ``strength`` descending.

    Query parameters
    ----------------
    skip  : int  — pagination offset (default 0)
    limit : int  — max rows returned (default 200, max 1 000)
    """
    limit = min(limit, 1000)
    result = await db.execute(
        select(Relationship)
        .join(
            Concept,
            Relationship.source_concept_id == Concept.id,
        )
        .where(Concept.project_id == settings.DEFAULT_PROJECT_ID)
        .order_by(Relationship.strength.desc().nullslast())
        .offset(skip)
        .limit(limit)
    )
    relationships = result.scalars().all()
    return [RelationshipResponse.model_validate(r) for r in relationships]


# ---------------------------------------------------------------------------
# POST /detect-gaps  — synchronous, returns GapDetectionResponse
# ---------------------------------------------------------------------------

@router.post("/detect-gaps", response_model=GapDetectionResponse)
async def detect_gaps(
    db: AsyncSession = Depends(get_db),
) -> GapDetectionResponse:
    """
    Detect knowledge gaps in the project concept graph.

    Runs three passes and returns results synchronously:

    **Pass 1 — Expected Topics** *(LLM)*
    The LLM inspects all existing concepts and identifies domain topics that
    are conspicuously absent.  New synthetic ``Concept`` rows are created for
    each suggestion so they appear as positioned gap nodes on the concept map.

    **Pass 2 — Bridging Gaps** *(graph topology + LLM)*
    Clusters that are semantically related (centroid cosine similarity ≥ 0.40)
    but share no ``Relationship`` rows represent areas where the literature has
    not connected two related sub-fields.  The LLM suggests a bridging concept
    whose embedding is placed at the midpoint of the two cluster centroids.

    **Pass 3 — Weak Coverage** *(statistical)*
    Existing broad concepts (``generality_score > 0.50``) with very low
    coverage (``coverage_score < 0.20``) are flagged in-place as
    under-explored — no new nodes are created.

    **Re-runnable** — previous synthetic gap nodes are deleted and all gap
    flags are reset before each run, so this endpoint is always safe to call
    again after new documents are ingested or the collection changes.
    """
    detector = GapDetector()
    try:
        result: GapDetectionResult = await detector.detect_gaps(
            settings.DEFAULT_PROJECT_ID, db
        )
    except Exception as exc:
        logger.exception("detect_gaps: unexpected error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gap detection failed: {exc}",
        )

    message = (
        f"Detected {result.total_gaps} gap(s): "
        f"{result.expected_gaps_count} expected topic(s), "
        f"{result.bridging_gaps_count} bridging gap(s), "
        f"{result.weak_coverage_count} weak-coverage concept(s)."
    )
    logger.info(message)

    return GapDetectionResponse(
        project_id=result.project_id,
        expected_gaps_count=result.expected_gaps_count,
        bridging_gaps_count=result.bridging_gaps_count,
        weak_coverage_count=result.weak_coverage_count,
        total_gaps=result.total_gaps,
        suggestions=result.suggestions,
        message=message,
    )


# ---------------------------------------------------------------------------
# GET /gaps  — list all detected gaps with suggestions
# ---------------------------------------------------------------------------

@router.get("/gaps", response_model=List[GapItem])
async def list_gaps(
    subtype: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
) -> List[GapItem]:
    """
    Return all detected gap concepts and flagged weak-coverage concepts,
    ordered by importance (critical first) then by gap subtype.

    Query parameters
    ----------------
    subtype : str  — optional filter: ``expected_topic``, ``bridging``,
                     or ``weak_coverage``
    """
    query = (
        select(Concept)
        .where(
            Concept.project_id == settings.DEFAULT_PROJECT_ID,
            Concept.is_gap == True,  # noqa: E712
        )
        .order_by(Concept.coverage_score.asc().nullsfirst())
    )
    res = await db.execute(query)
    gap_concepts = res.scalars().all()

    _importance_rank = {"critical": 0, "important": 1, "nice_to_have": 2}

    items: List[GapItem] = []
    for c in gap_concepts:
        meta = c.metadata_json or {}
        gap_subtype = meta.get("gap_subtype", "weak_coverage")

        if subtype and gap_subtype != subtype:
            continue

        gap_type_val = c.gap_type.value if c.gap_type else "missing_link"

        items.append(
            GapItem(
                id=c.id,
                name=c.name,
                description=c.description,
                gap_type=GapTypeSchema(gap_type_val),
                gap_subtype=gap_subtype,
                importance=meta.get("importance"),
                suggestions=meta.get("suggestions", []),
                related_to=meta.get("related_to", []),
                is_synthetic=bool(meta.get("is_synthetic_gap", False)),
                coverage_score=c.coverage_score,
                generality_score=c.generality_score,
                cluster_label=c.cluster_label,
            )
        )

    items.sort(
        key=lambda g: (
            _importance_rank.get(g.importance or "important", 1),
            g.gap_subtype,
        )
    )
    return items


# ---------------------------------------------------------------------------
# GET /{concept_id}  — concept detail with aliases, docs, claims
# ---------------------------------------------------------------------------

@router.get("/{concept_id}", response_model=ConceptDetailResponse)
async def get_concept(
    concept_id: int,
    db: AsyncSession = Depends(get_db),
) -> ConceptDetailResponse:
    """
    Return full details for a single concept, including:

    - **aliases** — variant names merged into this canonical concept.
    - **source_documents** — documents that mention this concept via a claim.
    - **claims** — all extracted claims linking documents to this concept.
    """
    concept_res = await db.execute(
        select(Concept).where(
            Concept.id == concept_id,
            Concept.project_id == settings.DEFAULT_PROJECT_ID,
        )
    )
    concept = concept_res.scalar_one_or_none()
    if concept is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Concept {concept_id} not found.",
        )

    # Claims for this concept
    claims_res = await db.execute(
        select(Claim).where(Claim.concept_id == concept_id)
    )
    claims = claims_res.scalars().all()

    # Source documents (distinct documents that have claims for this concept)
    doc_ids = list({c.document_id for c in claims})
    source_documents: List[Dict] = []
    if doc_ids:
        docs_res = await db.execute(
            select(Document).where(Document.id.in_(doc_ids))
        )
        source_documents = [
            {"id": d.id, "filename": d.filename, "file_type": d.file_type}
            for d in docs_res.scalars().all()
        ]

    # Aliases stored in metadata_json by the normaliser
    aliases: List[str] = (concept.metadata_json or {}).get("aliases", [])

    return ConceptDetailResponse(
        id=concept.id,
        project_id=concept.project_id,
        name=concept.name,
        description=concept.description,
        generality_score=concept.generality_score,
        coverage_score=concept.coverage_score,
        consensus_score=concept.consensus_score,
        is_gap=concept.is_gap,
        gap_type=concept.gap_type,
        parent_concept_id=concept.parent_concept_id,
        cluster_label=concept.cluster_label,
        metadata_json=concept.metadata_json,
        aliases=aliases,
        source_documents=source_documents,
        claims=[ClaimResponse.model_validate(c) for c in claims],
        claim_count=len(claims),
    )


# ---------------------------------------------------------------------------
# POST /cluster  — synchronous, returns ClusteringResponse
# ---------------------------------------------------------------------------

@router.post("/cluster", response_model=ClusteringResponse)
async def cluster_concepts(
    db: AsyncSession = Depends(get_db),
) -> ClusteringResponse:
    """
    Cluster all concepts for the project using HDBSCAN (KMeans fallback).

    **Pipeline**

    1. Load concepts that have embeddings.
    2. L2-normalise embeddings; run HDBSCAN with adaptive ``min_cluster_size``
       (falls back to KMeans when HDBSCAN yields < 2 real clusters).
    3. Reassign noise points to the nearest cluster centroid.
    4. Recompute per-concept generality scores from doc/claim frequencies
       and cluster centrality.
    5. Build a 2-level hierarchy: highest-generality concept → cluster head
       (parent), all others → direct children.
    6. Detect cross-cluster bridge relationships (cosine > threshold).
    7. Persist ``cluster_label``, ``parent_concept_id``, ``generality_score``
       to the database and commit.

    Returns a full summary including per-cluster statistics.
    """
    clusterer = ConceptClusterer()
    try:
        result = await clusterer.cluster_project(settings.DEFAULT_PROJECT_ID, db)
    except Exception as exc:
        logger.exception("cluster_concepts: unexpected error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clustering failed: {exc}",
        )

    cluster_summaries: List[ClusterSummary] = [
        ClusterSummary(
            cluster_id=ci.cluster_id,
            label=ci.label,
            size=ci.size,
            parent_concept_id=ci.parent_concept_id,
            parent_concept_name=ci.parent_concept_name,
            avg_generality=ci.avg_generality,
        )
        for ci in result.clusters
    ]

    message = (
        f"Clustered {result.clustered_concepts} concepts into "
        f"{result.num_clusters} clusters using {result.algorithm_used}. "
        f"Reassigned {result.noise_reassigned} noise point(s). "
        f"Found {len(result.bridge_relationships)} cross-cluster bridge(s)."
    )
    logger.info(message)

    return ClusteringResponse(
        project_id=result.project_id,
        total_concepts=result.total_concepts,
        clustered_concepts=result.clustered_concepts,
        noise_reassigned=result.noise_reassigned,
        num_clusters=result.num_clusters,
        algorithm_used=result.algorithm_used,
        clusters=cluster_summaries,
        bridge_relationships_found=len(result.bridge_relationships),
        message=message,
    )


