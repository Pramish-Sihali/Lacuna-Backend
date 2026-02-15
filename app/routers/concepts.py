"""
Concept map and analysis endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Dict
import logging

from app.database import get_db
from app.config import settings
from app.models.database_models import Concept, Relationship, Claim, Document, Chunk
from app.models.schemas import ConceptResponse, ConceptMapResponse, ConceptNode, ConceptEdge
from app.services.embedding import EmbeddingService
from app.services.llm_extractor import LLMExtractor
from app.services.clustering import ClusteringService
from app.services.relationships import RelationshipService
from app.services.gap_detector import GapDetector
from app.services.normalizer import ConceptNormalizer

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/extract", status_code=status.HTTP_202_ACCEPTED)
async def extract_concepts(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Extract concepts from all documents in the background.

    Args:
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        Status message
    """
    background_tasks.add_task(process_concept_extraction, db)

    return {
        "message": "Concept extraction started in background",
        "status": "processing"
    }


async def process_concept_extraction(db: AsyncSession):
    """Background task to extract concepts from documents."""
    try:
        logger.info("Starting concept extraction...")

        # Get all documents
        result = await db.execute(
            select(Document).where(Document.project_id == settings.DEFAULT_PROJECT_ID)
        )
        documents = result.scalars().all()

        if not documents:
            logger.warning("No documents found for concept extraction")
            return

        # Initialize services
        llm_extractor = LLMExtractor()
        embedding_service = EmbeddingService()
        normalizer = ConceptNormalizer()

        all_extracted_concepts = []

        # Extract concepts from each document
        for document in documents:
            logger.info(f"Extracting concepts from {document.filename}")

            # Extract concepts using LLM
            concepts = await llm_extractor.extract_concepts(document.content_text)

            for concept_data in concepts:
                # Generate embedding for concept
                concept_embedding = await embedding_service.generate_embedding(
                    f"{concept_data['name']} {concept_data.get('description', '')}"
                )

                all_extracted_concepts.append({
                    **concept_data,
                    "embedding": concept_embedding,
                    "document_id": document.id
                })

        # Normalize and deduplicate concepts
        logger.info(f"Normalizing {len(all_extracted_concepts)} concepts...")
        normalized_concepts = await normalizer.normalize_concepts(all_extracted_concepts)

        # Save to database
        for concept_data in normalized_concepts:
            concept = Concept(
                project_id=settings.DEFAULT_PROJECT_ID,
                name=concept_data["name"],
                description=concept_data.get("description"),
                generality_score=concept_data.get("generality_score", 0.5),
                embedding=concept_data.get("embedding"),
                coverage_score=0.5,  # Will be updated later
                consensus_score=1.0,  # Default to full consensus
            )
            db.add(concept)

        await db.commit()

        logger.info(f"Concept extraction completed. Saved {len(normalized_concepts)} concepts.")

    except Exception as e:
        logger.error(f"Error in concept extraction: {e}")
        await db.rollback()


@router.get("/", response_model=List[ConceptResponse])
async def list_concepts(
    skip: int = 0,
    limit: int = 100,
    include_gaps_only: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    List all concepts.

    Args:
        skip: Number of concepts to skip
        limit: Maximum number to return
        include_gaps_only: Filter to only gap concepts
        db: Database session

    Returns:
        List of concepts
    """
    query = select(Concept).where(Concept.project_id == settings.DEFAULT_PROJECT_ID)

    if include_gaps_only:
        query = query.where(Concept.is_gap == True)

    query = query.offset(skip).limit(limit).order_by(Concept.coverage_score.desc())

    result = await db.execute(query)
    concepts = result.scalars().all()

    return [ConceptResponse.model_validate(c) for c in concepts]


@router.get("/map", response_model=ConceptMapResponse)
async def get_concept_map(db: AsyncSession = Depends(get_db)):
    """
    Get the complete concept map with nodes and edges.

    Args:
        db: Database session

    Returns:
        Concept map with nodes and edges
    """
    # Get all concepts
    concept_result = await db.execute(
        select(Concept).where(Concept.project_id == settings.DEFAULT_PROJECT_ID)
    )
    concepts = concept_result.scalars().all()

    # Get all relationships
    rel_result = await db.execute(select(Relationship))
    relationships = rel_result.scalars().all()

    # Build nodes
    nodes = []
    for concept in concepts:
        # Count children
        children_result = await db.execute(
            select(func.count(Concept.id)).where(Concept.parent_concept_id == concept.id)
        )
        children_count = children_result.scalar() or 0

        nodes.append(ConceptNode(
            id=concept.id,
            name=concept.name,
            description=concept.description,
            generality_score=concept.generality_score,
            coverage_score=concept.coverage_score,
            consensus_score=concept.consensus_score,
            is_gap=concept.is_gap,
            gap_type=concept.gap_type,
            cluster_label=concept.cluster_label,
            children_count=children_count
        ))

    # Build edges
    edges = []
    for rel in relationships:
        edges.append(ConceptEdge(
            source_id=rel.source_concept_id,
            target_id=rel.target_concept_id,
            relationship_type=rel.relationship_type,
            strength=rel.strength,
            confidence=rel.confidence
        ))

    # Calculate cluster distribution
    clusters = {}
    for concept in concepts:
        if concept.cluster_label is not None:
            clusters[concept.cluster_label] = clusters.get(concept.cluster_label, 0) + 1

    # Count gaps
    total_gaps = sum(1 for c in concepts if c.is_gap)

    return ConceptMapResponse(
        nodes=nodes,
        edges=edges,
        total_concepts=len(concepts),
        total_gaps=total_gaps,
        clusters=clusters
    )


@router.post("/cluster", status_code=status.HTTP_202_ACCEPTED)
async def cluster_concepts(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Cluster concepts using HDBSCAN in the background.

    Args:
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        Status message
    """
    background_tasks.add_task(process_clustering, db)

    return {
        "message": "Clustering started in background",
        "status": "processing"
    }


async def process_clustering(db: AsyncSession):
    """Background task to cluster concepts."""
    try:
        logger.info("Starting concept clustering...")

        # Get all concepts with embeddings
        result = await db.execute(
            select(Concept)
            .where(Concept.project_id == settings.DEFAULT_PROJECT_ID)
            .where(Concept.embedding.isnot(None))
        )
        concepts = result.scalars().all()

        if len(concepts) < settings.HDBSCAN_MIN_CLUSTER_SIZE:
            logger.warning(f"Not enough concepts for clustering: {len(concepts)}")
            return

        # Extract embeddings and IDs
        embeddings = [c.embedding for c in concepts]
        concept_ids = [c.id for c in concepts]
        generality_scores = [c.generality_score for c in concepts]

        # Perform clustering
        clustering_service = ClusteringService()
        cluster_result = await clustering_service.cluster_concepts(embeddings, concept_ids)

        # Update concepts with cluster labels
        for concept, label in zip(concepts, cluster_result["labels"]):
            concept.cluster_label = int(label) if label != -1 else None

        # Detect hierarchy
        hierarchy = await clustering_service.detect_hierarchy(
            cluster_result["labels"],
            embeddings,
            generality_scores
        )

        # Update parent relationships based on hierarchy
        for cluster_id, hierarchy_info in hierarchy["concept_hierarchies"].items():
            parent_concept_id = hierarchy_info["parent_concept"]
            children_indices = hierarchy_info["children_concepts"]

            for child_idx in children_indices:
                child_concept = concepts[child_idx]
                child_concept.parent_concept_id = concepts[parent_concept_id].id

        await db.commit()

        logger.info(f"Clustering completed for {len(concepts)} concepts")

    except Exception as e:
        logger.error(f"Error in clustering: {e}")
        await db.rollback()


@router.post("/detect-gaps", status_code=status.HTTP_202_ACCEPTED)
async def detect_gaps(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Detect knowledge gaps in the background.

    Args:
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        Status message
    """
    background_tasks.add_task(process_gap_detection, db)

    return {
        "message": "Gap detection started in background",
        "status": "processing"
    }


async def process_gap_detection(db: AsyncSession):
    """Background task to detect gaps."""
    try:
        logger.info("Starting gap detection...")

        # Get concepts
        concept_result = await db.execute(
            select(Concept).where(Concept.project_id == settings.DEFAULT_PROJECT_ID)
        )
        concepts = result.scalars().all()

        # Get relationships
        rel_result = await db.execute(select(Relationship))
        relationships = rel_result.scalars().all()

        # Get claims grouped by concept
        claim_result = await db.execute(select(Claim))
        all_claims = claim_result.scalars().all()

        claims_by_concept = {}
        for claim in all_claims:
            if claim.concept_id not in claims_by_concept:
                claims_by_concept[claim.concept_id] = []
            claims_by_concept[claim.concept_id].append({
                "claim_text": claim.claim_text,
                "claim_type": claim.claim_type,
                "confidence": claim.confidence
            })

        # Convert to dictionaries for gap detector
        concept_dicts = [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "embedding": c.embedding,
                "coverage_score": c.coverage_score,
                "consensus_score": c.consensus_score
            }
            for c in concepts
        ]

        relationship_dicts = [
            {
                "source_concept_id": r.source_concept_id,
                "target_concept_id": r.target_concept_id,
                "relationship_type": r.relationship_type,
                "strength": r.strength
            }
            for r in relationships
        ]

        # Detect gaps
        gap_detector = GapDetector()
        gaps = await gap_detector.detect_all_gaps(
            concept_dicts,
            relationship_dicts,
            claims_by_concept
        )

        # Score and prioritize gaps
        gaps = await gap_detector.score_gaps(gaps)

        # Mark concepts as gaps
        for gap in gaps:
            for concept_id in gap["concept_ids"]:
                concept = next((c for c in concepts if c.id == concept_id), None)
                if concept:
                    concept.is_gap = True
                    concept.gap_type = gap["gap_type"]

        await db.commit()

        logger.info(f"Gap detection completed. Found {len(gaps)} gaps")

    except Exception as e:
        logger.error(f"Error in gap detection: {e}")
        await db.rollback()
