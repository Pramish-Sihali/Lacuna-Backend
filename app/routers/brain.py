"""
Central brain / RAG endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging

from app.database import get_db
from app.config import settings
from app.models.database_models import BrainState, Concept, Chunk, Claim, Relationship
from app.models.schemas import (
    BrainQueryRequest,
    BrainQueryResponse,
    BrainStateResponse,
    ConceptResponse
)
from app.services.brain_service import BrainService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=BrainQueryResponse)
async def query_brain(
    request: BrainQueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Query the knowledge base using RAG.

    Args:
        request: Query request with query text and parameters
        db: Database session

    Returns:
        Query response with answer and relevant information
    """
    try:
        # Get all chunks with embeddings
        chunk_result = await db.execute(
            select(Chunk).where(Chunk.embedding.isnot(None))
        )
        chunks = chunk_result.scalars().all()

        # Get all concepts with embeddings
        concept_result = await db.execute(
            select(Concept)
            .where(Concept.project_id == settings.DEFAULT_PROJECT_ID)
            .where(Concept.embedding.isnot(None))
        )
        concepts = concept_result.scalars().all()

        if not chunks and not concepts:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No indexed content found. Please upload and process documents first."
            )

        # Convert to dictionaries
        chunk_dicts = [
            {
                "id": c.id,
                "content": c.content,
                "embedding": c.embedding,
                "document_id": c.document_id,
                "metadata_json": c.metadata_json
            }
            for c in chunks
        ]

        concept_dicts = [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "embedding": c.embedding,
                "is_gap": c.is_gap,
                "gap_type": c.gap_type,
                "coverage_score": c.coverage_score
            }
            for c in concepts
        ]

        # Query using brain service
        brain_service = BrainService()
        result = await brain_service.query(
            request.query,
            chunk_dicts,
            concept_dicts,
            top_k=request.top_k,
            include_gaps=request.include_gaps
        )

        # Convert relevant concepts to response schema
        relevant_concepts = [
            ConceptResponse(
                id=c["id"],
                project_id=settings.DEFAULT_PROJECT_ID,
                name=c["name"],
                description=c.get("description"),
                generality_score=None,
                coverage_score=c.get("coverage_score"),
                consensus_score=None,
                is_gap=c.get("is_gap", False),
                gap_type=c.get("gap_type"),
                parent_concept_id=None,
                cluster_label=None,
                metadata_json=None
            )
            for c in result["relevant_concepts"]
        ]

        # Extract document filenames
        relevant_doc_ids = set(
            chunk["document_id"]
            for chunk in result["relevant_chunks"]
            if chunk.get("document_id")
        )

        from app.models.database_models import Document
        doc_result = await db.execute(
            select(Document).where(Document.id.in_(relevant_doc_ids))
        )
        docs = doc_result.scalars().all()
        relevant_documents = [doc.filename for doc in docs]

        return BrainQueryResponse(
            query=result["query"],
            answer=result["answer"],
            relevant_concepts=relevant_concepts,
            relevant_documents=relevant_documents,
            confidence=result["confidence"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying brain: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.get("/consensus", response_model=BrainStateResponse)
async def get_consensus(db: AsyncSession = Depends(get_db)):
    """
    Get the current consensus state of the knowledge base.

    Args:
        db: Database session

    Returns:
        Brain state with consensus information
    """
    try:
        # Get existing brain state
        result = await db.execute(
            select(BrainState)
            .where(BrainState.project_id == settings.DEFAULT_PROJECT_ID)
            .order_by(BrainState.last_updated.desc())
        )
        brain_state = result.scalar_one_or_none()

        if brain_state:
            return BrainStateResponse.model_validate(brain_state)

        # If no brain state exists, create one
        return BrainStateResponse(
            id=0,
            project_id=settings.DEFAULT_PROJECT_ID,
            last_updated=None,
            summary_text="No consensus data available yet.",
            consensus_json={}
        )

    except Exception as e:
        logger.error(f"Error getting consensus: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving consensus: {str(e)}"
        )


@router.post("/build-consensus", response_model=BrainStateResponse)
async def build_consensus(db: AsyncSession = Depends(get_db)):
    """
    Build consensus from current knowledge base.

    Args:
        db: Database session

    Returns:
        Updated brain state
    """
    try:
        # Get all concepts
        concept_result = await db.execute(
            select(Concept).where(Concept.project_id == settings.DEFAULT_PROJECT_ID)
        )
        concepts = concept_result.scalars().all()

        # Get all claims
        claim_result = await db.execute(select(Claim))
        all_claims = claim_result.scalars().all()

        # Group claims by concept
        claims_by_concept = {}
        for claim in all_claims:
            if claim.concept_id not in claims_by_concept:
                claims_by_concept[claim.concept_id] = []
            claims_by_concept[claim.concept_id].append({
                "claim_text": claim.claim_text,
                "claim_type": claim.claim_type,
                "confidence": claim.confidence
            })

        # Get relationships
        rel_result = await db.execute(select(Relationship))
        relationships = rel_result.scalars().all()

        # Convert to dictionaries
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

        # Build consensus using brain service
        brain_service = BrainService()
        consensus = await brain_service.build_consensus(
            concept_dicts,
            claims_by_concept,
            relationship_dicts
        )

        # Update or create brain state
        result = await db.execute(
            select(BrainState)
            .where(BrainState.project_id == settings.DEFAULT_PROJECT_ID)
            .order_by(BrainState.last_updated.desc())
        )
        brain_state = result.scalar_one_or_none()

        if brain_state:
            brain_state.summary_text = consensus["summary"]
            brain_state.consensus_json = {
                "high_consensus_concepts": consensus["high_consensus_concepts"],
                "areas_of_disagreement": consensus["areas_of_disagreement"],
                "total_concepts": consensus["total_concepts"],
                "avg_consensus": consensus["avg_consensus"]
            }
        else:
            brain_state = BrainState(
                project_id=settings.DEFAULT_PROJECT_ID,
                summary_text=consensus["summary"],
                consensus_json={
                    "high_consensus_concepts": consensus["high_consensus_concepts"],
                    "areas_of_disagreement": consensus["areas_of_disagreement"],
                    "total_concepts": consensus["total_concepts"],
                    "avg_consensus": consensus["avg_consensus"]
                }
            )
            db.add(brain_state)

        # Update consensus scores for concepts
        for concept_info in consensus.get("high_consensus_concepts", []):
            concept = next((c for c in concepts if c.id == concept_info["id"]), None)
            if concept:
                concept.consensus_score = concept_info["score"]

        for concept_info in consensus.get("areas_of_disagreement", []):
            concept = next((c for c in concepts if c.id == concept_info["id"]), None)
            if concept:
                concept.consensus_score = concept_info["score"]

        await db.commit()
        await db.refresh(brain_state)

        logger.info("Consensus built successfully")

        return BrainStateResponse.model_validate(brain_state)

    except Exception as e:
        logger.error(f"Error building consensus: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error building consensus: {str(e)}"
        )
