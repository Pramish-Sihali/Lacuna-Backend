"""
Brain / RAG endpoints.

Routes
------
POST /api/brain/build    — compute consensus scores + LLM summary → BrainBuildResponse
POST /api/brain/chat     — RAG chat                               → BrainChatResponse
GET  /api/brain/status   — brain state info                       → BrainStatusResponse
POST /api/brain/rebuild  — clear + full rebuild                   → BrainBuildResponse
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.database_models import (
    BrainState,
    Concept,
    Document,
    Relationship,
)
from app.models.schemas import (
    BrainBuildResponse,
    BrainChatRequest,
    BrainChatResponse,
    BrainStatusResponse,
)
from app.services.brain_service import BrainService

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /build — compute consensus + generate brain summary
# ---------------------------------------------------------------------------

@router.post("/build", response_model=BrainBuildResponse, status_code=status.HTTP_200_OK)
async def build_brain(db: AsyncSession = Depends(get_db)):
    """
    Score every concept's consensus, generate an LLM synthesis summary,
    and persist a BrainState row.  Idempotent — calling it again updates
    the existing BrainState in-place.
    """
    try:
        brain_service = BrainService()
        result = await brain_service.build_brain(
            settings.DEFAULT_PROJECT_ID, db, clear_existing=False
        )
        return BrainBuildResponse(
            project_id=result.project_id,
            concepts_scored=result.concepts_scored,
            strong_consensus_count=result.strong_consensus_count,
            contested_count=result.contested_count,
            contradiction_count=result.contradiction_count,
            summary_text=result.summary_text,
            message=result.message,
        )
    except Exception as exc:
        logger.error("build_brain error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Brain build failed: {exc}",
        )


# ---------------------------------------------------------------------------
# POST /chat — RAG chat
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=BrainChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: BrainChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Answer a research question using the RAG pipeline:
    embed question → pgvector chunk search → concept lookup →
    claim/gap context → LLM answer.
    """
    try:
        brain_service = BrainService()
        result = await brain_service.chat(
            request.question,
            settings.DEFAULT_PROJECT_ID,
            db,
            top_k=request.top_k,
        )
        return BrainChatResponse(
            question=result.question,
            answer=result.answer,
            sources=result.sources,
            relevant_concepts=result.relevant_concepts,
            confidence=result.confidence,
        )
    except Exception as exc:
        logger.error("chat error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {exc}",
        )


# ---------------------------------------------------------------------------
# GET /status — brain state info
# ---------------------------------------------------------------------------

@router.get("/status", response_model=BrainStatusResponse, status_code=status.HTTP_200_OK)
async def get_brain_status(db: AsyncSession = Depends(get_db)):
    """
    Return a snapshot of the knowledge base health:
    document count, concept count, gap count, relationship count,
    average consensus score, and a derived health score.
    """
    try:
        project_id = settings.DEFAULT_PROJECT_ID

        # Latest BrainState
        brain_result = await db.execute(
            select(BrainState)
            .where(BrainState.project_id == project_id)
            .order_by(BrainState.last_updated.desc())
            .limit(1)
        )
        brain_state: Optional[BrainState] = brain_result.scalar_one_or_none()

        # Document count
        doc_count_result = await db.execute(
            select(func.count())
            .select_from(Document)
            .where(Document.project_id == project_id)
        )
        doc_count: int = doc_count_result.scalar() or 0

        # Concept count (non-gap)
        concept_count_result = await db.execute(
            select(func.count())
            .select_from(Concept)
            .where(Concept.project_id == project_id, Concept.is_gap == False)
        )
        concept_count: int = concept_count_result.scalar() or 0

        # Gap count
        gap_count_result = await db.execute(
            select(func.count())
            .select_from(Concept)
            .where(Concept.project_id == project_id, Concept.is_gap == True)
        )
        gap_count: int = gap_count_result.scalar() or 0

        # Relationship count
        rel_count_result = await db.execute(
            select(func.count())
            .select_from(Relationship)
            .join(Concept, Relationship.source_concept_id == Concept.id)
            .where(Concept.project_id == project_id)
        )
        relationship_count: int = rel_count_result.scalar() or 0

        # Average consensus score across non-gap concepts
        avg_result = await db.execute(
            select(func.avg(Concept.consensus_score))
            .where(
                Concept.project_id == project_id,
                Concept.is_gap == False,
                Concept.consensus_score.isnot(None),
            )
        )
        avg_consensus: Optional[float] = avg_result.scalar()

        # Health score heuristic (0–1)
        #   40 % — average consensus (0.5 neutral when no data)
        #   30 % — document coverage (saturates at 10 docs)
        #   30 % — concept richness  (saturates at 50 concepts)
        consensus_component = float(avg_consensus) if avg_consensus is not None else 0.5
        doc_component = min(doc_count / 10, 1.0)
        concept_component = min(concept_count / 50, 1.0)
        health_score = round(
            0.40 * consensus_component
            + 0.30 * doc_component
            + 0.30 * concept_component,
            4,
        )

        return BrainStatusResponse(
            project_id=project_id,
            last_updated=brain_state.last_updated if brain_state else None,
            doc_count=doc_count,
            concept_count=concept_count,
            gap_count=gap_count,
            relationship_count=relationship_count,
            avg_consensus=round(float(avg_consensus), 4) if avg_consensus is not None else None,
            health_score=health_score,
            summary_text=brain_state.summary_text if brain_state else None,
            has_brain=brain_state is not None,
        )

    except Exception as exc:
        logger.error("get_brain_status error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status check failed: {exc}",
        )


# ---------------------------------------------------------------------------
# POST /rebuild — clear existing state and do a full rebuild
# ---------------------------------------------------------------------------

@router.post("/rebuild", response_model=BrainBuildResponse, status_code=status.HTTP_200_OK)
async def rebuild_brain(db: AsyncSession = Depends(get_db)):
    """
    Delete the existing BrainState, reset all concept consensus scores,
    then run the full build pipeline from scratch.  Use when the document
    collection has changed significantly.
    """
    try:
        # Reset all consensus scores so stale values don't persist
        concept_result = await db.execute(
            select(Concept).where(Concept.project_id == settings.DEFAULT_PROJECT_ID)
        )
        for concept in concept_result.scalars().all():
            concept.consensus_score = None
        await db.flush()

        brain_service = BrainService()
        result = await brain_service.build_brain(
            settings.DEFAULT_PROJECT_ID, db, clear_existing=True
        )
        return BrainBuildResponse(
            project_id=result.project_id,
            concepts_scored=result.concepts_scored,
            strong_consensus_count=result.strong_consensus_count,
            contested_count=result.contested_count,
            contradiction_count=result.contradiction_count,
            summary_text=result.summary_text,
            message="[Rebuild] " + result.message,
        )
    except Exception as exc:
        logger.error("rebuild_brain error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Brain rebuild failed: {exc}",
        )
