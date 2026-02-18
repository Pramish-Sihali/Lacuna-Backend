"""
Pipeline orchestration endpoints.

Route summary
-------------
POST /process-all   — embed + extract all documents, then rebuild the knowledge graph.
POST /full-rebuild  — nuclear option: re-process every document and rebuild everything.
"""
from __future__ import annotations

import logging
import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import AsyncSessionLocal, get_db
from app.models.database_models import Document
from app.models.schemas import KnowledgeBuildResponse, PipelineProcessAllResponse
from app.services.pipeline import LacunaPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /process-all
# ---------------------------------------------------------------------------

@router.post(
    "/process-all",
    response_model=PipelineProcessAllResponse,
    status_code=status.HTTP_200_OK,
    summary="Process all documents and rebuild knowledge graph",
)
async def process_all(db: AsyncSession = Depends(get_db)) -> PipelineProcessAllResponse:
    """
    **Full batch pipeline** — process every document in the project and then
    rebuild the knowledge graph.

    For each document (in its own DB session to isolate failures):
    - Embed all un-embedded chunks.
    - Run LLM concept + claim + relationship extraction.

    After all documents have been processed:
    - Normalise concepts (dedup + merge).
    - Cluster via HDBSCAN.
    - Detect semantic relationships.
    - Detect knowledge gaps.
    - Compute consensus scores and build BrainState.

    Already-processed chunks and concepts are skipped via caching, so
    this endpoint is safe to call multiple times.

    Returns 200 with full statistics when complete.
    """
    docs_result = await db.execute(
        select(Document).where(Document.project_id == settings.DEFAULT_PROJECT_ID)
    )
    documents = docs_result.scalars().all()

    if not documents:
        return PipelineProcessAllResponse(
            project_id=settings.DEFAULT_PROJECT_ID,
            documents_processed=0,
            documents_failed=0,
            total_concepts=0,
            total_relationships=0,
            total_gaps=0,
            processing_time_seconds=0.0,
            errors=[],
            message="No documents found in the project.",
        )

    pipeline = LacunaPipeline()
    t0 = time.monotonic()
    docs_processed = 0
    docs_failed = 0
    errors: List[str] = []

    # ---- Per-document phase (fresh session per document) ----
    for doc in documents:
        async with AsyncSessionLocal() as doc_session:
            try:
                await pipeline.process_document(doc.id, doc_session)
                docs_processed += 1
                logger.info(
                    "process_all: ✓ document id=%d (%s)", doc.id, doc.filename
                )
            except Exception as exc:
                docs_failed += 1
                err_msg = f"doc id={doc.id} ({doc.filename}): {str(exc)[:120]}"
                errors.append(err_msg)
                logger.error("process_all: ✗ %s", err_msg)
                await doc_session.rollback()

    # ---- Project-level knowledge rebuild ----
    logger.info("process_all: starting knowledge rebuild …")
    try:
        knowledge = await pipeline.rebuild_project_knowledge(
            settings.DEFAULT_PROJECT_ID, db
        )
    except Exception as exc:
        logger.error("process_all: knowledge rebuild failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge rebuild failed after document processing: {exc}",
        )

    elapsed = round(time.monotonic() - t0, 2)
    message = (
        f"Processed {docs_processed}/{len(documents)} document(s) "
        f"({docs_failed} failed) in {elapsed}s. "
        f"Knowledge graph: {knowledge.num_clusters} cluster(s), "
        f"{knowledge.relationships_saved} relationship(s), "
        f"{knowledge.total_gaps} gap(s)."
    )
    if errors:
        message += f" See 'errors' for details on {len(errors)} failure(s)."

    logger.info("process_all: %s", message)

    return PipelineProcessAllResponse(
        project_id=settings.DEFAULT_PROJECT_ID,
        documents_processed=docs_processed,
        documents_failed=docs_failed,
        total_concepts=knowledge.concepts_after_normalize,
        total_relationships=knowledge.relationships_saved,
        total_gaps=knowledge.total_gaps,
        processing_time_seconds=elapsed,
        errors=errors,
        message=message,
    )


# ---------------------------------------------------------------------------
# POST /full-rebuild
# ---------------------------------------------------------------------------

@router.post(
    "/full-rebuild",
    response_model=KnowledgeBuildResponse,
    status_code=status.HTTP_200_OK,
    summary="Nuclear option: re-process all documents and fully rebuild knowledge",
)
async def full_rebuild(db: AsyncSession = Depends(get_db)) -> KnowledgeBuildResponse:
    """
    **Full rebuild** — re-embed and re-extract every document, then rebuild
    the knowledge graph from scratch.

    Use this when:
    - The embedding model has changed and existing vectors are stale.
    - Concept extraction prompts have been updated.
    - The project collection has changed significantly.

    **Warning:** This is the most expensive operation available.  For large
    collections it can take many minutes.

    Steps
    -----
    1. For each document: re-embed chunks + re-run LLM extraction.
    2. Normalise → cluster → relationships → gaps → consensus.

    Returns 200 with a full rebuild summary when complete.
    """
    pipeline = LacunaPipeline()
    t0 = time.monotonic()

    # ---- Re-process every document ----
    docs_result = await db.execute(
        select(Document).where(Document.project_id == settings.DEFAULT_PROJECT_ID)
    )
    documents = docs_result.scalars().all()

    for doc in documents:
        async with AsyncSessionLocal() as doc_session:
            try:
                await pipeline.process_document(doc.id, doc_session)
                logger.info(
                    "full_rebuild: ✓ document id=%d (%s)", doc.id, doc.filename
                )
            except Exception as exc:
                logger.warning(
                    "full_rebuild: ✗ document id=%d (%s): %s",
                    doc.id,
                    doc.filename,
                    exc,
                )
                await doc_session.rollback()

    # ---- Full knowledge rebuild ----
    logger.info("full_rebuild: starting full knowledge rebuild …")
    try:
        knowledge = await pipeline.rebuild_project_knowledge(
            settings.DEFAULT_PROJECT_ID, db
        )
    except Exception as exc:
        logger.error("full_rebuild: knowledge rebuild failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Full rebuild failed: {exc}",
        )

    elapsed = round(time.monotonic() - t0, 2)

    return KnowledgeBuildResponse(
        project_id=settings.DEFAULT_PROJECT_ID,
        concepts_before_normalize=knowledge.concepts_before_normalize,
        concepts_after_normalize=knowledge.concepts_after_normalize,
        merged_count=knowledge.merged_count,
        num_clusters=knowledge.num_clusters,
        algorithm_used=knowledge.algorithm_used,
        clustered_concepts=knowledge.clustered_concepts,
        relationships_found=knowledge.relationships_found,
        relationships_saved=knowledge.relationships_saved,
        expected_gaps=knowledge.expected_gaps,
        bridging_gaps=knowledge.bridging_gaps,
        weak_coverage_gaps=knowledge.weak_coverage_gaps,
        total_gaps=knowledge.total_gaps,
        concepts_scored=knowledge.concepts_scored,
        strong_consensus_count=knowledge.strong_consensus_count,
        contested_count=knowledge.contested_count,
        contradiction_count=knowledge.contradiction_count,
        brain_summary=knowledge.brain_summary,
        processing_time_seconds=elapsed,
        message=f"[Full Rebuild] {knowledge.message}",
    )
