"""
Semantic similarity search endpoint.

GET /api/search/similar  — embed a query text and return the top-K most
                           similar document chunks via pgvector.
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.schemas import SimilarChunkResult, SimilarChunksResponse
from app.services.embedding import OllamaEmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/similar", response_model=SimilarChunksResponse)
async def find_similar_chunks(
    query: str = Query(..., min_length=1, description="Natural-language query text"),
    top_k: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    threshold: float = Query(
        0.7,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score (0–1)",
    ),
    document_id: int | None = Query(
        None,
        description="Restrict search to a single document (optional)",
    ),
    db: AsyncSession = Depends(get_db),
) -> SimilarChunksResponse:
    """
    Embed *query*, then return the *top_k* most semantically similar chunks
    stored in PostgreSQL, ranked by cosine similarity (highest first).

    ### Parameters
    | name        | default | notes                                      |
    |-------------|---------|---------------------------------------------|
    | query       | —       | required; the text to search for            |
    | top_k       | 10      | max results returned (1–100)                |
    | threshold   | 0.7     | minimum similarity; lower = broader results |
    | document_id | null    | optional; restrict to one document          |

    ### Response
    Each result includes the chunk text, its position in the document,
    similarity score, and document metadata (filename, file_type).
    """
    svc = OllamaEmbeddingService()

    # Embed the query
    query_embedding = await svc.embed_text(query)
    if query_embedding is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Could not generate an embedding for the query. "
                "Check that Ollama is running and the embedding model is loaded."
            ),
        )

    # Vector search via pgvector
    raw_results = await svc.find_similar_chunks(
        query_embedding=query_embedding,
        db=db,
        top_k=top_k,
        threshold=threshold,
        document_id=document_id,
    )

    results: List[SimilarChunkResult] = [
        SimilarChunkResult(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            chunk_index=r["chunk_index"],
            content=r["content"],
            similarity=r["similarity"],
            filename=r["filename"],
            file_type=r["file_type"],
            metadata_json=r["metadata_json"],
        )
        for r in raw_results
    ]

    logger.info(
        "find_similar_chunks: query=%r → %d results (threshold=%.2f, top_k=%d)",
        query[:80],
        len(results),
        threshold,
        top_k,
    )

    return SimilarChunksResponse(
        query=query,
        results=results,
        total_results=len(results),
    )
