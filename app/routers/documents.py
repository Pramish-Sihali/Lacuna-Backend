"""
Document upload and management endpoints.

POST /upload  — parse + chunk a PDF or DOCX; embeddings generated separately.
GET  /        — list documents with chunk counts.
GET  /{id}    — document metadata + chunk count.
DELETE /{id}  — delete document, chunks, and file from disk.
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db, AsyncSessionLocal
from app.models.database_models import Chunk, Claim, Concept, Document, Project
from app.models.schemas import (
    ClaimResponse,
    ConceptResponse,
    DocumentResponse,
    DocumentUploadResponse,
    EmbedDocumentResponse,
    EmbedAllResponse,
    ExtractionResponse,
    ProcessDocumentResponse,
)
from app.services.chunking import ChunkingService
from app.services.document_parser import DocumentParser
from app.services.embedding import OllamaEmbeddingService
from app.services.llm_extractor import OllamaLLMService

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> DocumentUploadResponse:
    """
    Upload a PDF or DOCX, parse its text, and split it into chunks.

    - Max file size: 50 MB (configurable via MAX_FILE_SIZE)
    - File is stored with a UUID filename to avoid collisions
    - Embeddings are **not** generated here; use POST /api/concepts/extract next
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload must include a filename.",
        )

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{file_ext}'. "
                f"Accepted: {', '.join(settings.SUPPORTED_FILE_TYPES)}"
            ),
        )

    # Use a UUID-based name on disk to prevent collisions
    stored_name = f"{uuid.uuid4().hex}{file_ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, stored_name)
    file_size = 0

    try:
        # Stream to disk while enforcing the size limit
        async with aiofiles.open(file_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)   # 1 MB slices
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > settings.MAX_FILE_SIZE:
                    # Clean up partial file before raising
                    await out.close()
                    _safe_remove(file_path)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"File exceeds the {settings.MAX_FILE_SIZE // (1024 * 1024)} MB "
                            "size limit."
                        ),
                    )
                await out.write(chunk)

        logger.info(
            f"Saved {file.filename!r} → {file_path} ({file_size:,} bytes)"
        )

        # Parse
        parser = DocumentParser()
        try:
            parsed_doc = await parser.parse_document(file_path, file_ext)
        except RuntimeError as exc:
            _safe_remove(file_path)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )

        if not parsed_doc.full_text.strip():
            _safe_remove(file_path)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Document contains no extractable text.",
            )

        # Ensure the default project exists
        proj_result = await db.execute(
            select(Project).where(Project.id == settings.DEFAULT_PROJECT_ID)
        )
        project = proj_result.scalar_one_or_none()
        if not project:
            project = Project(
                id=settings.DEFAULT_PROJECT_ID,
                name="Default Project",
                description="Default project for uploaded documents",
            )
            db.add(project)
            await db.flush()

        # Persist document record (keep the original display name)
        document = Document(
            project_id=settings.DEFAULT_PROJECT_ID,
            filename=file.filename,          # human-readable display name
            file_path=file_path,             # UUID-based path on disk
            file_type=file_ext.lstrip("."),
            content_text=parsed_doc.full_text,
            metadata_json=parsed_doc.metadata,
        )
        db.add(document)
        await db.flush()   # populate document.id before creating chunks

        # Chunk (section-aware because we pass the ParsedDocument)
        # Embeddings will be generated separately in the concepts/extract step
        chunker = ChunkingService()
        chunks = await chunker.chunk_document(
            parsed_doc,
            metadata={"filename": file.filename},
        )

        for chunk_data in chunks:
            db.add(
                Chunk(
                    document_id=document.id,
                    content=chunk_data["content"],
                    chunk_index=chunk_data["chunk_index"],
                    embedding=None,           # filled in by the embedding step
                    metadata_json=chunk_data.get("metadata"),
                )
            )

        await db.commit()

        logger.info(
            f"Document {file.filename!r} stored as id={document.id} "
            f"with {len(chunks)} chunks."
        )

        return DocumentUploadResponse(
            id=document.id,
            filename=document.filename,
            file_type=document.file_type,
            status="processed",
            message=(
                f"Document uploaded and parsed successfully. "
                f"{len(chunks)} chunks created."
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Unexpected error processing {file.filename!r}")
        _safe_remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {exc}",
        )


# ---------------------------------------------------------------------------
# Embed all un-embedded chunks across every document
# ---------------------------------------------------------------------------

@router.post("/embed-all", response_model=EmbedAllResponse)
async def embed_all_documents(
    db: AsyncSession = Depends(get_db),
) -> EmbedAllResponse:
    """
    Embed every un-embedded chunk across all documents in the default project.

    Processes documents sequentially (Ollama is the bottleneck; within each
    document up to 3 concurrent requests are made via the semaphore).

    Returns the total number of chunks embedded this run.
    """
    docs_result = await db.execute(
        select(Document).where(Document.project_id == settings.DEFAULT_PROJECT_ID)
    )
    documents = docs_result.scalars().all()

    if not documents:
        return EmbedAllResponse(
            documents_processed=0,
            total_embedded=0,
            total_chunks=0,
            message="No documents found in the project.",
        )

    svc = OllamaEmbeddingService()
    docs_processed = 0
    grand_embedded = 0
    grand_total = 0

    for doc in documents:
        # Use a fresh session per document so a single failure doesn't
        # roll back the entire batch.
        async with AsyncSessionLocal() as doc_session:
            try:
                embedded, total = await svc.embed_document_chunks(doc.id, doc_session)
                grand_embedded += embedded
                grand_total += total
                docs_processed += 1
            except Exception as exc:
                logger.error(
                    "embed_all_documents: failed on document id=%d (%s): %s",
                    doc.id,
                    doc.filename,
                    exc,
                )
                await doc_session.rollback()

    logger.info(
        "embed_all_documents: %d docs processed, %d/%d chunks embedded",
        docs_processed,
        grand_embedded,
        grand_total,
    )
    return EmbedAllResponse(
        documents_processed=docs_processed,
        total_embedded=grand_embedded,
        total_chunks=grand_total,
        message=(
            f"Embedded {grand_embedded} of {grand_total} chunks "
            f"across {docs_processed} document(s)."
        ),
    )


# ---------------------------------------------------------------------------
# Embed all chunks for a single document
# ---------------------------------------------------------------------------

@router.post("/{document_id}/embed", response_model=EmbedDocumentResponse)
async def embed_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> EmbedDocumentResponse:
    """
    Embed all un-embedded chunks for a specific document.

    Already-embedded chunks are skipped (cache hit).  After embedding,
    an averaged document-level vector is stored in ``metadata_json``.

    Returns the number of chunks embedded and the total chunk count.
    """
    # Verify the document exists and belongs to the default project
    doc_result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.project_id == settings.DEFAULT_PROJECT_ID,
        )
    )
    document = doc_result.scalar_one_or_none()
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found.",
        )

    svc = OllamaEmbeddingService()
    embedded, total = await svc.embed_document_chunks(document_id, db)

    if total == 0:
        embed_status = "no_chunks"
        msg = "Document has no chunks to embed."
    elif embedded == total:
        embed_status = "completed"
        msg = f"All {total} chunks embedded successfully."
    else:
        embed_status = "partial"
        msg = f"{embedded} of {total} chunks embedded (some may have failed)."

    logger.info(
        "embed_document: document id=%d — %s (%d/%d)",
        document_id,
        embed_status,
        embedded,
        total,
    )
    return EmbedDocumentResponse(
        document_id=document_id,
        embedded_chunks=embedded,
        total_chunks=total,
        status=embed_status,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Extract concepts + claims for a single document
# ---------------------------------------------------------------------------

@router.post("/{document_id}/extract", response_model=ExtractionResponse)
async def extract_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> ExtractionResponse:
    """
    Run LLM-based concept and claim extraction for a document.

    Must be called **after** the document has been embedded
    (`POST /{document_id}/embed`).  Existing concepts that already appear in
    the project (matched by normalised name or embedding similarity ≥ 0.85)
    are reused rather than duplicated.

    Returns a summary of what was extracted and persisted.
    """
    doc_result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.project_id == settings.DEFAULT_PROJECT_ID,
        )
    )
    if doc_result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found.",
        )

    svc = OllamaLLMService()
    try:
        summary = await svc.process_document(document_id, db)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        )
    except Exception as exc:
        logger.exception("extract_document: unexpected error for id=%d", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {exc}",
        )

    status_msg = (
        f"Extracted {summary.concepts_saved} new concepts and "
        f"{summary.claims_saved} claims from "
        f"{summary.chunks_processed} chunks."
    )
    if summary.errors:
        status_msg += f" {len(summary.errors)} chunk(s) had errors."

    return ExtractionResponse(
        document_id=summary.document_id,
        document_title=summary.document_title,
        chunks_processed=summary.chunks_processed,
        chunks_skipped=summary.chunks_skipped,
        concepts_extracted=summary.concepts_extracted,
        concepts_saved=summary.concepts_saved,
        claims_saved=summary.claims_saved,
        relationships_found=summary.relationships_found,
        errors=summary.errors,
        message=status_msg,
    )


# ---------------------------------------------------------------------------
# Full pipeline: embed → extract
# ---------------------------------------------------------------------------

@router.post("/{document_id}/process", response_model=ProcessDocumentResponse)
async def process_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> ProcessDocumentResponse:
    """
    **Full pipeline** for a single document: embed all chunks, then extract
    concepts and claims.

    This is the primary endpoint the frontend should call after uploading a
    document.  It combines `POST /{document_id}/embed` and
    `POST /{document_id}/extract` into one call.

    Steps
    -----
    1. Verify the document exists.
    2. Embed all un-embedded chunks (skips already-embedded ones).
    3. Run LLM concept + claim extraction over every chunk.
    4. Return a combined progress report.
    """
    doc_result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.project_id == settings.DEFAULT_PROJECT_ID,
        )
    )
    document = doc_result.scalar_one_or_none()
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found.",
        )

    # --- phase 1: embed ---
    embed_svc = OllamaEmbeddingService()
    try:
        embedded_chunks, total_chunks = await embed_svc.embed_document_chunks(
            document_id, db
        )
    except Exception as exc:
        logger.exception("process_document: embed phase failed for id=%d", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding phase failed: {exc}",
        )

    # --- phase 2: extract ---
    llm_svc = OllamaLLMService()
    try:
        summary = await llm_svc.process_document(document_id, db)
    except Exception as exc:
        logger.exception(
            "process_document: extraction phase failed for id=%d", document_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction phase failed: {exc}",
        )

    message = (
        f"Processed document '{document.filename}': "
        f"embedded {embedded_chunks}/{total_chunks} chunks, "
        f"saved {summary.concepts_saved} concepts and "
        f"{summary.claims_saved} claims."
    )
    if summary.errors:
        message += f" {len(summary.errors)} chunk(s) had extraction errors."

    logger.info("process_document id=%d: %s", document_id, message)

    return ProcessDocumentResponse(
        document_id=document_id,
        document_title=document.filename,
        embedded_chunks=embedded_chunks,
        total_chunks=total_chunks,
        chunks_processed=summary.chunks_processed,
        chunks_skipped=summary.chunks_skipped,
        concepts_extracted=summary.concepts_extracted,
        concepts_saved=summary.concepts_saved,
        claims_saved=summary.claims_saved,
        relationships_found=summary.relationships_found,
        errors=summary.errors,
        message=message,
    )


# ---------------------------------------------------------------------------
# Concepts extracted from a document
# ---------------------------------------------------------------------------

@router.get("/{document_id}/concepts", response_model=List[ConceptResponse])
async def get_document_concepts(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> List[ConceptResponse]:
    """
    Return all concepts that were extracted from this document.

    Concepts are project-scoped; this endpoint filters to those linked to the
    document via at least one Claim row (i.e., the document asserts something
    about those concepts).
    """
    doc_result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    if doc_result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found.",
        )

    # Concepts that appear in at least one Claim for this document
    concept_ids_result = await db.execute(
        select(Claim.concept_id)
        .where(Claim.document_id == document_id)
        .distinct()
    )
    concept_ids = [row[0] for row in concept_ids_result.all()]

    if not concept_ids:
        return []

    concepts_result = await db.execute(
        select(Concept).where(Concept.id.in_(concept_ids))
    )
    concepts = concepts_result.scalars().all()

    return [ConceptResponse.model_validate(c) for c in concepts]


# ---------------------------------------------------------------------------
# Claims extracted from a document
# ---------------------------------------------------------------------------

@router.get("/{document_id}/claims", response_model=List[ClaimResponse])
async def get_document_claims(
    document_id: int,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
) -> List[ClaimResponse]:
    """
    Return all claims extracted from this document, with pagination.

    Claims link the document to specific concepts and describe *how* the
    document relates to each concept (supports / contradicts / extends /
    complements).
    """
    doc_result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    if doc_result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found.",
        )

    claims_result = await db.execute(
        select(Claim)
        .where(Claim.document_id == document_id)
        .offset(skip)
        .limit(limit)
    )
    claims = claims_result.scalars().all()

    return [ClaimResponse.model_validate(c) for c in claims]


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
) -> List[DocumentResponse]:
    """
    List all documents in the default project, newest first.

    Supports pagination via `skip` and `limit` query parameters.
    """
    docs_result = await db.execute(
        select(Document)
        .where(Document.project_id == settings.DEFAULT_PROJECT_ID)
        .order_by(Document.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    documents = docs_result.scalars().all()

    # Fetch chunk counts for all documents in a single GROUP BY query
    doc_ids = [d.id for d in documents]
    counts: Dict[int, int] = {}
    if doc_ids:
        counts_result = await db.execute(
            select(Chunk.document_id, func.count(Chunk.id).label("cnt"))
            .where(Chunk.document_id.in_(doc_ids))
            .group_by(Chunk.document_id)
        )
        counts = {row.document_id: row.cnt for row in counts_result}

    return [
        DocumentResponse(
            id=doc.id,
            project_id=doc.project_id,
            filename=doc.filename,
            file_type=doc.file_type,
            content_text=doc.content_text,
            metadata_json=doc.metadata_json,
            created_at=doc.created_at,
            chunk_count=counts.get(doc.id, 0),
        )
        for doc in documents
    ]


# ---------------------------------------------------------------------------
# Get by ID
# ---------------------------------------------------------------------------

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Return metadata and chunk count for a single document."""
    doc_result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = doc_result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    count_result = await db.execute(
        select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
    )
    chunk_count: int = count_result.scalar_one() or 0

    return DocumentResponse(
        id=document.id,
        project_id=document.project_id,
        filename=document.filename,
        file_type=document.file_type,
        content_text=document.content_text,
        metadata_json=document.metadata_json,
        created_at=document.created_at,
        chunk_count=chunk_count,
    )


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Delete a document, all its chunks, and its file from disk.

    Chunk and claim deletion is handled by the CASCADE constraint on the
    documents foreign key.
    """
    doc_result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = doc_result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    _safe_remove(document.file_path)

    await db.delete(document)
    await db.commit()

    logger.info(f"Deleted document id={document_id} ({document.filename!r})")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _safe_remove(path: str) -> None:
    """Delete a file silently, logging warnings but never raising."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as exc:
        logger.warning(f"Could not remove file {path!r}: {exc}")
