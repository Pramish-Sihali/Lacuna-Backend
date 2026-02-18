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
from app.database import get_db
from app.models.database_models import Chunk, Document, Project
from app.models.schemas import DocumentResponse, DocumentUploadResponse
from app.services.chunking import ChunkingService
from app.services.document_parser import DocumentParser

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

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
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
