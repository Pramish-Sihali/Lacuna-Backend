"""
Document upload and management endpoints.
"""
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
import os
import logging
from pathlib import Path
import aiofiles

from app.database import get_db
from app.config import settings
from app.models.database_models import Document, Chunk, Project
from app.models.schemas import DocumentUploadResponse, DocumentResponse
from app.services.document_parser import DocumentParser
from app.services.chunking import ChunkingService
from app.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload and process a document.

    Args:
        file: Uploaded file (PDF or DOCX)
        db: Database session

    Returns:
        DocumentUploadResponse with upload status
    """
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Supported: {', '.join(settings.SUPPORTED_FILE_TYPES)}"
        )

    # Validate file size
    file_size = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)

    try:
        # Save file
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(chunk_size):
                file_size += len(chunk)
                if file_size > settings.MAX_FILE_SIZE:
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / 1024 / 1024}MB"
                    )
                await f.write(chunk)

        logger.info(f"Saved file: {file_path} ({file_size} bytes)")

        # Parse document
        parser = DocumentParser()
        parsed_data = await parser.parse_document(file_path, file_ext)

        # Get or create default project
        project = await db.execute(
            select(Project).where(Project.id == settings.DEFAULT_PROJECT_ID)
        )
        project = project.scalar_one_or_none()

        if not project:
            # Create default project
            project = Project(
                id=settings.DEFAULT_PROJECT_ID,
                name="Default Project",
                description="Default project for uploaded documents"
            )
            db.add(project)
            await db.flush()

        # Create document record
        document = Document(
            project_id=settings.DEFAULT_PROJECT_ID,
            filename=file.filename,
            file_path=file_path,
            file_type=file_ext.replace(".", ""),
            content_text=parsed_data["text"],
            metadata_json=parsed_data["metadata"]
        )
        db.add(document)
        await db.flush()

        # Chunk document
        chunking_service = ChunkingService()
        chunks = await chunking_service.chunk_document(
            parsed_data["text"],
            metadata={"filename": file.filename}
        )

        # Generate embeddings for chunks
        embedding_service = EmbeddingService()
        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = await embedding_service.generate_embeddings_batch(chunk_texts)

        # Save chunks to database
        for chunk_data, embedding in zip(chunks, embeddings):
            chunk = Chunk(
                document_id=document.id,
                content=chunk_data["content"],
                chunk_index=chunk_data["chunk_index"],
                embedding=embedding,
                metadata_json=chunk_data.get("metadata")
            )
            db.add(chunk)

        await db.commit()

        logger.info(f"Document {file.filename} processed successfully with {len(chunks)} chunks")

        return DocumentUploadResponse(
            id=document.id,
            filename=document.filename,
            file_type=document.file_type,
            status="processed",
            message=f"Document uploaded and processed successfully. Created {len(chunks)} chunks."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        # Clean up file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """
    List all documents.

    Args:
        skip: Number of documents to skip
        limit: Maximum number of documents to return
        db: Database session

    Returns:
        List of documents
    """
    result = await db.execute(
        select(Document)
        .where(Document.project_id == settings.DEFAULT_PROJECT_ID)
        .offset(skip)
        .limit(limit)
        .order_by(Document.created_at.desc())
    )
    documents = result.scalars().all()

    # Get chunk counts for each document
    response_documents = []
    for doc in documents:
        chunk_result = await db.execute(
            select(Chunk).where(Chunk.document_id == doc.id)
        )
        chunks = chunk_result.scalars().all()

        response_documents.append(
            DocumentResponse(
                id=doc.id,
                project_id=doc.project_id,
                filename=doc.filename,
                file_type=doc.file_type,
                content_text=doc.content_text,
                metadata_json=doc.metadata_json,
                created_at=doc.created_at,
                chunk_count=len(chunks)
            )
        )

    return response_documents


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific document by ID.

    Args:
        document_id: Document ID
        db: Database session

    Returns:
        Document details
    """
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Get chunk count
    chunk_result = await db.execute(
        select(Chunk).where(Chunk.document_id == document.id)
    )
    chunks = chunk_result.scalars().all()

    return DocumentResponse(
        id=document.id,
        project_id=document.project_id,
        filename=document.filename,
        file_type=document.file_type,
        content_text=document.content_text,
        metadata_json=document.metadata_json,
        created_at=document.created_at,
        chunk_count=len(chunks)
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a document and its associated chunks.

    Args:
        document_id: Document ID
        db: Database session
    """
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Delete file from filesystem
    if os.path.exists(document.file_path):
        try:
            os.remove(document.file_path)
        except Exception as e:
            logger.warning(f"Failed to delete file {document.file_path}: {e}")

    # Delete from database (cascades to chunks)
    await db.delete(document)
    await db.commit()

    logger.info(f"Deleted document {document_id}")
